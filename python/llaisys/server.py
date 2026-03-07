import os
import time
import uuid
import json
import asyncio
from typing import List, Optional, Union, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from transformers import AutoTokenizer

import llaisys
from llaisys import DeviceType

# ==============================================================================
# Global State & Configuration
# ==============================================================================

class InferenceRequest:
    def __init__(self, 
                 input_ids: List[int], 
                 generation_config: Dict[str, Any],
                 response_queue: asyncio.Queue,
                 session_id: Optional[str] = None):
        self.input_ids = input_ids # Remaining prompt tokens to be processed
        self.generation_config = generation_config
        self.response_queue = response_queue
        self.session_id = session_id
        self.created_at = time.time()
        self.generated_len = 0 # How many new tokens generated
        self.finished = False
        self.session = None # Assigned session object

class GlobalState:
    model: Optional[llaisys.models.Qwen2] = None
    tokenizer: Optional[AutoTokenizer] = None
    model_path: str = "/home/wsl/model/DeepSeek-R1-Distill-Qwen-1.5B"  # Default path
    session_store: Dict[str, Any] = {}
    request_queue: asyncio.Queue = None # Initialize in lifespan
    
state = GlobalState()

# Lock for model access (Project #3 Requirement)
# Even with worker, we need to protect model if multiple workers exist (though we use 1 worker for now)
generation_lock = asyncio.Lock()

# ==============================================================================
# Data Models (OpenAI Compatible)
# ==============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-r1-distill-qwen-1.5b"
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 50
    n: Optional[int] = 1
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    seed: Optional[int] = -1
    session_id: Optional[str] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Dict[str, int]

class ChatCompletionStreamResponseChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamResponseChoice]

# ==============================================================================
# Background Worker
# ==============================================================================

MAX_BATCH_SIZE = 8 # Configurable

async def worker_loop():
    print("Worker loop started.")
    
    # Active requests currently being processed
    active_requests: List[InferenceRequest] = []
    
    while True:
        try:
            # 1. Fetch new requests if we have capacity
            # We want to fill up to MAX_BATCH_SIZE
            while len(active_requests) < MAX_BATCH_SIZE:
                if state.request_queue.empty():
                    break
                try:
                    # Non-blocking fetch
                    req = state.request_queue.get_nowait()
                    active_requests.append(req)
                except asyncio.QueueEmpty:
                    break
            
            # If no requests, wait for one
            if not active_requests:
                req = await state.request_queue.get()
                active_requests.append(req)
            
            # 2. Prepare Batch
            sessions = []
            batch_input_ids = []
            top_ks = []
            top_ps = []
            temperatures = []
            seeds = []
            
            requests_to_remove = []
            
            for req in active_requests:
                # Initialize session if needed
                if req.session is None:
                    if req.session_id and req.session_id in state.session_store:
                        req.session = state.session_store[req.session_id]
                    else:
                        # Create new session
                        # Ideally lock here if needed, but model ops are safe now? 
                        # create_session is C++ allocation, should be fast.
                        req.session = state.model.create_session()
                        if req.session_id:
                            state.session_store[req.session_id] = req.session
                
                sessions.append(req.session)
                
                # Determine input for this step
                # If we have unprocessed prompt tokens, feed them (Prefill)
                # Note: Our current infer_batch only supports 1 token step per session effectively
                # because we didn't implement chunked prefill in C++ (it loops).
                # So we feed 1 token at a time even for prompt.
                # Optimization: In real system, we'd feed all prompt tokens at once.
                # Here, we treat prompt as just a sequence of tokens to force feed.
                
                if len(req.input_ids) > 0:
                    # Prefill phase: feed next token from prompt
                    # But wait, Qwen2Model::infer_batch implementation loops `step < batch_token_ids[b].size()`
                    # So we CAN feed multiple tokens!
                    # However, if we feed multiple, we get multiple outputs? 
                    # Our infer_batch implementation only samples ONE token at the end of the sequence provided.
                    # So for prefill, we should provide the WHOLE prompt, but we only care about the last output?
                    # NO. The prompt tokens are INPUTS. The model generates ONE token after them.
                    # So we should provide ALL remaining prompt tokens.
                    
                    next_tokens = req.input_ids
                    # We consume all prompt tokens in one go
                    req.input_ids = [] 
                    batch_input_ids.append(next_tokens)
                else:
                    # Decode phase: we rely on the PREVIOUS output token being in the KV cache?
                    # Wait, our generate_batch API expects INPUT tokens.
                    # If we just generated a token in previous step, we need to feed it back!
                    # BUT `generate_batch` returns the NEW token.
                    # We need to store it and feed it in next step.
                    
                    # ISSUE: `InferenceRequest` needs to store the `last_token` to feed in next step.
                    # In standard generate(), this is handled by the loop.
                    # Here we must manage it manually.
                    
                    if hasattr(req, 'last_token'):
                        batch_input_ids.append([req.last_token])
                    else:
                        # Should not happen if logic is correct
                        # Unless prompt was empty?
                        batch_input_ids.append([state.model._meta.end_token]) # Dummy?
                        
                top_ks.append(req.generation_config["top_k"])
                top_ps.append(req.generation_config["top_p"])
                temperatures.append(req.generation_config["temperature"])
                seeds.append(req.generation_config["seed"])
            
            # 3. Run Inference
            if not sessions:
                continue
                
            # This runs ONE forward pass for everyone
            # For those with multiple tokens (prompt), it processes them all and returns next token.
            new_tokens = state.model.generate_batch(
                sessions,
                batch_input_ids,
                top_ks,
                top_ps,
                temperatures,
                seeds
            )
            
            # 4. Process Results
            for i, req in enumerate(active_requests):
                new_token = new_tokens[i]
                req.last_token = new_token
                req.generated_len += 1
                
                # Send to client
                await req.response_queue.put({"token_id": new_token})
                
                # Check finish conditions
                is_eos = (new_token == state.model._meta.end_token)
                max_len_reached = (req.generated_len >= req.generation_config["max_new_tokens"])
                
                if is_eos or max_len_reached:
                    await req.response_queue.put({"done": True})
                    req.finished = True
                    requests_to_remove.append(req)
                    
                    # Mark task done in global queue if it came from there
                    # (Though we already took it out)
                    try:
                        state.request_queue.task_done()
                    except ValueError:
                        pass
            
            # 5. Remove finished requests
            for req in requests_to_remove:
                active_requests.remove(req)
                
            # Yield to event loop
            await asyncio.sleep(0)
            
        except asyncio.CancelledError:
            print("Worker loop cancelled.")
            break
        except Exception as e:
            print(f"Worker loop error: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(1)

# ==============================================================================
# Lifespan & App Initialization
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Queues
    state.request_queue = asyncio.Queue()

    # Startup: Load Model
    print(f"Loading tokenizer from {state.model_path}...")
    state.tokenizer = AutoTokenizer.from_pretrained(state.model_path, trust_remote_code=True)
    
    print(f"Loading LLAISYS model from {state.model_path}...")
    state.model = llaisys.models.Qwen2(state.model_path, DeviceType.CPU)
    
    print("Model loaded successfully!")

    # Start Worker
    worker_task = asyncio.create_task(worker_loop())

    yield
    
    # Shutdown: Clean up resources
    print("Shutting down...")
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="LLAISYS Chatbot Server", lifespan=lifespan)

# ==============================================================================
# Helper Functions
# ==============================================================================

def _format_sse(data: BaseModel) -> str:
    return data.model_dump_json()

# ==============================================================================
# API Endpoints
# ==============================================================================

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not state.model or not state.tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Prepare Prompt
    try:
        # Convert Pydantic models to dicts for apply_chat_template
        messages_list = [msg.model_dump() for msg in request.messages]
        prompt_text = state.tokenizer.apply_chat_template(
            messages_list, tokenize=False, add_generation_prompt=True
        )
        input_ids = state.tokenizer.encode(prompt_text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tokenization failed: {str(e)}")

    request_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())

    # 2. Create Request Object
    response_queue = asyncio.Queue()
    gen_config = {
        "max_new_tokens": request.max_tokens,
        "top_k": request.top_k,
        "top_p": request.top_p if request.top_p is not None else 0.0,
        "temperature": request.temperature,
        "seed": request.seed if request.seed is not None else -1
    }
    
    req = InferenceRequest(
        input_ids=input_ids,
        generation_config=gen_config,
        response_queue=response_queue,
        session_id=request.session_id
    )

    # 3. Put into Queue
    await state.request_queue.put(req)

    # 4. Stream Response (SSE)
    async def event_generator():
        # Send initial role
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=request.model,
            choices=[ChatCompletionStreamResponseChoice(
                index=0,
                delta={"role": "assistant"},
                finish_reason=None
            )]
        )
        yield {"data": _format_sse(chunk)}

        # Stream tokens from queue
        while True:
            item = await response_queue.get()
            
            if "error" in item:
                # Handle error (maybe send a special event or just log)
                print(f"Error from worker: {item['error']}")
                break
                
            if "done" in item:
                break
                
            token_id = item["token_id"]
            word = state.tokenizer.decode([token_id], skip_special_tokens=True)
            
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=request.model,
                choices=[ChatCompletionStreamResponseChoice(
                    index=0,
                    delta={"content": word},
                    finish_reason=None
                )]
            )
            yield {"data": _format_sse(chunk)}
            
        # Send finish
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=request.model,
            choices=[ChatCompletionStreamResponseChoice(
                index=0,
                delta={},
                finish_reason="stop"
            )]
        )
        yield {"data": _format_sse(chunk)}
        yield {"data": "[DONE]"}

    if request.stream:
        return EventSourceResponse(event_generator())
    else:
        # Non-stream: Accumulate all tokens
        content = ""
        while True:
            item = await response_queue.get()
            if "error" in item:
                raise HTTPException(status_code=500, detail=item["error"])
            if "done" in item:
                break
            token_id = item["token_id"]
            content += state.tokenizer.decode([token_id], skip_special_tokens=True)

        usage = {
            "prompt_tokens": len(input_ids),
            "completion_tokens": 0, # TODO: count tokens
            "total_tokens": 0
        }

        return ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=request.model,
            choices=[ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop"
            )],
            usage=usage
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
