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
        self.input_ids = input_ids
        self.generation_config = generation_config
        self.response_queue = response_queue
        self.session_id = session_id
        self.created_at = time.time()

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

async def worker_loop():
    print("Worker loop started.")
    while True:
        try:
            # 1. Get request from queue
            req: InferenceRequest = await state.request_queue.get()
            
            # 2. Process request (Project #4: Step 1 - Simple Serial Processing)
            # In future steps, we will batch multiple requests here.
            
            # Resolve session
            session = None
            if req.session_id:
                if req.session_id in state.session_store:
                    session = state.session_store[req.session_id]
                else:
                    # Create new session if not found (or should we error?)
                    # For robustness, let's create one.
                    # Note: Model access should be thread-safe now for sessions, 
                    # but model.create_session() might not be. Let's lock just in case or assume safe.
                    # Actually, create_session allocates memory, so it's better to be safe.
                    async with generation_lock:
                        session = state.model.create_session()
                        state.session_store[req.session_id] = session
            
            # Generate
            # We still use the lock to protect the model execution if needed,
            # although we made model stateless. 
            # However, for now, let's ensure serial execution as per P4-2 requirement "looping process".
            
            try:
                # We use stream=True always for the worker to push tokens back to queue
                stream_gen = state.model.generate(
                    req.input_ids,
                    **req.generation_config,
                    stream=True,
                    session=session
                )
                
                for token_id in stream_gen:
                    # Put token into response queue
                    await req.response_queue.put({"token_id": token_id})
                    # Yield to event loop
                    await asyncio.sleep(0)
                
                # Signal done
                await req.response_queue.put({"done": True})
                
            except Exception as e:
                print(f"Error during generation: {e}")
                await req.response_queue.put({"error": str(e)})
                await req.response_queue.put({"done": True})
            
            # 3. Mark task as done
            state.request_queue.task_done()
            
        except asyncio.CancelledError:
            print("Worker loop cancelled.")
            break
        except Exception as e:
            print(f"Worker loop error: {e}")
            await asyncio.sleep(1) # Prevent busy loop on error

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
