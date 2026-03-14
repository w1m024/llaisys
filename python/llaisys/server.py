import time
import uuid
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from transformers import AutoTokenizer

import llaisys
from llaisys import DeviceType

# ==============================================================================
# Global State & Configuration
# ==============================================================================

@dataclass
class SessionState:
    session: Any
    cache_token_ids: List[int] = field(default_factory=list)
    in_flight: bool = False

class InferenceRequest:
    def __init__(self, 
                 input_ids: List[int], 
                 generation_config: Dict[str, Any],
                 response_queue: asyncio.Queue,
                 session_state: SessionState,
                 session_id: Optional[str] = None,
                 persistent: bool = False):
        self.input_ids = input_ids # Remaining prompt tokens to be processed
        self.generation_config = generation_config
        self.response_queue = response_queue
        self.session_state = session_state
        self.session_id = session_id
        self.persistent = persistent
        self.created_at = time.time()
        self.generated_len = 0 # How many new tokens generated
        self.finished = False
        self.last_token: Optional[int] = None

class GlobalState:
    model: Optional[llaisys.models.Qwen2] = None
    tokenizer: Optional[AutoTokenizer] = None
    model_path: str = "/home/wsl/model/DeepSeek-R1-Distill-Qwen-1.5B"  # Default path
    session_store: Dict[str, SessionState] = {}
    request_queue: asyncio.Queue = None # Initialize in lifespan
    
state = GlobalState()

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
            consumed_inputs = []
            
            requests_to_remove = []
            
            for req in active_requests:
                sessions.append(req.session_state.session)

                if req.input_ids:
                    next_tokens = list(req.input_ids)
                    req.input_ids = []
                elif req.last_token is not None:
                    next_tokens = [req.last_token]
                else:
                    raise RuntimeError("Request has no tokens to process and no cached decode token")

                batch_input_ids.append(next_tokens)
                consumed_inputs.append(next_tokens)
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
                req.session_state.cache_token_ids.extend(consumed_inputs[i])
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
                    _cleanup_request(req)
                    
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
            for req in active_requests:
                await req.response_queue.put({"error": "Worker cancelled"})
                _cleanup_request(req, discard_session=req.persistent)
            print("Worker loop cancelled.")
            break
        except Exception as e:
            print(f"Worker loop error: {e}")
            import traceback
            traceback.print_exc()
            for req in active_requests:
                await req.response_queue.put({"error": str(e)})
                _cleanup_request(req, discard_session=req.persistent)
            active_requests.clear()
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

    for session_state in list(state.session_store.values()):
        _destroy_session(session_state)
    state.session_store.clear()

app = FastAPI(title="LLAISYS Chatbot Server", lifespan=lifespan)

# ==============================================================================
# Helper Functions
# ==============================================================================

def _format_sse(data: BaseModel) -> str:
    return data.model_dump_json()


def _is_prefix(prefix: List[int], tokens: List[int]) -> bool:
    return len(prefix) <= len(tokens) and tokens[: len(prefix)] == prefix


def _destroy_session(session_state: SessionState):
    if state.model is None:
        return
    state.model.destroy_session(session_state.session)


def _cleanup_request(req: InferenceRequest, discard_session: bool = False):
    req.session_state.in_flight = False
    if discard_session or not req.persistent:
        _destroy_session(req.session_state)
        if req.session_id:
            state.session_store.pop(req.session_id, None)


def _prepare_session(prompt_ids: List[int], session_id: Optional[str]) -> tuple[SessionState, List[int], bool]:
    if not session_id:
        session_state = SessionState(session=state.model.create_session(), in_flight=True)
        return session_state, list(prompt_ids), False

    session_state = state.session_store.get(session_id)
    if session_state and session_state.in_flight:
        raise HTTPException(status_code=409, detail=f"Session '{session_id}' is busy")

    if session_state and _is_prefix(session_state.cache_token_ids, prompt_ids):
        delta_ids = prompt_ids[len(session_state.cache_token_ids):]
        if delta_ids:
            session_state.in_flight = True
            return session_state, delta_ids, True

    if session_state:
        _destroy_session(session_state)

    session_state = SessionState(session=state.model.create_session(), in_flight=True)
    state.session_store[session_id] = session_state
    return session_state, list(prompt_ids), True

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
    session_state, delta_input_ids, persistent = _prepare_session(input_ids, request.session_id)
    
    req = InferenceRequest(
        input_ids=delta_input_ids,
        generation_config=gen_config,
        response_queue=response_queue,
        session_state=session_state,
        session_id=request.session_id,
        persistent=persistent,
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
                yield {"event": "error", "data": item["error"]}
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
        completion_tokens = 0
        while True:
            item = await response_queue.get()
            if "error" in item:
                raise HTTPException(status_code=500, detail=item["error"])
            if "done" in item:
                break
            token_id = item["token_id"]
            completion_tokens += 1
            content += state.tokenizer.decode([token_id], skip_special_tokens=True)

        usage = {
            "prompt_tokens": len(input_ids),
            "completion_tokens": completion_tokens,
            "total_tokens": len(input_ids) + completion_tokens,
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
