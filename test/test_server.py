import asyncio
import contextlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from llaisys import server
from llaisys.chat_format import assistant_prefills_think, normalize_assistant_text


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(message["content"] for message in messages)

    def encode(self, text):
        if not text.strip():
            return []
        return [int(token) for token in text.split()]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(str(token_id) for token_id in token_ids if token_id != 0)


class FakeModel:
    def __init__(self, session_sequences, delay=0.0):
        self._meta = SimpleNamespace(end_token=0)
        self._pending_sequences = [list(sequence) for sequence in session_sequences]
        self._session_sequences = {}
        self._next_session_id = 1
        self.batch_calls = []
        self.destroyed_sessions = []
        self.delay = delay

    def create_session(self):
        session = f"session-{self._next_session_id}"
        self._next_session_id += 1
        if not self._pending_sequences:
            raise AssertionError("No token sequence prepared for new session")
        self._session_sequences[session] = self._pending_sequences.pop(0)
        return session

    def destroy_session(self, session):
        self.destroyed_sessions.append(session)

    def generate_batch(
        self,
        sessions,
        batch_input_ids,
        top_ks,
        top_ps,
        temperatures,
        seeds,
    ):
        if self.delay:
            time.sleep(self.delay)
        self.batch_calls.append(
            {
                "sessions": list(sessions),
                "inputs": [list(token_ids) for token_ids in batch_input_ids],
            }
        )
        outputs = []
        for session in sessions:
            outputs.append(self._session_sequences[session].pop(0))
        return outputs


def _reset_server_state(fake_model):
    server.state.model = fake_model
    server.state.tokenizer = FakeTokenizer()
    try:
        server.state.request_queue = asyncio.Queue()
    except RuntimeError:
        server.state.request_queue = None
    server.state.session_store = {}


async def _run_with_worker(*coroutines):
    worker = asyncio.create_task(server.worker_loop())
    try:
        return await asyncio.gather(*coroutines)
    finally:
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker


async def _collect_stream(response):
    events = []
    async for item in response.body_iterator:
        events.append(item)
    return events


def test_chat_completions_batches_concurrent_requests():
    async def scenario():
        fake_model = FakeModel([[101], [202]])
        _reset_server_state(fake_model)

        req1 = server.ChatCompletionRequest(
            messages=[server.ChatMessage(role="user", content="1 2")],
            max_tokens=1,
        )
        req2 = server.ChatCompletionRequest(
            messages=[server.ChatMessage(role="user", content="3 4")],
            max_tokens=1,
        )

        task1 = asyncio.create_task(server.chat_completions(req1))
        task2 = asyncio.create_task(server.chat_completions(req2))
        await asyncio.sleep(0)

        resp1, resp2 = await _run_with_worker(task1, task2)

        assert resp1.choices[0].message.content == "101"
        assert resp2.choices[0].message.content == "202"
        assert len(fake_model.batch_calls) == 1
        assert len(fake_model.batch_calls[0]["sessions"]) == 2
        assert fake_model.batch_calls[0]["inputs"] == [[1, 2], [3, 4]]

    asyncio.run(scenario())


def test_chat_format_normalizes_prefilled_think_output():
    assert assistant_prefills_think("prefix<think>\n") is True
    assert normalize_assistant_text("reasoning</think>answer", True) == "<think>\nreasoning</think>answer"


def test_chat_format_leaves_plain_answer_unchanged():
    assert assistant_prefills_think("plain prompt") is False
    assert normalize_assistant_text("plain answer", False) == "plain answer"


def test_chat_completions_reuses_session_with_incremental_prompt_suffix():
    async def scenario():
        fake_model = FakeModel([[10, 20]])
        _reset_server_state(fake_model)

        first_request = server.ChatCompletionRequest(
            messages=[server.ChatMessage(role="user", content="1 2")],
            max_tokens=1,
            session_id="chat-1",
        )
        second_request = server.ChatCompletionRequest(
            messages=[
                server.ChatMessage(role="user", content="1 2"),
                server.ChatMessage(role="assistant", content="10"),
                server.ChatMessage(role="user", content="3"),
            ],
            max_tokens=1,
            session_id="chat-1",
        )

        first_response, = await _run_with_worker(server.chat_completions(first_request))
        second_response, = await _run_with_worker(server.chat_completions(second_request))

        assert first_response.choices[0].message.content == "10"
        assert second_response.choices[0].message.content == "20"
        assert fake_model.batch_calls[0]["inputs"] == [[1, 2]]
        assert fake_model.batch_calls[1]["inputs"] == [[10, 3]]
        assert server.state.session_store["chat-1"].cache_token_ids == [1, 2, 10, 3]
        assert fake_model.destroyed_sessions == []

    asyncio.run(scenario())


def test_chat_completions_destroys_ephemeral_sessions_after_finish():
    async def scenario():
        fake_model = FakeModel([[77]])
        _reset_server_state(fake_model)

        request = server.ChatCompletionRequest(
            messages=[server.ChatMessage(role="user", content="8 9")],
            max_tokens=1,
        )

        response, = await _run_with_worker(server.chat_completions(request))

        assert response.choices[0].message.content == "77"
        assert fake_model.destroyed_sessions == ["session-1"]
        assert server.state.session_store == {}

    asyncio.run(scenario())


def test_chat_completions_rejects_busy_session():
    fake_model = FakeModel([[55]])
    _reset_server_state(fake_model)

    session_state = server.SessionState(session="session-busy", in_flight=True)
    server.state.session_store["chat-1"] = session_state

    request = server.ChatCompletionRequest(
        messages=[server.ChatMessage(role="user", content="1")],
        session_id="chat-1",
    )

    with pytest.raises(server.HTTPException) as exc_info:
        asyncio.run(server.chat_completions(request))

    assert exc_info.value.status_code == 409


def test_delete_chat_session_releases_idle_session():
    fake_model = FakeModel([[55]])
    _reset_server_state(fake_model)

    session_state = server.SessionState(session="session-idle", in_flight=False)
    server.state.session_store["chat-1"] = session_state

    response = asyncio.run(server.delete_chat_session("chat-1"))

    assert response == {"deleted": True}
    assert fake_model.destroyed_sessions == ["session-idle"]
    assert server.state.session_store == {}


def test_delete_chat_session_rejects_busy_session():
    fake_model = FakeModel([[55]])
    _reset_server_state(fake_model)

    session_state = server.SessionState(session="session-busy", in_flight=True)
    server.state.session_store["chat-1"] = session_state

    with pytest.raises(server.HTTPException) as exc_info:
        asyncio.run(server.delete_chat_session("chat-1"))

    assert exc_info.value.status_code == 409


def test_cancel_chat_session_stops_stream_and_releases_busy_session():
    async def scenario():
        fake_model = FakeModel([[11, 12, 13, 0]], delay=0.02)
        _reset_server_state(fake_model)

        request = server.ChatCompletionRequest(
            messages=[server.ChatMessage(role="user", content="1 2")],
            max_tokens=4,
            stream=True,
            session_id="chat-1",
        )

        response = await server.chat_completions(request)
        worker = asyncio.create_task(server.worker_loop())
        try:
            stream_iter = response.body_iterator.__aiter__()

            role_event = await stream_iter.__anext__()
            role_payload = json.loads(role_event["data"])
            assert role_payload["choices"][0]["delta"]["role"] == "assistant"

            token_event = await stream_iter.__anext__()
            token_payload = json.loads(token_event["data"])
            assert token_payload["choices"][0]["delta"]["content"] == "11"

            cancel_response = await server.cancel_chat_session("chat-1")
            assert cancel_response == {"cancelled": True}

            remaining_tokens = []
            saw_done = False
            async for event in stream_iter:
                data = event["data"]
                if data == "[DONE]":
                    saw_done = True
                    continue
                payload = json.loads(data)
                content = payload["choices"][0]["delta"].get("content")
                if content:
                    remaining_tokens.append(content)

            assert remaining_tokens == []
            assert saw_done is True
            assert server.state.session_store["chat-1"].in_flight is False
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker

    asyncio.run(scenario())


def test_chat_completions_streams_concurrent_requests_to_separate_sse_queues():
    async def scenario():
        fake_model = FakeModel([[11, 12], [21, 22]])
        _reset_server_state(fake_model)

        req1 = server.ChatCompletionRequest(
            messages=[server.ChatMessage(role="user", content="1 2")],
            max_tokens=2,
            stream=True,
        )
        req2 = server.ChatCompletionRequest(
            messages=[server.ChatMessage(role="user", content="3 4")],
            max_tokens=2,
            stream=True,
        )

        resp1 = await server.chat_completions(req1)
        resp2 = await server.chat_completions(req2)

        worker = asyncio.create_task(server.worker_loop())
        try:
            stream1, stream2 = await asyncio.gather(
                _collect_stream(resp1),
                _collect_stream(resp2),
            )
        finally:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker

        assert len(fake_model.batch_calls) == 2
        assert fake_model.batch_calls[0]["inputs"] == [[1, 2], [3, 4]]
        assert fake_model.batch_calls[1]["inputs"] == [[11], [21]]

        def parse_tokens(events):
            tokens = []
            saw_done = False
            for event in events:
                data = event["data"]
                if data == "[DONE]":
                    saw_done = True
                    continue
                payload = json.loads(data)
                delta = payload["choices"][0]["delta"]
                content = delta.get("content")
                if content:
                    tokens.append(content)
            assert saw_done is True
            return tokens

        assert parse_tokens(stream1) == ["11", "12"]
        assert parse_tokens(stream2) == ["21", "22"]

    asyncio.run(scenario())


def test_chat_ui_route_returns_index_file():
    response = asyncio.run(server.chat_ui())

    assert str(response.path).endswith("python/llaisys/static/index.html")
