#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import math
import statistics
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, List


def percentile(values: List[float], ratio: float) -> float:
    if not values:
        return 0.0
    if ratio <= 0:
        return values[0]
    if ratio >= 100:
        return values[-1]
    rank = math.ceil((ratio / 100.0) * len(values)) - 1
    rank = max(0, min(rank, len(values) - 1))
    return values[rank]


def make_request(url: str, payload: Dict[str, object], timeout: float, stream: bool) -> Dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if stream:
                chunks: List[str] = []
                completion_tokens = 0
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    delta = event["choices"][0]["delta"]
                    content = delta.get("content")
                    if content:
                        chunks.append(content)
                        completion_tokens += 1
                content = "".join(chunks)
                prompt_tokens = 0
            else:
                payload = json.loads(response.read().decode("utf-8"))
                usage = payload.get("usage", {})
                content = payload["choices"][0]["message"]["content"]
                prompt_tokens = int(usage.get("prompt_tokens", 0))
                completion_tokens = int(usage.get("completion_tokens", 0))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc

    elapsed = time.perf_counter() - started_at
    return {
        "latency": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "content": content,
    }


def run_user(user_index: int, args: argparse.Namespace, barrier: threading.Barrier) -> List[Dict[str, object]]:
    session_id = None
    history: List[Dict[str, str]] = []
    results: List[Dict[str, object]] = []

    if args.session_mode == "sticky":
        session_id = f"bench-user-{user_index}"

    barrier.wait()

    for round_index in range(args.rounds):
        prompt = args.prompt.format(user=user_index, round=round_index)
        messages = [{"role": "user", "content": prompt}]
        if session_id:
            history.append(messages[0])
            request_messages = list(history)
        else:
            request_messages = messages

        payload: Dict[str, object] = {
            "model": args.model_name,
            "messages": request_messages,
            "max_tokens": args.max_tokens,
            "stream": args.stream,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        }
        if session_id:
            payload["session_id"] = session_id

        result = make_request(args.url, payload, args.timeout, args.stream)
        result["user"] = user_index
        result["round"] = round_index
        results.append(result)

        if session_id:
            history.append({"role": "assistant", "content": str(result["content"])})

    return results


def print_summary(results: List[Dict[str, object]], wall_time: float):
    latencies = sorted(float(item["latency"]) for item in results)
    completion_tokens = sum(int(item["completion_tokens"]) for item in results)
    prompt_tokens = sum(int(item["prompt_tokens"]) for item in results)

    print(f"requests={len(results)} users={len({item['user'] for item in results})}")
    print(f"wall_time={wall_time:.3f}s")
    print(f"throughput_req={len(results) / wall_time:.3f} req/s")
    print(f"throughput_tok={completion_tokens / wall_time:.3f} tok/s")
    print(f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}")
    print(f"latency_avg={statistics.mean(latencies):.3f}s")
    print(f"latency_p50={percentile(latencies, 50):.3f}s")
    print(f"latency_p95={percentile(latencies, 95):.3f}s")


def run_warmup(args: argparse.Namespace):
    if args.warmup <= 0:
        return

    payload = {
        "model": args.model_name,
        "messages": [{"role": "user", "content": args.prompt.format(user=0, round=0)}],
        "max_tokens": args.max_tokens,
        "stream": args.stream,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
    }

    print(f"Running warmup requests: {args.warmup}")
    for _ in range(args.warmup):
        make_request(args.url, payload, args.timeout, args.stream)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark /v1/chat/completions with concurrent users.",
    )
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model-name", type=str, default="deepseek-r1-distill-qwen-1.5b")
    parser.add_argument("--users", type=int, default=2, help="Concurrent users.")
    parser.add_argument("--rounds", type=int, default=2, help="Requests per user.")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--prompt", type=str, default="Explain prefix caching in one sentence.")
    parser.add_argument(
        "--session-mode",
        choices=("none", "sticky"),
        default="none",
        help="Use sticky session_id per user to benchmark multi-turn chat.",
    )
    parser.add_argument("--stream", action="store_true", help="Use SSE streaming responses.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    if args.users < 2:
        raise SystemExit("--users must be at least 2 for the P4-6 concurrency benchmark")
    if args.rounds < 1:
        raise SystemExit("--rounds must be at least 1")

    run_warmup(args)

    barrier = threading.Barrier(args.users)
    started_at = time.perf_counter()
    results: List[Dict[str, object]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.users) as executor:
        futures = [executor.submit(run_user, user_index, args, barrier) for user_index in range(args.users)]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())

    wall_time = time.perf_counter() - started_at
    print_summary(results, wall_time)


if __name__ == "__main__":
    main()
