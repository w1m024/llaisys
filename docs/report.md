# 项目报告

## 完成情况
- Assignment #1：Tensor
- Assignment #2：Operators
- Assignment #3：Large Language Model Inference
- Project #1：CPU 优化（部分）
- Project #3：聊天服务
- Project #4：多用户推理服务

## Project #1：CPU 优化（部分）
这一部分主要包含两类工作：一类是构建配置优化，另一类是热点算子并行化。
构建层面, 为 CPU 路径打开了 `-O3`、`-march=native` 和 OpenMP。

算子层面，`linear` 的 CPU 实现加入了 `#pragma omp parallel for`，把 batch 维度上的计算分给多个线程.

## Project #3：聊天服务
项目 3 的核心r任务是新增了一个 `sample` 算子，支持 `top_k`、`top_p`、`temperature` 和 `seed` 等采样参数。

在模型侧，采样参数已经接入 `Qwen2` 推理过程。现在 `Qwen2.generate()` 既支持同步生成，也支持 `stream=True` 的逐 token 输出；同时保留了默认 session，用来兼容原有调用方式。

在服务侧，已经实现了基于 FastAPI 的聊天接口，主入口是 `POST /v1/chat/completions`，同时支持普通返回和 SSE 流式输出。除 API 外，还提供了两个交互入口：一个是命令行脚本 `scripts/chat_cli.py`，另一个是 `python/llaisys/static/` 下的静态网页。

## Project #4：多用户推理服务
项目 4 的核心任务是将会话状态从模型里拆了出来，并引入了 `Qwen2Session`。拆分之后，模型本身主要负责权重、配置和共享缓存；每个 session 自己维护 KV cache、序列长度和中间张量。这样做后不同用户之间不会互相污染上下文，也是后面做多用户调度打基础。

在推理接口上，已经补齐了 batch 推理链路，包括 C++ 侧的 `infer_batch` / `process_batch`，以及 Python 侧的 `generate_batch()`。服务端则在 `python/llaisys/server.py` 里实现了请求队列、活动请求池和后台 worker。基本实现了 continuous batching ，worker 会不断从队列里取请求，尽量拼满一个 batch，做一次批量前向，然后把未完成的请求留到下一轮继续处理。

为了让 continuous batching 真正可用，还补了两点。一是算子支持，`self_attention` 已能处理 batch 输入；二是缓存机制，Qwen2 后端支持分块 KV cache 和 prefix cache 复用。这样对于多轮对话或者前缀相同的请求，系统可以直接复用已有状态，只对增量部分继续推理。

验证方面，添加了 `test/test_server.py` 和 `scripts/bench_server.py`。前者覆盖并发请求、session 复用、忙碌会话冲突、取消和清理逻辑，后者可以做并发压测并输出吞吐、平均延迟和 P95 延迟。