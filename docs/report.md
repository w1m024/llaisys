# 项目报告

这份报告按当前分支的真实实现编写。原模板里把 `Project #4` 写成了“支持新模型”，但结合现有代码、提交记录和参考提示，这一部分更准确的内容其实是“多用户推理服务”，因此下面按实际落地情况展开。

## 完成情况
- Assignment #1：Tensor
- Assignment #2：Operators
- Assignment #3：Large Language Model Inference
- Project #1：CPU 优化（部分）
- Project #3：聊天服务
- Project #4：多用户推理服务

## Project #1：CPU 优化（部分）
这一部分主要包含两类工作：一类是构建配置优化，另一类是热点算子并行化。构建层面在 `xmake.lua` 和 `xmake/cpu.lua` 里保持了 `C++17` 配置，同时为 CPU 路径打开了 `-O3`、`-march=native` 和 OpenMP。这样可以先把编译器自动向量化和多线程能力用起来，为后续算子优化提供基础收益。

算子层面，优先处理了推理里的热点路径。`linear` 的 CPU 实现加入了 `#pragma omp parallel for`，把 batch 维度上的计算分给多个线程；`self_attention` 不仅补齐了 batch 维支持，还能处理 3D/4D 输入，为后面的连续批处理做准备。也就是说，这部分优化不是只停留在构建参数上，而是落实到了真正会反复执行的算子里。

实现结果方面，CPU 路径的基础功能已经能正常工作。当前环境里实际跑通了：

```bash
python test/test_tensor.py
python test/test_runtime.py --device cpu
python test/ops/test_sample.py
```

实测方面，使用 `python test/benchmark_linear.py` 进行了一个简单 benchmark。在当前 16 线程 CPU 环境下，`M=32, K=64, N=32` 的线性层平均耗时约为 `0.0133 ms`，`M=128, K=4096, N=4096` 的线性层平均耗时约为 `1003.73 ms`。这组数据不代表完整性能上限，但足以说明 CPU 优化已经真实落地，而不是只写在文档里。

## Project #3：聊天服务
项目 3 的核心是把“能推理”进一步做成“能聊天”。为此，新增了一个 `sample` 算子，支持 `top_k`、`top_p`、`temperature` 和 `seed` 等采样参数。这个算子不只是后端实现，同时也一路打通到了 C API、ctypes 包装和 Python 上层接口，这样 Python 侧可以直接调用，也方便后续接到模型生成链路里。

在模型侧，采样参数已经接入 `Qwen2` 推理过程。现在 `Qwen2.generate()` 既支持同步生成，也支持 `stream=True` 的逐 token 输出；同时保留了默认 session，用来兼容原有调用方式。这样一来，模型不再只能走固定的 `argmax` 路径，而是可以按照采样参数产生更自然的回复。

在服务侧，已经实现了基于 FastAPI 的聊天接口，主入口是 `POST /v1/chat/completions`，同时支持普通返回和 SSE 流式输出。除了 API 之外，仓库里还提供了两个交互入口：一个是命令行脚本 `scripts/chat_cli.py`，另一个是 `python/llaisys/static/` 下的静态网页。这部分已经基本满足“聊天服务 + 流式输出 + 交互入口”的要求。

## Project #4：多用户推理服务
项目 4 的重点不再是单轮推理，而是让同一个模型实例可以稳定地服务多个会话。这里将会话状态从模型里拆了出来，并引入了 `Qwen2Session`。拆分之后，模型本身主要负责权重、配置和共享缓存；每个 session 自己维护 KV cache、序列长度和中间张量。这样做的好处是不同用户之间不会互相污染上下文，也是后面做多用户调度的前提。

在推理接口上，已经补齐了 batch 推理链路，包括 C++ 侧的 `infer_batch` / `process_batch`，以及 Python 侧的 `generate_batch()`。服务端则在 `python/llaisys/server.py` 里实现了请求队列、活动请求池和后台 worker。worker 会不断从队列里取请求，尽量拼满一个 batch，做一次批量前向，然后把未完成的请求留到下一轮继续处理，这就是 continuous batching 的基本思路。

为了让 continuous batching 真正可用，还补了两类底层能力。第一类是算子支持，`self_attention` 现在已经能处理 batch 输入；第二类是缓存机制，Qwen2 后端支持分块 KV cache 和 prefix cache 复用。这样对于多轮对话或者前缀相同的请求，系统可以直接复用已有状态，只对增量部分继续推理，减少重复计算。

验证入口方面，仓库里已经有 `test/test_server.py` 和 `scripts/bench_server.py`。前者覆盖并发请求、session 复用、忙碌会话冲突、取消和清理逻辑，后者可以做并发压测并输出吞吐、平均延迟和 P95 延迟。当前服务的边界也比较明确：它默认加载一个全局 CPU Qwen2 模型实例，通过多 session 和请求调度来支持多用户，而不是分布式多副本服务。

## 总结
整体来看，当前分支已经完成了从 Tensor、CPU 算子到 Qwen2 推理的主链路，并在这个基础上继续扩展了 CPU 优化、聊天服务和多用户推理服务。报告重点不是把所有项目名字都写满，而是把已经真正落地、能够被代码和测试证明的部分说明清楚。

当前分支最稳定的路径仍然是 `C++17 + CPU + Qwen2`。因此这份报告也主要围绕这条主线展开：前端用 Python 做包装和服务，后端用 C++ 完成张量、算子、模型和缓存调度；在此之上，再把单用户推理扩展成支持流式聊天和多用户批处理的完整系统。
