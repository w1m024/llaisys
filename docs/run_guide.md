# 复现流程

## 版本要求

- C++17
- Python>=3.9
- torch>=2.4.0


## 环境准备

```bash
xmake -r
xmake install
pip install ./python
```



## 项目#1：CPU 优化

如果要观察项目#1相关性能，可以执行：

```bash
python test/ops/linear.py --device cpu --profile
python test/ops/self_attention.py --device cpu --profile
python test/benchmark_linear.py
```

## 项目#3：聊天服务复现

### CLI 交互

```bash
python scripts/chat_cli.py --model /home/wsl/model/DeepSeek-R1-Distill-Qwen-1.5B
```

CLI 当前支持：

- 多轮对话历史
- 流式输出
- session 复用

### 启动 FastAPI 服务

服务启动时必须显式指定模型路径：

```bash
PYTHONPATH=python python -m llaisys.server --model /abs/path/to/qwen2
```

如果还需要自定义监听地址，也同样在启动命令里显式传入：

```bash
PYTHONPATH=python python -m llaisys.server --model /abs/path/to/qwen2 --host 127.0.0.1 --port 8000
```

### 验证接口

非流式请求：

```bash
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"stream":false,"max_tokens":8}'
```

流式请求：

```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"stream":true,"max_tokens":8}'
```

## 项目#4：多用户服务复现

### 服务端测试

```bash
python -m pytest test/test_server.py
```

当前测试覆盖的重点包括：

- 并发请求能被同一批次处理
- `session_id` 的增量 prompt 复用
- 临时会话的销毁
- 忙碌会话的冲突处理
- 会话取消与清理

### 并发压测

先启动服务端，再运行压测脚本：

```bash
python scripts/bench_server.py --users 2 --rounds 2
python scripts/bench_server.py --users 4 --rounds 4 --session-mode sticky --stream
```

脚本会输出：

- 请求吞吐
- token 吞吐
- 平均延迟
- P50 / P95 延迟
