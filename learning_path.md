# Nano-vLLM 学习指南 (Learning Guide)

## 目录 (Table of Contents)
1. [项目概述](#项目概述)
2. [核心概念](#核心概念)
3. [架构设计](#架构设计)
4. [学习路径](#学习路径)
5. [核心组件详解](#核心组件详解)
6. [优化技术](#优化技术)
7. [实践练习](#实践练习)

---

## 项目概述

### 什么是 Nano-vLLM？
Nano-vLLM 是一个从零开始构建的轻量级 vLLM (Virtual Large Language Model) 实现。它用约 1,300 行 Python 代码实现了与 vLLM 相当的推理性能。

### 关键特性
- 🚀 **快速离线推理** - 与 vLLM 相当的推理速度
- 📖 **可读代码库** - 清晰简洁的实现
- ⚡ **优化套件** - 包含前缀缓存、张量并行、Torch 编译、CUDA 图等优化

### 项目统计
- 总代码行数: ~1,314 行
- 核心文件数: 21 个 Python 文件
- 主要依赖: PyTorch, Transformers, Flash-Attention, Triton

---

## 核心概念

### 1. LLM 推理引擎
大语言模型推理引擎负责高效地执行模型推理，处理多个请求的调度、批处理和内存管理。

### 2. 关键技术概念

#### KV Cache (键值缓存)
- 在 Transformer 模型中，attention 机制需要计算 Query、Key、Value
- KV Cache 缓存已计算的 Key 和 Value，避免重复计算
- 大幅提升解码阶段的推理速度

#### Paged Attention
- 将 KV Cache 划分为固定大小的块 (blocks)
- 每个块可以独立管理，类似操作系统的分页内存管理
- 提高内存利用率，支持更大的批处理

#### Continuous Batching
- 动态批处理技术，不同长度的序列可以在同一批次中处理
- 请求完成后立即释放资源，新请求可以动态加入
- 大幅提升吞吐量

#### Prefix Caching
- 缓存常见的提示词前缀
- 相同前缀的请求可以共享 KV Cache
- 减少重复计算，提升效率

---

## 架构设计

### 整体架构图
```
用户请求
    ↓
LLM (llm.py)
    ↓
LLMEngine (engine/llm_engine.py)
    ↓
├── Scheduler (engine/scheduler.py)      ← 调度器：管理请求队列和资源分配
│   └── BlockManager (engine/block_manager.py)  ← 内存管理：分配和回收 KV Cache 块
│
└── ModelRunner (engine/model_runner.py)  ← 模型执行器：运行模型推理
    ├── Qwen3ForCausalLM (models/qwen3.py)  ← 模型定义
    └── Sampler (layers/sampler.py)        ← 采样器：生成下一个 token
```

### 目录结构
```
nanovllm/
├── __init__.py          # 包入口
├── llm.py              # 用户 API 入口
├── config.py           # 配置管理
├── sampling_params.py  # 采样参数
│
├── engine/             # 核心推理引擎
│   ├── llm_engine.py   # 主引擎：协调调度和执行
│   ├── scheduler.py    # 调度器：管理请求队列
│   ├── model_runner.py # 模型执行器：运行推理
│   ├── sequence.py     # 序列管理：请求状态跟踪
│   └── block_manager.py # 内存管理：KV Cache 块管理
│
├── models/             # 模型定义
│   └── qwen3.py       # Qwen3 模型实现
│
├── layers/            # 神经网络层
│   ├── attention.py   # 注意力机制
│   ├── sampler.py    # Token 采样
│   ├── activation.py # 激活函数
│   ├── linear.py     # 线性层（支持张量并行）
│   ├── layernorm.py  # 层归一化
│   ├── rotary_embedding.py # 旋转位置编码
│   └── embed_head.py # 词嵌入和输出层
│
└── utils/            # 工具函数
    ├── loader.py     # 模型加载
    └── context.py    # 推理上下文管理
```

---

## 学习路径

### 第一阶段：基础概念 (1-2 天)
**目标**: 理解 LLM 推理的基本概念和工作流程

1. **了解 Transformer 架构**
   - 学习 attention 机制
   - 理解 encoder-decoder 和 decoder-only 架构
   - 阅读: `models/qwen3.py` (模型结构)

2. **理解推理过程**
   - Prefill 阶段：处理输入提示词
   - Decode 阶段：逐 token 生成输出
   - 阅读: `example.py` (简单示例)

3. **运行第一个例子**
   ```python
   from nanovllm import LLM, SamplingParams
   llm = LLM("path/to/model", enforce_eager=True)
   outputs = llm.generate(["Hello!"], SamplingParams())
   ```

### 第二阶段：核心流程 (2-3 天)
**目标**: 掌握推理引擎的核心工作流程

1. **从入口开始**
   - 阅读顺序: `llm.py` → `engine/llm_engine.py`
   - 理解: 初始化流程、请求添加、推理步骤

2. **理解序列管理**
   - 阅读: `engine/sequence.py`
   - 理解: 序列状态、token 管理、块表

3. **掌握调度逻辑**
   - 阅读: `engine/scheduler.py`
   - 理解: Prefill/Decode 调度、抢占机制

4. **关键函数追踪**
   ```
   LLMEngine.generate()
     ↓
   LLMEngine.add_request()  # 添加请求到调度器
     ↓
   LLMEngine.step()         # 执行一步推理
     ↓
   Scheduler.schedule()     # 选择要执行的序列
     ↓
   ModelRunner.run()        # 运行模型
     ↓
   Scheduler.postprocess()  # 处理输出，更新状态
   ```

### 第三阶段：内存管理 (2-3 天)
**目标**: 理解 KV Cache 的分页内存管理

1. **学习 Paged Attention**
   - 了解为什么需要分页管理
   - 理解块 (block) 的概念

2. **深入 BlockManager**
   - 阅读: `engine/block_manager.py`
   - 理解: 块分配、引用计数、哈希缓存

3. **追踪内存操作**
   - 分配: `BlockManager.allocate()`
   - 回收: `BlockManager.deallocate()`
   - 追加: `BlockManager.may_append()`

4. **理解前缀缓存**
   - 哈希机制: 如何识别相同的前缀
   - 共享机制: 引用计数如何工作

### 第四阶段：模型执行 (3-4 天)
**目标**: 掌握模型推理的实现细节

1. **ModelRunner 工作流程**
   - 阅读: `engine/model_runner.py`
   - 理解: 初始化、预热、KV Cache 分配

2. **数据准备**
   - Prefill: `prepare_prefill()`
   - Decode: `prepare_decode()`
   - 理解: input_ids, positions, slot_mapping, block_tables

3. **注意力计算**
   - 阅读: `layers/attention.py`
   - 理解: Flash Attention 的使用
   - 学习: Triton kernel 的 KV Cache 存储

4. **模型前向传播**
   - 阅读: `models/qwen3.py`
   - 理解: 层次结构 (Embedding → Layers → Norm → LM Head)
   - 追踪: token 如何从输入转换为输出 logits

5. **采样过程**
   - 阅读: `layers/sampler.py`
   - 理解: 温度缩放、概率计算、token 采样

### 第五阶段：高级优化 (3-5 天)
**目标**: 理解各种性能优化技术

1. **CUDA Graph**
   - 阅读: `ModelRunner.capture_cudagraph()`
   - 理解: 为什么可以加速、如何捕获和回放

2. **张量并行 (Tensor Parallelism)**
   - 阅读: `layers/linear.py`
   - 理解: 权重切分、all-reduce 通信
   - 学习: 多 GPU 协作推理

3. **Torch Compile**
   - 查看: `@torch.compile` 装饰器的使用
   - 理解: 动态编译优化

4. **上下文管理**
   - 阅读: `utils/context.py`
   - 理解: 全局状态管理、线程安全

5. **模型加载**
   - 阅读: `utils/loader.py`
   - 理解: SafeTensors 加载、权重映射

### 第六阶段：实践与扩展 (持续)
**目标**: 动手实践，扩展和优化

1. **性能分析**
   - 运行: `bench.py`
   - 分析吞吐量、延迟
   - 对比不同配置的性能

2. **代码修改练习**
   - 修改块大小，观察内存和性能变化
   - 调整批处理大小
   - 实现简单的调度策略改进

3. **支持新模型**
   - 参考 `models/qwen3.py`
   - 尝试添加其他模型支持 (如 Llama, Mistral)

4. **添加新功能**
   - 实现 beam search
   - 添加更多采样策略 (top-k, top-p)
   - 支持流式输出

---

## 核心组件详解

### 1. Sequence (序列)
**文件**: `engine/sequence.py`

**作用**: 表示单个推理请求，跟踪其状态和 token 序列

**关键属性**:
- `token_ids`: 所有 token (包括 prompt 和生成的)
- `block_table`: 分配的 KV Cache 块 ID 列表
- `status`: 序列状态 (WAITING, RUNNING, FINISHED)
- `num_cached_tokens`: 已缓存的 token 数量 (用于前缀缓存)

**状态转换**:
```
WAITING → RUNNING → FINISHED
   ↑         ↓
   └─────────┘ (抢占时回到 WAITING)
```

### 2. Scheduler (调度器)
**文件**: `engine/scheduler.py`

**作用**: 管理请求队列，决定哪些序列应该被执行

**核心队列**:
- `waiting`: 等待执行的序列队列
- `running`: 正在执行的序列队列

**调度策略**:
1. **Prefill 优先**: 优先处理新请求的 prefill
2. **资源限制**: 考虑批处理大小和内存限制
3. **抢占机制**: 内存不足时抢占低优先级序列

**关键方法**:
- `schedule()`: 选择要执行的序列，返回 (序列列表, 是否 prefill)
- `postprocess()`: 处理模型输出，更新序列状态

### 3. BlockManager (块管理器)
**文件**: `engine/block_manager.py`

**作用**: 管理 KV Cache 内存，实现分页内存管理和前缀缓存

**核心数据结构**:
- `blocks`: 所有块的列表
- `free_block_ids`: 空闲块 ID 队列
- `hash_to_block_id`: 哈希到块 ID 的映射 (用于前缀缓存)

**关键机制**:
1. **引用计数**: 跟踪块的使用情况，支持共享
2. **哈希缓存**: 通过哈希识别相同的 token 序列
3. **延迟哈希**: 只对完整的块计算哈希

**前缀缓存工作原理**:
```
序列1: [A, B, C, D, E, F, G, H]
序列2: [A, B, C, D, X, Y, Z]

块划分 (block_size=4):
序列1: [A,B,C,D] [E,F,G,H]
序列2: [A,B,C,D] [X,Y,Z]

结果: 两个序列共享第一个块 [A,B,C,D]
```

### 4. ModelRunner (模型执行器)
**文件**: `engine/model_runner.py`

**作用**: 执行实际的模型推理，管理 GPU 资源

**初始化流程**:
1. 初始化分布式进程组 (用于张量并行)
2. 加载模型到 GPU
3. 预热模型 (warmup)
4. 分配 KV Cache
5. 捕获 CUDA Graph (可选)

**推理流程**:
1. **数据准备**: 根据 prefill/decode 准备不同的输入
2. **模型运行**: 执行前向传播得到 logits
3. **采样**: 从 logits 中采样下一个 token

**CUDA Graph 优化**:
- 预先捕获固定大小批次的计算图
- 推理时直接回放图，减少 kernel 启动开销
- 支持多个批次大小 (1, 2, 4, 8, 16, 32, ...)

### 5. Attention (注意力层)
**文件**: `layers/attention.py`

**作用**: 实现高效的注意力计算

**关键技术**:
1. **Flash Attention**: 内存高效的注意力实现
   - Prefill: 使用 `flash_attn_varlen_func`
   - Decode: 使用 `flash_attn_with_kvcache`

2. **Triton Kernel**: 自定义的 KV Cache 存储 kernel
   - 高效地将 K, V 存储到 KV Cache
   - 支持任意的 slot mapping

**数据流**:
```
Q, K, V (当前计算)
    ↓
store_kvcache (存储到 KV Cache)
    ↓
flash_attention (使用 KV Cache 计算)
    ↓
Output
```

---

## 优化技术

### 1. Continuous Batching (连续批处理)
**原理**: 动态调整批次，序列完成后立即处理新请求

**优势**:
- 提高 GPU 利用率
- 降低平均延迟
- 提升吞吐量

**实现**: 在 `Scheduler.schedule()` 中动态构建批次

### 2. Prefix Caching (前缀缓存)
**原理**: 缓存并共享相同的 token 序列

**实现机制**:
1. 对完整的块计算哈希
2. 新序列分配时查找是否有相同哈希的块
3. 如果找到且 token 匹配，共享该块 (引用计数 +1)

**适用场景**:
- 多轮对话 (共享 system prompt)
- 批量处理相似请求
- Few-shot learning (共享 examples)

### 3. Paged Attention (分页注意力)
**原理**: 将 KV Cache 划分为固定大小的块

**优势**:
- 灵活的内存管理
- 减少内存碎片
- 支持更大的批处理

### 4. Tensor Parallelism (张量并行)
**原理**: 将模型权重切分到多个 GPU

**实现**:
- **列并行**: 将输出维度切分 (如 QKV projection)
- **行并行**: 将输入维度切分 (如 output projection)
- **通信**: 使用 all-reduce 同步梯度

**文件**: `layers/linear.py` (QKVParallelLinear, RowParallelLinear)

### 5. CUDA Graph
**原理**: 预先捕获计算图，避免重复的 kernel 启动开销

**适用场景**:
- Decode 阶段 (固定的计算模式)
- 批次大小 ≤ 512

**限制**:
- Prefill 阶段不适用 (输入长度变化)
- 需要固定的输入形状

### 6. Torch Compile
**原理**: 使用 PyTorch 2.0 的编译优化

**使用**: `@torch.compile` 装饰器 (如 Sampler)

**优势**:
- 自动融合操作
- 减少内存访问
- 提升计算效率

---

## 实践练习

### 练习 1: 追踪一个请求的完整生命周期
**任务**: 在代码中添加打印语句，跟踪一个请求从创建到完成的全过程

**关键点**:
1. 请求创建: `LLMEngine.add_request()`
2. 调度选择: `Scheduler.schedule()`
3. 内存分配: `BlockManager.allocate()`
4. 模型推理: `ModelRunner.run()`
5. 采样输出: `Sampler.forward()`
6. 状态更新: `Scheduler.postprocess()`
7. 请求完成: 序列状态变为 FINISHED

### 练习 2: 分析前缀缓存效果
**任务**: 测试前缀缓存对性能的影响

**步骤**:
1. 准备多个共享相同前缀的请求
2. 在 `BlockManager.allocate()` 中统计缓存命中率
3. 对比有/无前缀缓存的性能差异

### 练习 3: 实现简单的调度策略
**任务**: 修改 `Scheduler` 实现优先级调度

**建议**:
1. 为 Sequence 添加优先级属性
2. 修改 `schedule()` 方法，优先处理高优先级请求
3. 测试不同优先级的延迟差异

### 练习 4: 性能调优
**任务**: 调整参数，优化特定场景的性能

**参数**:
- `max_num_seqs`: 最大批处理大小
- `max_num_batched_tokens`: 最大批处理 token 数
- `kvcache_block_size`: KV Cache 块大小
- `gpu_memory_utilization`: GPU 内存利用率

**测试场景**:
1. 长文本生成
2. 高并发短文本
3. 多轮对话

### 练习 5: 添加监控指标
**任务**: 实现性能监控和可观测性

**建议指标**:
- 平均延迟
- P50, P90, P99 延迟
- 吞吐量 (tokens/s)
- GPU 内存使用率
- KV Cache 利用率
- 前缀缓存命中率

---

## 学习资源

### 必读论文
1. **Attention Is All You Need** - Transformer 原始论文
2. **Efficient Memory Management for Large Language Model Serving with PagedAttention** - vLLM 论文
3. **FlashAttention: Fast and Memory-Efficient Exact Attention** - Flash Attention

### 推荐阅读
- PyTorch 官方文档: 分布式训练、CUDA 编程
- Flash Attention GitHub: 实现细节
- vLLM GitHub: 完整实现参考

### 调试技巧
1. **使用 enforce_eager=True**: 禁用 CUDA Graph，便于调试
2. **添加断点**: 在关键函数设置断点
3. **打印中间结果**: 查看 tensor 形状和值
4. **使用 torch.cuda.synchronize()**: 确保 GPU 操作完成

### 常见问题

**Q: 为什么 decode 阶段可以使用 CUDA Graph？**
A: Decode 阶段每次只生成一个 token，输入形状固定（batch_size, 1），计算图固定，适合预先捕获。

**Q: 前缀缓存如何识别相同的序列？**
A: 使用 xxhash 计算块内 token 的哈希值，相同哈希且 token 完全匹配才认为是相同的块。

**Q: 为什么需要引用计数？**
A: 多个序列可能共享同一个块（前缀缓存），引用计数跟踪块的使用情况，只有引用为 0 时才能释放。

**Q: 张量并行如何减少通信？**
A: 列并行后接行并行时，中间结果保持切分状态，只在必要时做 all-reduce，减少通信次数。

---

## 总结

Nano-vLLM 是学习 LLM 推理引擎的绝佳项目：
- ✅ 代码简洁，易于理解
- ✅ 包含主流优化技术
- ✅ 性能接近生产级别
- ✅ 适合深入学习和扩展

**建议学习时间**: 2-3 周（每天 2-3 小时）

**学习方法**:
1. 📖 先阅读本指南，理解整体架构
2. 🔍 按学习路径逐步深入代码
3. 🎯 完成实践练习，加深理解
4. 🚀 尝试扩展和优化

祝学习愉快！如有问题，欢迎查看代码注释或参考相关论文。
