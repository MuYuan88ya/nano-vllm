import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """
    模型执行器类 (Model Runner Class)

    负责实际的模型推理执行，是推理引擎中最复杂的组件。
    支持张量并行（多 GPU）、CUDA Graph 优化、KV Cache 管理等高级特性。

    核心职责：
    1. 模型加载和初始化
    2. KV Cache 内存分配和管理
    3. 准备推理输入数据（prefill 和 decode 不同）
    4. 执行模型前向传播
    5. 采样生成新 token
    6. CUDA Graph 捕获和回放（decode 阶段优化）
    7. 多进程通信（张量并行）

    多进程架构：
    - Rank 0（主进程）：完整功能，负责调度和采样
    - Rank 1-N（工作进程）：只运行模型推理，等待主进程指令
    - 通信方式：共享内存 (SharedMemory) + 同步事件 (Event)

    优化技术：
    1. 张量并行（Tensor Parallelism）：多 GPU 并行计算
    2. CUDA Graph：减少 kernel 启动开销（decode 阶段）
    3. Flash Attention：高效的注意力计算
    4. Paged Attention：分页 KV Cache 管理
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        初始化模型执行器

        参数：
            config: 配置对象
            rank: 当前进程的 rank（0 为主进程）
            event: 同步事件
                - Rank 0: 事件列表（用于通知所有工作进程）
                - Rank > 0: 单个事件（用于接收主进程通知）

        初始化流程：
        1. 初始化分布式进程组（NCCL，用于 GPU 通信）
        2. 设置 GPU 设备和数据类型
        3. 加载模型权重
        4. 预热模型（评估内存使用）
        5. 分配 KV Cache 内存
        6. 捕获 CUDA Graph（如果启用）
        7. 工作进程进入消息循环
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size  # KV Cache 块大小
        self.enforce_eager = config.enforce_eager  # 是否禁用 CUDA Graph
        self.world_size = config.tensor_parallel_size  # 总进程数（GPU 数）
        self.rank = rank  # 当前进程的 rank
        self.event = event  # 同步事件

        # 初始化分布式进程组（NCCL 用于 GPU 间通信）
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        # 设置当前进程使用的 GPU
        torch.cuda.set_device(rank)
        # 保存原始数据类型，切换到模型数据类型
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")  # 默认在 GPU 上创建张量
        # 加载模型（张量并行：权重自动切分到各 GPU）
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        # 初始化采样器（只在主进程使用）
        self.sampler = Sampler()
        # 预热模型：评估内存使用，确定可用的 KV Cache 大小
        self.warmup_model()
        # 分配 KV Cache 内存
        self.allocate_kv_cache()
        # 捕获 CUDA Graph（如果启用）
        if not self.enforce_eager:
            self.capture_cudagraph()
        # 恢复原始数据类型和设备
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 工作进程进入消息循环（主进程继续执行）
        if self.world_size > 1:
            if rank == 0:
                # 主进程：创建共享内存
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)  # 1MB
                dist.barrier()  # 等待所有进程就绪
            else:
                # 工作进程：连接到共享内存
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                # 进入消息循环，等待主进程指令
                self.loop()

    def exit(self):
        """
        清理资源，关闭进程

        清理步骤：
        1. 关闭共享内存
        2. 同步所有进程
        3. 主进程删除共享内存
        4. 释放 CUDA Graph 资源
        5. 销毁分布式进程组
        """
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()  # 确保所有进程都关闭了共享内存
            if self.rank == 0:
                self.shm.unlink()  # 主进程删除共享内存
        if not self.enforce_eager:
            # 释放 CUDA Graph 相关资源
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()  # 等待 GPU 操作完成
        dist.destroy_process_group()  # 销毁分布式进程组

    def loop(self):
        """
        工作进程的消息循环

        工作进程启动后进入此循环，等待主进程的指令：
        1. 从共享内存读取方法名和参数
        2. 调用相应的方法
        3. 如果收到 "exit" 指令，退出循环
        """
        while True:
            method_name, args = self.read_shm()  # 从共享内存读取指令
            self.call(method_name, *args)  # 执行指令
            if method_name == "exit":
                break  # 退出循环

    def read_shm(self):
        """
        从共享内存读取数据（工作进程调用）

        数据格式：
        - 前 4 字节：数据长度（小端序）
        - 后续字节：pickle 序列化的 (method_name, *args)

        流程：
        1. 等待事件（主进程写入完成）
        2. 读取数据长度
        3. 反序列化数据
        4. 清除事件（准备下次接收）

        返回：
            (method_name, args)
        """
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()  # 等待主进程写入
        n = int.from_bytes(self.shm.buf[0:4], "little")  # 读取长度
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])  # 反序列化
        self.event.clear()  # 清除事件
        return method_name, args

    def write_shm(self, method_name, *args):
        """
        写入数据到共享内存（主进程调用）

        数据格式：
        - 前 4 字节：数据长度（小端序）
        - 后续字节：pickle 序列化的 [method_name, *args]

        流程：
        1. 序列化数据
        2. 写入长度和数据
        3. 设置所有事件（通知工作进程）

        参数：
            method_name: 要调用的方法名
            *args: 方法参数
        """
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])  # 序列化
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")  # 写入长度
        self.shm.buf[4:n+4] = data  # 写入数据
        # 通知所有工作进程
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """
        调用方法（支持多进程）

        主进程调用此方法时：
        1. 将指令写入共享内存（通知工作进程）
        2. 执行本地方法

        工作进程调用此方法时：
        - 直接执行本地方法

        参数：
            method_name: 要调用的方法名
            *args: 方法参数

        返回：
            方法的返回值（只在主进程有意义）
        """
        if self.world_size > 1 and self.rank == 0:
            # 主进程：通知所有工作进程
            self.write_shm(method_name, *args)
        # 获取方法并调用
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """
        预热模型，评估内存使用

        目的：
        1. 触发 PyTorch 的内存分配
        2. 测量模型运行时的峰值内存
        3. 为 KV Cache 分配预留足够空间

        流程：
        1. 清空 GPU 缓存，重置内存统计
        2. 构造最大批次的输入（prefill 阶段）
        3. 执行一次前向传播
        4. 清空缓存，准备分配 KV Cache
        """
        torch.cuda.empty_cache()  # 清空缓存
        torch.cuda.reset_peak_memory_stats()  # 重置峰值统计
        # 构造最大批次的输入
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        # 创建虚拟序列（每个序列都是 max_model_len 长度）
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        # 执行一次 prefill（触发内存分配）
        self.run(seqs, True)
        # 清空缓存，为 KV Cache 分配腾出空间
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        分配 KV Cache 内存

        KV Cache 大小计算：
        1. 评估模型运行时的内存使用（通过 warmup）
        2. 计算单个块的内存大小（block_bytes）
        3. 根据可用内存和 gpu_memory_utilization 确定块数

        KV Cache 形状：
        [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        - 2: Key 和 Value
        - num_layers: 模型层数
        - num_blocks: 块数（动态计算）
        - block_size: 每块的 token 数（256）
        - num_kv_heads: KV 头数（张量并行切分后）
        - head_dim: 每个头的维度

        内存分配策略：
        - 使用 gpu_memory_utilization 比例的总内存
        - 减去模型和激活的内存使用
        - 剩余空间分配给 KV Cache
        """
        config = self.config
        hf_config = config.hf_config
        # 获取 GPU 内存信息
        free, total = torch.cuda.mem_get_info()
        used = total - free  # 当前已使用内存
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]  # 预热时的峰值
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]  # 当前分配
        # 计算每个进程的 KV 头数（张量并行切分）
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 计算单个块的内存大小（字节）
        # 2: K 和 V, num_layers: 每层都需要, block_size: token 数, dtype.itemsize: 字节数
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        # 计算可分配的块数
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0, "没有足够的内存分配 KV Cache"
        # 分配 KV Cache 张量
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        # 将 KV Cache 绑定到模型的注意力层
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # 为每个注意力层分配对应的 KV Cache 视图
                module.k_cache = self.kv_cache[0, layer_id]  # Key cache
                module.v_cache = self.kv_cache[1, layer_id]  # Value cache
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        准备块表（Block Tables）张量

        块表用于 Paged Attention，记录每个序列的 KV Cache 块 ID。

        输入：每个序列的 block_table 长度可能不同
        输出：统一形状的张量，短的序列用 -1 填充

        形状：[batch_size, max_num_blocks]
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        # 填充 -1 到统一长度
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # 转换为张量并传输到 GPU
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        准备 Prefill 阶段的输入数据

        Prefill 特点：
        - 处理整个 prompt（多个 tokens）
        - 使用 Flash Attention 的 varlen（变长）接口
        - 支持前缀缓存（部分 tokens 可能已缓存）

        返回的数据：
        - input_ids: 需要处理的所有 token IDs（排除已缓存的）
        - positions: 每个 token 在序列中的位置
        - cu_seqlens_q/k: 累积序列长度（用于 Flash Attention）
        - max_seqlen_q/k: 最大序列长度
        - slot_mapping: token 到 KV Cache 位置的映射
        - block_tables: 块表（如果有前缀缓存）

        前缀缓存：
        - seq.num_cached_tokens: 已缓存的 token 数
        - 只需处理新的 tokens（从 num_cached_tokens 开始）
        - 但注意力计算需要所有 tokens 的 KV（包括缓存的）
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]  # Query 的累积序列长度
        cu_seqlens_k = [0]  # Key 的累积序列长度
        max_seqlen_q = 0  # Query 的最大序列长度
        max_seqlen_k = 0  # Key 的最大序列长度
        slot_mapping = []  # Token 到 KV Cache 的映射
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)  # 序列总长度
            # 只处理未缓存的 tokens
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            # Query 长度：未缓存的 tokens
            seqlen_q = seqlen - seq.num_cached_tokens
            # Key 长度：所有 tokens（包括缓存的）
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            # 构建 slot_mapping（跳过 warmup 序列）
            if not seq.block_table:    # warmup 时没有 block_table
                continue
            # 计算每个 token 在 KV Cache 中的位置
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    # 完整块：所有 token 位置
                    end = start + self.block_size
                else:
                    # 最后一个块：只到实际 token 数
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        # 如果有前缀缓存（cu_seqlens_k > cu_seqlens_q），需要 block_tables
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        # 转换为张量并传输到 GPU
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 设置全局上下文（用于注意力层访问）
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        准备 Decode 阶段的输入数据

        Decode 特点：
        - 每个序列只处理最后一个 token
        - 使用 Flash Attention 的 KV Cache 接口
        - 批次大小固定（每序列一个 token）

        返回的数据：
        - input_ids: 每个序列的最后一个 token
        - positions: 每个 token 在序列中的位置
        - slot_mapping: token 在 KV Cache 中的位置
        - context_lens: 每个序列的长度（用于注意力掩码）
        - block_tables: 块表
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)  # 只处理最后一个 token
            positions.append(len(seq) - 1)  # 位置索引
            context_lens.append(len(seq))  # 序列长度
            # 计算最后一个 token 在 KV Cache 中的位置
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        # 转换为张量并传输到 GPU
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        # 设置全局上下文（Decode 模式）
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """
        准备采样参数（只在主进程使用）

        提取每个序列的温度参数，用于控制生成的随机性

        返回：
            temperatures 张量，形状 [batch_size]
        """
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        运行模型前向传播

        根据不同情况选择执行方式：
        1. Prefill 阶段：直接执行（input_ids 长度不固定）
        2. Decode 阶段 + 启用 CUDA Graph + batch_size <= 512：使用 CUDA Graph
        3. 其他情况：直接执行

        CUDA Graph 优化：
        - 预先捕获固定批次大小的计算图
        - 推理时直接回放，减少 kernel 启动开销
        - 只适用于 Decode 阶段（输入形状固定）

        返回：
            logits 张量，形状 [batch_size, vocab_size]
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 直接执行模型
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 使用 CUDA Graph 回放
            bs = input_ids.size(0)  # 批次大小
            context = get_context()
            # 选择合适的 graph（找到 >= bs 的最小 graph）
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # 更新 graph 的输入变量
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)  # 先填充 -1
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # 回放计算图
            graph.replay()
            # 返回输出（只取前 bs 个）
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        执行一次推理（主方法）

        流程：
        1. 准备输入数据（prefill 或 decode）
        2. 准备采样参数（只在主进程）
        3. 运行模型得到 logits
        4. 采样得到 token IDs（只在主进程）
        5. 重置上下文

        参数：
            seqs: 要推理的序列列表
            is_prefill: 是否为 prefill 阶段

        返回：
            token_ids 列表（只在主进程有值，工作进程返回 None）
        """
        # 步骤1：准备输入数据
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # 步骤2：准备采样参数（只在主进程）
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # 步骤3：运行模型
        logits = self.run_model(input_ids, positions, is_prefill)
        # 步骤4：采样（只在主进程）
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        # 步骤5：重置上下文
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        捕获 CUDA Graph（Decode 阶段优化）

        CUDA Graph 原理：
        - 预先记录固定输入形状的 kernel 调用序列
        - 推理时直接回放，避免重复的 kernel 启动开销
        - 大幅减少 CPU 开销，提升小批次推理速度

        捕获策略：
        - 为多个批次大小捕获 graph（1, 2, 4, 8, 16, ..., max_bs）
        - 推理时选择 >= 实际批次大小的最小 graph
        - 使用 graph pool 共享 memory pool（减少内存碎片）

        限制：
        - 只适用于 Decode 阶段（输入形状固定）
        - 批次大小 <= 512（更大的批次启动开销占比低）
        - 不能动态改变输入形状
        """
        config = self.config
        hf_config = config.hf_config
        # 最大批次大小
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 预分配最大批次的输入张量（复用这些张量）
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 要捕获的批次大小列表
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 为每个批次大小捕获 graph（倒序：从大到小）
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            # 设置上下文（Decode 模式）
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # 预热（确保所有 kernel 都已编译）
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            # 捕获 graph
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            # 创建 memory pool（第一次捕获后）
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            # 保存 graph
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存 graph 的输入/输出变量（用于回放时更新）
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
