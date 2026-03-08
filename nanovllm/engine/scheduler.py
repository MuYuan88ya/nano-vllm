from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    调度器类 (Scheduler Class)

    负责管理推理请求的调度和资源分配，是推理引擎的核心组件之一。

    核心职责：
    1. 请求队列管理：维护 waiting 和 running 两个队列
    2. 调度决策：决定哪些序列应该被执行（prefill 或 decode）
    3. 资源管理：协调 BlockManager 分配和释放 KV Cache 内存
    4. 抢占机制：内存不足时抢占低优先级序列

    调度策略：
    - Prefill 优先：新请求的 prefill 优先于 decode（提高首 token 延迟）
    - Continuous Batching：动态批处理，序列完成后立即处理新请求
    - 抢占（Preemption）：内存不足时，暂停部分 running 序列

    两阶段推理：
    1. Prefill 阶段：处理输入 prompt，计算所有 token 的 KV Cache（并行）
    2. Decode 阶段：逐个生成新 token，每次只处理一个 token（自回归）

    数据流：
    WAITING 队列 → schedule() 选择序列 → RUNNING 队列 → 模型推理 → postprocess() 更新状态
    """

    def __init__(self, config: Config):
        """
        初始化调度器

        参数：
            config: 配置对象，包含批处理大小、内存设置等参数

        关键配置：
            max_num_seqs: 单批次最大序列数（GPU 并行能力）
            max_num_batched_tokens: 单批次最大 token 数（内存限制）
            eos: EOS token ID，用于判断序列是否完成
        """
        self.max_num_seqs = config.max_num_seqs  # 批次最大序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 批次最大 token 数
        self.eos = config.eos  # EOS token ID
        # 初始化块管理器：管理 KV Cache 的分配和回收
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # WAITING 队列：等待调度的序列（新请求或被抢占的序列）
        self.waiting: deque[Sequence] = deque()
        # RUNNING 队列：正在执行的序列（已分配 KV Cache）
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """
        检查是否所有请求都已完成

        返回：
            True 如果没有等待或运行的序列，False 否则

        用途：
            在主循环中判断是否可以退出推理
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """
        添加新的推理请求到等待队列

        参数：
            seq: 新的序列对象（状态为 WAITING）

        流程：
            用户请求 → LLMEngine.add_request() → Scheduler.add()
        """
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        调度器的核心方法：选择要执行的序列

        调度逻辑：
        1. 优先处理 prefill（新请求或被抢占后重新开始的请求）
        2. 如果没有 prefill，处理 decode（继续生成）
        3. 考虑资源限制：批次大小、token 数、KV Cache 内存

        返回：
            (scheduled_seqs, is_prefill)
            - scheduled_seqs: 被选中执行的序列列表
            - is_prefill: True 表示 prefill 阶段，False 表示 decode 阶段

        注意：
            一个批次中的所有序列要么都是 prefill，要么都是 decode
            不能混合 prefill 和 decode（因为计算模式完全不同）
        """
        # ========== 阶段1：尝试调度 Prefill ==========
        scheduled_seqs = []
        num_seqs = 0  # 当前批次中的序列数
        num_batched_tokens = 0  # 当前批次中的 token 数

        # 遍历等待队列，选择可以 prefill 的序列
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]  # 查看队首序列（不移除）

            # 检查是否可以添加这个序列到当前批次
            # 限制条件：
            # 1. token 数不能超过 max_num_batched_tokens
            # 2. 必须有足够的 KV Cache 块可分配
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break  # 无法添加，停止选择

            # 可以添加到批次
            num_seqs += 1
            self.block_manager.allocate(seq)  # 分配 KV Cache 块
            # 计算需要处理的 token 数（排除已缓存的部分，用于前缀缓存优化）
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING  # 更新状态：等待 → 运行
            self.waiting.popleft()  # 从等待队列移除
            self.running.append(seq)  # 加入运行队列
            scheduled_seqs.append(seq)

        # 如果成功调度了 prefill 序列，返回
        if scheduled_seqs:
            return scheduled_seqs, True  # True 表示 prefill 阶段

        # ========== 阶段2：调度 Decode ==========
        # 只有在没有 prefill 任务时才处理 decode
        # 遍历运行队列，选择可以 decode 的序列
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()  # 从队首取出序列（FIFO）

            # 检查是否有足够的 KV Cache 空间追加新 token
            while not self.block_manager.can_append(seq):
                # 空间不足，需要抢占其他序列
                if self.running:
                    # 抢占队尾的序列（LIFO，最后进入的最先被抢占）
                    self.preempt(self.running.pop())
                else:
                    # 没有其他序列可抢占，抢占当前序列自己
                    self.preempt(seq)
                    break
            else:
                # 有足够空间，可以处理这个序列
                num_seqs += 1
                # 预分配可能需要的新块（如果当前块将满）
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        # 必须至少调度一个序列（否则会死锁）
        assert scheduled_seqs, "调度器未能选择任何序列"

        # 将选中的序列放回 running 队列的前面（保持顺序）
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False  # False 表示 decode 阶段

    def preempt(self, seq: Sequence):
        """
        抢占一个序列（内存不足时释放资源）

        抢占机制：
        - 释放序列占用的 KV Cache 块
        - 将序列状态改为 WAITING
        - 移动到等待队列的队首（优先重新调度）

        注意：
        - 被抢占的序列不会丢失进度（token_ids 保留）
        - 但 KV Cache 被释放，重新调度时需要重新计算（prefill）
        - 如果有前缀缓存，可能只需重新计算部分 KV

        参数：
            seq: 要抢占的序列

        副作用：
            - 序列状态：RUNNING → WAITING
            - KV Cache：释放所有块
            - 队列：从 running 移到 waiting 队首
        """
        seq.status = SequenceStatus.WAITING  # 更新状态：运行 → 等待
        self.block_manager.deallocate(seq)  # 释放 KV Cache 块
        self.waiting.appendleft(seq)  # 加入等待队列的队首（高优先级）

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """
        处理模型输出，更新序列状态

        在每次推理步骤后调用，负责：
        1. 将新生成的 token 追加到序列
        2. 检查序列是否完成（达到 EOS 或 max_tokens）
        3. 释放完成序列的资源

        参数：
            seqs: 刚刚执行推理的序列列表
            token_ids: 模型输出的 token ID 列表（与 seqs 一一对应）

        流程：
            模型推理 → 采样得到 token_ids → postprocess() 更新状态

        完成条件：
        1. 生成了 EOS token（如果 ignore_eos=False）
        2. 达到 max_tokens 限制
        """
        for seq, token_id in zip(seqs, token_ids):
            # 追加新生成的 token 到序列
            seq.append_token(token_id)

            # 检查是否完成生成
            # 条件1：遇到 EOS token（且不忽略 EOS）
            # 条件2：达到最大生成长度
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED  # 更新状态：运行 → 完成
                self.block_manager.deallocate(seq)  # 释放 KV Cache 块
                self.running.remove(seq)  # 从运行队列移除
                # 注意：完成的序列不会被添加回任何队列，将被引擎回收
