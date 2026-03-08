import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """
    LLM 推理引擎类 (LLM Engine Class)

    最顶层的推理引擎，协调所有组件完成 LLM 推理任务。
    这是用户与推理系统交互的主要接口。

    核心组件：
    1. Scheduler: 管理请求队列和调度策略
    2. ModelRunner: 执行实际的模型推理（支持多 GPU 张量并行）
    3. Tokenizer: 文本与 token ID 的转换

    工作流程：
    1. 初始化：启动多进程（张量并行）、加载模型、分配 KV Cache
    2. 添加请求：用户提交 prompt，转换为 Sequence 对象
    3. 执行推理：循环调用 step()，每步生成一批 tokens
    4. 返回结果：收集完成的序列，解码为文本

    多进程架构（张量并行）：
    - Rank 0（主进程）：运行完整引擎逻辑，管理调度
    - Rank 1-N（工作进程）：只运行模型推理，等待主进程指令
    - 通信：通过共享内存 (SharedMemory) 传递序列数据
    """

    def __init__(self, model, **kwargs):
        """
        初始化 LLM 推理引擎

        参数：
            model: 模型路径（HuggingFace 格式）
            **kwargs: 配置参数，包括：
                - tensor_parallel_size: 张量并行的 GPU 数量
                - max_num_seqs: 最大批处理序列数
                - enforce_eager: 是否禁用 CUDA Graph
                - gpu_memory_utilization: GPU 内存利用率
                等（详见 Config 类）

        初始化流程：
        1. 解析配置参数
        2. 启动多进程（如果 tensor_parallel_size > 1）
        3. 初始化 ModelRunner（加载模型、分配 KV Cache）
        4. 加载 Tokenizer
        5. 初始化 Scheduler
        6. 注册退出清理函数
        """
        # 从 kwargs 中提取 Config 相关的参数
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # 初始化多进程列表和事件列表（用于张量并行）
        self.ps = []  # 工作进程列表
        self.events = []  # 同步事件列表（用于进程间通信）

        # 获取 spawn 上下文（避免 fork 的问题，特别是在 CUDA 环境）
        ctx = mp.get_context("spawn")

        # 启动工作进程（Rank 1 到 N-1）
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()  # 创建同步事件
            # 启动 ModelRunner 进程（每个进程运行在不同的 GPU 上）
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # 初始化主进程的 ModelRunner（Rank 0）
        # 传入 events 用于通知工作进程
        self.model_runner = ModelRunner(config, 0, self.events)

        # 加载 Tokenizer（用于 prompt 编码和输出解码）
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)

        # 更新配置中的 EOS token ID（从 tokenizer 获取）
        config.eos = self.tokenizer.eos_token_id

        # 初始化调度器（管理请求队列和资源分配）
        self.scheduler = Scheduler(config)

        # 注册退出清理函数（确保进程正确关闭）
        atexit.register(self.exit)

    def exit(self):
        """
        清理资源，关闭所有进程

        在程序退出时自动调用（通过 atexit）
        或手动调用（用于显式清理）

        清理步骤：
        1. 通知所有进程退出
        2. 删除 ModelRunner（释放 GPU 内存）
        3. 等待所有工作进程结束
        """
        # 通知所有进程退出
        self.model_runner.call("exit")
        # 删除 ModelRunner（触发 __del__，释放资源）
        del self.model_runner
        # 等待所有工作进程结束
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        添加一个推理请求到调度器

        参数：
            prompt: 输入提示词，可以是字符串或 token ID 列表
            sampling_params: 采样参数（温度、最大长度等）

        流程：
        1. 如果 prompt 是字符串，使用 tokenizer 编码为 token IDs
        2. 创建 Sequence 对象
        3. 添加到调度器的等待队列

        注意：
            此方法只是将请求加入队列，实际推理在 step() 中执行
        """
        if isinstance(prompt, str):
            # 字符串 prompt：编码为 token IDs
            prompt = self.tokenizer.encode(prompt)
        # 创建序列对象（包含 token IDs 和采样参数）
        seq = Sequence(prompt, sampling_params)
        # 添加到调度器的等待队列
        self.scheduler.add(seq)

    def step(self):
        """
        执行一步推理（核心方法）

        一步推理包括：
        1. 调度：选择要执行的序列（prefill 或 decode）
        2. 推理：运行模型得到 logits
        3. 采样：从 logits 中采样得到 token IDs
        4. 后处理：更新序列状态，检查是否完成

        返回：
            (outputs, num_tokens)
            - outputs: 完成的序列列表，格式为 [(seq_id, token_ids), ...]
            - num_tokens: 本步处理的 token 数
                > 0: prefill 阶段，值为实际处理的 token 数
                < 0: decode 阶段，值为 -序列数（用于区分阶段）

        性能指标：
            - Prefill 吞吐量：tokens/s（并行处理多个 tokens）
            - Decode 吞吐量：sequences/s（自回归，每序列一个 token）
        """
        # 步骤1：调度 - 选择要执行的序列
        seqs, is_prefill = self.scheduler.schedule()

        # 步骤2 & 3：推理和采样 - 运行模型并采样 tokens
        # ModelRunner 内部处理：模型前向 → 采样 → 返回 token IDs
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # 步骤4：后处理 - 更新序列状态
        self.scheduler.postprocess(seqs, token_ids)

        # 收集完成的序列（状态为 FINISHED）
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        # 计算本步处理的 token 数（用于统计吞吐量）
        if is_prefill:
            # Prefill: 计算实际处理的 token 数（多个序列的总和）
            num_tokens = sum(len(seq) for seq in seqs)
        else:
            # Decode: 返回负的序列数（用于区分 prefill 和 decode）
            num_tokens = -len(seqs)

        return outputs, num_tokens

    def is_finished(self):
        """
        检查是否所有请求都已完成

        返回：
            True 如果没有待处理或正在处理的请求，False 否则

        用途：
            在主循环中判断是否可以退出
        """
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        批量生成文本（主要的用户接口）

        参数：
            prompts: prompt 列表，可以是字符串或 token ID 列表
            sampling_params: 采样参数，可以是单个参数（应用到所有 prompts）
                           或参数列表（与 prompts 一一对应）
            use_tqdm: 是否显示进度条

        返回：
            生成结果列表，每个元素是一个字典：
            {
                "text": 生成的文本（解码后）,
                "token_ids": 生成的 token ID 列表
            }

        工作流程：
        1. 初始化进度条（可选）
        2. 添加所有请求到调度器
        3. 循环执行 step() 直到所有请求完成
        4. 收集和解码结果
        5. 按 seq_id 排序返回（保持输入顺序）

        性能监控：
        - Prefill 吞吐量：tokens/s
        - Decode 吞吐量：tokens/s
        """
        # 初始化进度条
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        # 统一采样参数格式（如果是单个参数，复制到所有 prompts）
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 添加所有请求到调度器
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        # 初始化输出字典（seq_id -> token_ids）
        outputs = {}
        # 初始化吞吐量统计
        prefill_throughput = decode_throughput = 0.

        # 主循环：持续执行推理直到所有请求完成
        while not self.is_finished():
            t = perf_counter()  # 记录开始时间

            # 执行一步推理
            output, num_tokens = self.step()

            # 更新进度条（显示吞吐量）
            if use_tqdm:
                if num_tokens > 0:
                    # Prefill 阶段：计算 tokens/s
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    # Decode 阶段：计算 tokens/s（负数转正）
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })

            # 收集完成的序列
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)  # 更新进度条

        # 按 seq_id 排序输出（保持输入顺序）
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]

        # 解码 token IDs 为文本
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]

        # 关闭进度条
        if use_tqdm:
            pbar.close()

        return outputs
