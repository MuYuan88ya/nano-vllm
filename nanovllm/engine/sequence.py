from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    序列状态枚举类 (Sequence Status Enum)

    定义了推理请求在其生命周期中的三个状态：
    - WAITING: 等待调度和执行
    - RUNNING: 正在执行推理
    - FINISHED: 已完成生成

    状态转换流程：WAITING → RUNNING → FINISHED
    注意：当资源不足时，RUNNING 状态可能被抢占回到 WAITING 状态
    """
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """
    序列类 (Sequence Class)

    表示单个推理请求，跟踪其完整的生命周期和状态。
    每个 Sequence 对象包含：
    - 输入提示词的 tokens
    - 已生成的 tokens
    - KV Cache 的内存块分配信息
    - 采样参数和执行状态

    核心概念：
    1. Token 管理：维护完整的 token 序列（prompt + completion）
    2. 块表 (Block Table)：记录分配的 KV Cache 块 ID，用于分页注意力
    3. 前缀缓存：通过 num_cached_tokens 跟踪已缓存的 tokens
    4. 状态跟踪：管理从 WAITING 到 FINISHED 的完整状态转换
    """

    # 类级别常量：KV Cache 块的大小（每个块包含 256 个 tokens 的 KV）
    block_size = 256

    # 全局序列计数器，用于生成唯一的序列 ID
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """
        初始化一个新的推理序列

        参数：
            token_ids: 输入提示词的 token ID 列表（来自 tokenizer 编码）
            sampling_params: 采样参数，控制生成行为（温度、最大长度等）

        初始化的属性：
            seq_id: 唯一序列标识符，用于跟踪和管理请求
            status: 初始状态为 WAITING（等待调度）
            token_ids: token 序列的副本，会随着生成过程不断追加新 tokens
            last_token: 最后一个 token，用于快速访问（解码阶段每次只需最后一个）
            num_tokens: 当前总 token 数（prompt + 已生成的）
            num_prompt_tokens: 原始提示词的 token 数，用于区分 prompt 和 completion
            num_cached_tokens: 已缓存的 token 数（用于前缀缓存优化）
            block_table: KV Cache 块 ID 列表，实现分页内存管理
            temperature: 采样温度，控制生成的随机性
            max_tokens: 最大生成 token 数
            ignore_eos: 是否忽略 EOS token（用于强制生成到 max_tokens）
        """
        self.seq_id = next(Sequence.counter)  # 分配唯一 ID
        self.status = SequenceStatus.WAITING  # 初始状态：等待调度
        self.token_ids = copy(token_ids)      # 深拷贝避免外部修改
        self.last_token = token_ids[-1]      # 缓存最后一个 token
        self.num_tokens = len(self.token_ids)  # 当前 token 总数
        self.num_prompt_tokens = len(token_ids)  # prompt 长度（不变）
        self.num_cached_tokens = 0  # 前缀缓存：已缓存的 token 数
        self.block_table = []  # KV Cache 块表，初始为空（未分配）
        self.temperature = sampling_params.temperature  # 采样温度
        self.max_tokens = sampling_params.max_tokens  # 最大生成长度
        self.ignore_eos = sampling_params.ignore_eos  # 是否忽略 EOS

    def __len__(self):
        """返回序列的总 token 数（prompt + completion）"""
        return self.num_tokens

    def __getitem__(self, key):
        """支持索引和切片访问 token_ids，如 seq[0] 或 seq[:10]"""
        return self.token_ids[key]

    @property
    def is_finished(self):
        """判断序列是否已完成生成"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已生成的 token 数（不包括 prompt）"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """获取原始提示词的 token 列表"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """获取已生成的 token 列表（不包括 prompt）"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """
        已缓存的完整块数量
        用于前缀缓存：跟踪哪些块的 KV 已经被缓存并可以复用
        """
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """
        当前需要的总块数（向上取整）
        例如：257 个 tokens，block_size=256，需要 2 个块
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """
        最后一个块中的 token 数量（可能不满一个块）
        例如：257 个 tokens，block_size=256，最后一块有 1 个 token
        用于计算 slot mapping 时确定最后一个块的有效范围
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """
        获取第 i 个块的 token 列表

        参数：
            i: 块索引（0-based）

        返回：
            该块包含的 token 列表（最多 block_size 个）

        用途：
            - 在 BlockManager 中计算块的哈希值（用于前缀缓存）
            - 验证缓存命中时 token 序列是否完全匹配
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """
        追加新生成的 token（解码阶段每步调用一次）

        参数：
            token_id: 新生成的 token ID

        更新：
            - 将 token 添加到 token_ids 列表
            - 更新 last_token 缓存
            - 增加 token 计数
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        序列化状态（用于多进程通信）

        优化：在解码阶段，只传输 last_token 而不是完整的 token_ids
        这大幅减少了进程间通信的数据量

        返回：
            - num_tokens, num_prompt_tokens, num_cached_tokens, block_table（始终需要）
            - token_ids（prefill 阶段）或 last_token（decode 阶段）
        """
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        """
        反序列化状态（用于多进程通信）

        从序列化数据恢复序列对象，区分 prefill 和 decode 阶段
        """
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            # Prefill 阶段：恢复完整的 token_ids
            self.token_ids = state[-1]
        else:
            # Decode 阶段：只恢复 last_token
            # 注意：完整的 token_ids 在主进程中维护，工作进程只需要 last_token
            self.last_token = state[-1]
