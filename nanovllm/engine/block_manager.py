from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """
    KV Cache 块类 (Block Class)

    表示一个固定大小的 KV Cache 块，用于分页内存管理。
    每个块可以存储固定数量（block_size）个 tokens 的 Key 和 Value。

    核心属性：
        block_id: 块的唯一标识符（在 KV Cache 张量中的索引）
        ref_count: 引用计数，跟踪有多少个序列正在使用这个块
        hash: 块内容的哈希值，用于前缀缓存的快速查找
        token_ids: 块内存储的 token ID 列表（用于精确匹配验证）

    引用计数机制：
        - ref_count > 0: 块正在被使用，不能释放
        - ref_count == 0: 块空闲，可以被分配给新序列
        - 多个序列可以共享同一个块（前缀缓存），每个引用增加计数
    """

    def __init__(self, block_id):
        """
        初始化一个 KV Cache 块

        参数：
            block_id: 块在 KV Cache 张量中的索引位置
        """
        self.block_id = block_id  # 块 ID（唯一标识）
        self.ref_count = 0  # 引用计数：当前有多少序列使用这个块
        self.hash = -1  # 哈希值：-1 表示未计算或不完整
        self.token_ids = []  # token 序列：用于精确匹配验证

    def update(self, hash: int, token_ids: list[int]):
        """
        更新块的哈希值和 token 内容

        当块被完全填满（达到 block_size）时调用，计算哈希用于前缀缓存

        参数：
            hash: 块内容的哈希值（基于 token_ids 和前缀哈希）
            token_ids: 块内存储的完整 token 列表
        """
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """
        重置块状态，准备分配给新序列

        注意：将 ref_count 设为 1（表示有一个新序列使用）
        """
        self.ref_count = 1
        self.hash = -1  # 清除哈希，因为内容将被新数据覆盖
        self.token_ids = []  # 清除旧的 token 数据


class BlockManager:
    """
    块管理器类 (Block Manager Class)

    实现分页内存管理和前缀缓存，是 vLLM 的核心创新之一。

    核心功能：
    1. 分页内存管理：类似操作系统的虚拟内存，将 KV Cache 划分为固定大小的块
    2. 前缀缓存：通过哈希识别相同的 token 序列，实现块级别的共享
    3. 引用计数：支持多个序列共享同一个块，自动回收未使用的块

    数据结构：
        blocks: 所有块的列表（固定大小，预分配）
        free_block_ids: 空闲块 ID 队列（FIFO）
        used_block_ids: 已使用块 ID 集合
        hash_to_block_id: 哈希到块 ID 的映射（前缀缓存索引）

    前缀缓存原理：
        两个序列如果有相同的前缀（如共享 system prompt），它们的 KV Cache
        也完全相同，可以共享同一组块，大幅节省内存并减少重复计算。

    示例：
        序列1: [A, B, C, D, E, F, G, H]
        序列2: [A, B, C, D, X, Y, Z]

        块划分 (block_size=4):
        序列1: [A,B,C,D] [E,F,G,H]
        序列2: [A,B,C,D] [X,Y,Z]

        结果：两个序列共享第一个块 [A,B,C,D]，节省一半内存
    """

    def __init__(self, num_blocks: int, block_size: int):
        """
        初始化块管理器

        参数：
            num_blocks: 总块数（由可用 GPU 内存决定）
            block_size: 每个块的大小（token 数量，通常为 256）
        """
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()  # 哈希索引：快速查找缓存
        self.free_block_ids: deque[int] = deque(range(num_blocks))  # 空闲块队列
        self.used_block_ids: set[int] = set()  # 已使用块集合

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算 token 序列的哈希值（用于前缀缓存）

        使用 xxhash 算法，速度快且冲突率低。
        支持增量计算：新块的哈希依赖于前一个块的哈希（形成哈希链）

        参数：
            token_ids: 要计算哈希的 token 列表
            prefix: 前一个块的哈希值（-1 表示第一个块）

        返回：
            哈希值（64位整数）

        哈希链示例：
            Block 0: hash([A,B,C,D])
            Block 1: hash(hash0 + [E,F,G,H])
            Block 2: hash(hash1 + [I,J,K,L])
            这确保了即使 token 内容相同但位置不同，哈希也会不同
        """
        h = xxhash.xxh64()
        if prefix != -1:
            # 包含前缀哈希，形成哈希链
            h.update(prefix.to_bytes(8, "little"))
        # 将 token_ids 转换为字节并计算哈希
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """
        分配一个空闲块（内部方法）

        从空闲队列中取出块，标记为已使用，并初始化引用计数

        参数：
            block_id: 要分配的块 ID

        返回：
            分配的块对象
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0, "试图分配一个正在使用的块"
        block.reset()  # 重置块状态，ref_count 设为 1
        self.free_block_ids.remove(block_id)  # 从空闲队列移除
        self.used_block_ids.add(block_id)  # 加入已使用集合
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """
        释放一个块（内部方法）

        将块标记为空闲，可以被新序列使用

        参数：
            block_id: 要释放的块 ID
        """
        assert self.blocks[block_id].ref_count == 0, "试图释放一个仍在使用的块"
        self.used_block_ids.remove(block_id)  # 从已使用集合移除
        self.free_block_ids.append(block_id)  # 加入空闲队列

    def can_allocate(self, seq: Sequence) -> bool:
        """
        检查是否有足够的空闲块来满足序列的需求

        参数：
            seq: 要检查的序列

        返回：
            True 如果空闲块足够，False 否则
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        为序列分配 KV Cache 块（支持前缀缓存）

        核心算法：
        1. 遍历序列需要的每个块
        2. 计算块的哈希值（只对完整块）
        3. 查找哈希表，检查是否有缓存命中
        4. 如果命中且 token 完全匹配，共享该块（引用计数 +1）
        5. 如果未命中，分配新块

        前缀缓存的两级验证：
        - 一级：哈希匹配（快速排除大部分不匹配）
        - 二级：token 精确匹配（避免哈希冲突导致的错误）

        参数：
            seq: 要分配块的序列

        副作用：
            - 更新 seq.block_table（分配的块 ID 列表）
            - 更新 seq.num_cached_tokens（缓存命中的 token 数）
            - 更新块的引用计数和哈希索引
        """
        assert not seq.block_table, "序列已经分配过块，不能重复分配"
        h = -1  # 前缀哈希，初始为 -1
        cache_miss = False  # 缓存是否失效标志

        # 遍历序列需要的每个块
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)  # 获取第 i 个块的 tokens
            # 只对完整块计算哈希（最后一个块可能不完整）
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1

            # 查找哈希表，检查是否有缓存
            block_id = self.hash_to_block_id.get(h, -1)

            # 验证缓存：哈希匹配 + token 精确匹配
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True  # 缓存未命中

            if cache_miss:
                # 缓存未命中：分配新块
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 缓存命中：共享现有块
                seq.num_cached_tokens += self.block_size  # 增加缓存计数
                if block_id in self.used_block_ids:
                    # 块已被其他序列使用，增加引用计数
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 块在哈希表中但未被使用（之前被释放），重新分配
                    block = self._allocate_block(block_id)

            # 更新块的哈希和 token 信息（用于后续匹配）
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id

            # 将块 ID 添加到序列的块表
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """
        释放序列占用的所有块

        使用引用计数：
        - 如果 ref_count > 1，只减少计数（其他序列仍在使用）
        - 如果 ref_count == 1，减为 0 后释放块

        参数：
            seq: 要释放块的序列

        副作用：
            - 清空 seq.block_table
            - 重置 seq.num_cached_tokens
            - 更新块的引用计数，释放未使用的块
        """
        # 倒序遍历块表（通常后面的块更可能独占）
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1  # 减少引用计数
            if block.ref_count == 0:
                # 没有序列使用这个块了，释放它
                self._deallocate_block(block_id)

        # 清空序列的块相关信息
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        检查是否可以为序列追加新 token

        在解码阶段，每生成一个新 token，可能需要分配新块：
        - 如果当前块已满（token 数 % block_size == 1），需要分配新块
        - 否则，可以继续使用当前块

        参数：
            seq: 要追加 token 的序列

        返回：
            True 如果可以追加（有足够空闲块或当前块未满）
        """
        # 如果最后一个块只有 1 个 token，说明之前的块已满，需要新块
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        为序列追加 token 做准备（可能分配新块）

        在解码阶段调用，处理三种情况：
        1. 最后一个块刚满（token 数 % block_size == 1）：分配新块
        2. 最后一个块刚好填满（token 数 % block_size == 0）：更新哈希
        3. 最后一个块未满（其他情况）：无需操作

        参数：
            seq: 要追加 token 的序列

        副作用：
            - 可能分配新块并添加到 seq.block_table
            - 可能更新最后一个块的哈希和索引
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]

        if len(seq) % self.block_size == 1:
            # 情况1：最后一个块已满，需要分配新块
            assert last_block.hash != -1, "完整的块必须有哈希值"
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)

        elif len(seq) % self.block_size == 0:
            # 情况2：最后一个块刚好填满，计算哈希并索引
            assert last_block.hash == -1, "未完成的块不应有哈希值"
            token_ids = seq.block(seq.num_blocks-1)  # 获取最后一个块的 tokens
            # 计算哈希（包含前一个块的哈希）
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            # 更新块的哈希和索引
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id

        else:
            # 情况3：最后一个块未满，无需操作
            assert last_block.hash == -1, "未完成的块不应有哈希值"
