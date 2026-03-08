from dataclasses import dataclass
import torch


@dataclass
class Context:
    """
    推理上下文类 (Context Class)

    存储当前推理步骤的全局状态，供注意力层使用。
    这种设计避免了在每层之间传递大量参数。

    属性说明：
        is_prefill: 是否为 prefill 阶段（True）或 decode 阶段（False）

        Prefill 阶段参数（使用 Flash Attention varlen 接口）：
            cu_seqlens_q: Query 的累积序列长度，形状 [batch_size + 1]
            cu_seqlens_k: Key 的累积序列长度，形状 [batch_size + 1]
            max_seqlen_q: Query 的最大序列长度
            max_seqlen_k: Key 的最大序列长度

        Decode 阶段参数（使用 Flash Attention KV Cache 接口）：
            context_lens: 每个序列的长度，形状 [batch_size]

        共用参数：
            slot_mapping: Token 到 KV Cache 位置的映射
                - Prefill: 所有需要存储的 token 位置
                - Decode: 每个序列最后一个 token 的位置
            block_tables: 块表，记录每个序列的 KV Cache 块 ID
                形状 [batch_size, max_num_blocks]

    工作流程：
    1. ModelRunner 在 prepare_prefill/decode 中设置上下文
    2. 注意力层通过 get_context() 获取上下文
    3. ModelRunner 在推理结束后重置上下文
    """
    is_prefill: bool = False  # 是否为 prefill 阶段
    cu_seqlens_q: torch.Tensor | None = None  # Query 累积序列长度（prefill）
    cu_seqlens_k: torch.Tensor | None = None  # Key 累积序列长度（prefill）
    max_seqlen_q: int = 0  # Query 最大序列长度（prefill）
    max_seqlen_k: int = 0  # Key 最大序列长度（prefill）
    slot_mapping: torch.Tensor | None = None  # Token 到 KV Cache 的映射
    context_lens: torch.Tensor | None = None  # 每个序列的长度（decode）
    block_tables: torch.Tensor | None = None  # 块表（Paged Attention）


# 全局上下文实例（线程不安全，但在推理场景中单线程使用）
_CONTEXT = Context()


def get_context():
    """
    获取当前的推理上下文

    返回：
        全局 Context 对象

    用途：
        在注意力层中获取当前推理步骤的参数
    """
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """
    设置推理上下文

    参数：
        is_prefill: 是否为 prefill 阶段
        其他参数：对应 Context 的属性

    用途：
        在 ModelRunner.prepare_prefill/decode 中调用，
        为即将执行的推理步骤设置上下文参数
    """
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)


def reset_context():
    """
    重置推理上下文

    用途：
        在推理步骤结束后调用，清理上下文状态
        避免残留数据影响下一次推理
    """
    global _CONTEXT
    _CONTEXT = Context()
