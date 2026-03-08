import torch
from torch import nn


class Sampler(nn.Module):
    """
    采样器类 (Sampler Class)

    从模型输出的 logits 中采样下一个 token。
    使用 Gumbel-Max 采样技巧实现高效的温度采样。

    采样方法：
    - 温度缩放：logits / temperature（控制随机性）
    - Gumbel-Max：通过 Gumbel 噪声实现采样
        sample = argmax(logits - log(-log(uniform_noise)))

    优势：
    - GPU 友好：完全向量化，无需循环
    - 支持批处理：一次采样多个序列
    - Torch compile 优化：自动融合操作，提升性能
    """

    def __init__(self):
        super().__init__()

    @torch.compile  # PyTorch 2.0 编译优化：融合操作，减少 kernel 启动
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        从 logits 中采样 tokens

        参数：
            logits: 模型输出的 logits，形状 [batch_size, vocab_size]
            temperatures: 采样温度，形状 [batch_size]
                - 温度越低，输出越确定（接近 greedy）
                - 温度越高，输出越随机

        返回：
            sample_tokens: 采样的 token IDs，形状 [batch_size]

        算法：Gumbel-Max 采样
        1. 温度缩放：logits = logits / temperature
        2. 计算概率：probs = softmax(logits)
        3. Gumbel 采样：
            - 生成 Gumbel 噪声：g = -log(-log(uniform))
            - 等价于：sample = argmax(log(probs) + g)
            - 实现：sample = argmax(probs / exp(g))

        优化：
        - 使用 exponential(1) 生成 Gumbel 分布
        - 直接除以指数噪声，避免显式计算 log
        - clamp_min 防止数值问题
        """
        # 步骤1：温度缩放（控制随机性）
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        # 步骤2：计算概率分布
        probs = torch.softmax(logits, dim=-1)

        # 步骤3：Gumbel-Max 采样
        # 生成指数分布噪声：exponential(1) ~ -log(uniform)
        # Gumbel 采样：argmax(probs / exp(noise))
        # 等价于：argmax(log(probs) - log(uniform))
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)

        return sample_tokens
