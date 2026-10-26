"""
SwiGLU 前馈网络
===============

来源：Shazeer (2020) GLU Variants，被 2026 年全部主流模型采用
（Llama / Qwen / DeepSeek / Kimi / Gemma / Ministral / gpt-oss）。

    FFN(x) = ( SiLU(x W_gate) ⊙ (x W_up) ) W_down

为什么 SwiGLU 赢了
------------------
· 相比 ReLU FFN：门控机制让**信息通路可控**（门接近 0 时直接阻断），
  梯度路径更平滑，同等参数下困惑度更低。
· 相比 GELU-GLU：SiLU(x)=x·σ(x) 处处可微且无上界，没有 GELU 的近似误差。
· 工程细节：W_gate 与 W_up 合并成一次 [D → 2I] 的 matmul，
  把两次 kernel 启动合成一次，实打实的吞吐提升。

参数量：3·D·I（gate/up/down 各一份），这也是 FFN 占 Transformer 参数大头的原因。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SwiGLU"]


class SwiGLU(nn.Module):
    """门控线性单元前馈层。"""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.w_gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gu = self.w_gate_up(x)
        g, u = gu.chunk(2, dim=-1)
        return self.w_down(self.drop(F.silu(g) * u))

    def extra_repr(self) -> str:  # pragma: no cover
        return f"dim={self.dim}, hidden={self.hidden}"
