"""
归一化层：RMSNorm 与 QK-Norm
============================

RMSNorm（Zhang & Sennrich, 2019）
---------------------------------
    RMS(x) = sqrt( (1/d)·Σ x_i² + ε )
    out    = (x / RMS(x)) ⊙ g

与 LayerNorm 的关键差别：不做去均值（无 μ）、不做平移（无 β），只做**重缩放**。
数学后果：RMSNorm 是关于输入的**齐次函数**（f(αx) = f(x)·sign(α) 尺度不变），
这带来两个好处——(1) 少两次全量规约，更快；(2) 尺度不变性让残差流的范数增长更可控，
是 2026 年全系模型（Llama/Qwen/DeepSeek/Kimi/Gemma/gpt-oss）无一例外选择它的原因。
本文件里 LayerNorm 一行都不会出现——它已经退场。

**证书**：RMSNorm 的齐次性是机器可检的不变量
    ‖RMSNorm(α·x)‖₂ ≈ ‖RMSNorm(x)‖₂ · |α| / ‖x‖₂ · ‖x‖₂  …… 更直接地写：
    RMSNorm(x) / ‖RMSNorm(x)‖₂ == x / ‖x‖₂   （方向保持，仅长度归一）
由 cert.invariants 检查。

QK-Norm（2026 稳定性标配）
-------------------------
    对 query/key 的头向量施加 RMSNorm（Qwen 3 / Gemma 3 / OLMo 2 采用）。
    作用是给注意力 logit q·kᵀ/√d 加一道硬上界，防止训练中期 logit 爆炸。
    Kimi K2 走的是另一条路：QK-Clip（对超出阈值的 logit 直接缩放），本仓库两者都实现。
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["RMSNorm", "QKNorm"]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization。

    Args:
        dim: 归一化维度。
        eps: 数值稳定项。注意：ε 加在**根号内**（先平方和再 +ε·d 的等价写法），
             这与原论文一致；若加在根号外会在 x→0 时产生梯度尖峰。
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 低精度输入提升到 fp32 做规约：fp16/bf16 下 Σx² 极易溢出
        # （bf16 最大约 3.4e38 尚可，但 fp16 上界仅 65504，(2^8)²·512 就已越界）
        # 注意 fp64 必须**保持** fp64：无条件 .float() 会把双精度静默降到单精度，
        # 使梯度检验的残差从 1e-9 恶化到 1e-2 —— 这个坑由 GRAD 证书 GRAD-01/02 抓到。
        dtype = x.dtype
        xf = x.float() if dtype in (torch.float16, torch.bfloat16) else x
        ms = xf.pow(2).mean(dim=-1, keepdim=True)
        out = xf * torch.rsqrt(ms + self.eps)
        weight = self.weight.to(xf.dtype) if xf.dtype != dtype else self.weight
        return (out * weight).to(dtype)

    def extra_repr(self) -> str:  # pragma: no cover - 仅美化打印
        return f"dim={self.dim}, eps={self.eps}"


class QKNorm(nn.Module):
    """对 query / key 的头向量做 RMSNorm。

    数学上它把注意力 logit 夹在一个跟 ‖q‖‖k‖ 有关的界内：
        |q̂·k̂ᵀ| ≤ ‖q̂‖·‖k̂‖ ≤ √d_qk · √d_qk = d_qk   （RMSNorm 后每头范数 ≈ √d）
    再除以 √d 缩放，logit 的量级就被锁死在 O(√d)，而不是随训练无界增长。
    """

    def __init__(self, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., head_dim]（精度处理同上，fp64 透传不降精度）
        dtype = x.dtype
        xf = x.float() if dtype in (torch.float16, torch.bfloat16) else x
        ms = xf.pow(2).mean(dim=-1, keepdim=True)
        out = xf * torch.rsqrt(ms + self.eps)
        weight = self.weight.to(xf.dtype) if xf.dtype != dtype else self.weight
        return (out * weight).to(dtype)

    def extra_repr(self) -> str:  # pragma: no cover
        return f"head_dim={self.head_dim}, eps={self.eps}"
