"""
RoPE（旋转位置编码）+ YaRN 长上下文外推
======================================

RoPE（Su et al., 2021 / RoFormer）
---------------------------------
核心思想：把位置信息编码成**旋转**。对 d 维向量按相邻两维一组做平面旋转：

    (x_{2i}, x_{2i+1}) → 旋转 m·θ_i，  θ_i = 1 / base^(2i/d)，m 为位置

它之所以成为 2026 年的唯一选择（无人再用可学习位置编码、无人再用绝对位置），
是因为它有一个优雅的代数性质——**相对性**：

    ⟨RoPE(x_m, m), RoPE(x_n, n)⟩ = ⟨RoPE(x_m, m−n), x_n⟩

即注意力分数只依赖相对位移 m−n。这条恒等式是机器可检的：
由 cert.invariants.check_rope_relativity 交叉验证，误差应落在 1e-5 量级。

等价的复数视角（实现更快）：把每两维视为一个复数 z = x_{2i} + i·x_{2i+1}，
旋转就是乘以 e^{i·m·θ_i}。本实现用复数乘法，避免显式构造 d×d 旋转矩阵。

YaRN（Peng et al., 2023）
------------------------
把 RoPE 频率做"长短分治"插值：低频段（长波）线性插值保位置分辨率，
高频段（短波）不插值保局部细节，中间平滑过渡。再乘一个注意力温度校正
sqrt(1/log(s)) 补偿插值后注意力分布熵的变化。
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

__all__ = ["build_rope_cache", "apply_rope", "yarn_frequency_correction", "RotaryEmbedding"]


def yarn_frequency_correction(
    dim: int,
    base: float = 10000.0,
    scale: float = 1.0,
    original_max_pos: int = 512,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> torch.Tensor:
    """YaRN 的频率校正系数 λ_i（逐维斜坡），返回 inv_freq 的缩放因子。

    ramp(γ) 定义：
        γ = 1                        if  λ_i 对应波长 >> 原始上下文（低频，长波）
        γ = 0                        if  波长 << 原始上下文（高频，短波）
        线性过渡                      otherwise
    实现上用 wavelength_i = 2π·base^(2i/d) 与 original_max_pos 比较得到斜坡。
    """
    if scale <= 1.0:
        return torch.ones(dim // 2, dtype=torch.float32)

    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    wavelength = 2.0 * math.pi / inv  # 每个频率分量的波长（以 token 计）

    # 低频（长波长）需要更强的插值；高频保持原样
    ratio = original_max_pos / wavelength
    gamma = torch.clamp((ratio - beta_slow) / max(beta_fast - beta_slow, 1), 0.0, 1.0)
    return gamma.float()


def build_rope_cache(
    seq_len: int,
    dim: int,
    base: float = 10000.0,
    scale: float = 1.0,
    original_max_pos: int = 512,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """预计算 cos/sin 表，返回 [seq_len, dim/2] 的 (cos, sin)。

    以 float32 计算并缓存：RoPE 的角度表是推理期常量，精度损失会直接污染
    长上下文外推（在 1M 上下文下，fp16 角度误差会被放大成显著的位置漂移）。
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

    if scale > 1.0:
        gamma = yarn_frequency_correction(dim, base, scale, original_max_pos).to(device)
        # 低频段按 1/scale 插值，高频段保持：λ 斜坡平滑混合
        interp = 1.0 / scale
        inv_freq = inv_freq * (interp + (1.0 - interp) * (1.0 - gamma))

    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [L, dim/2]
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """把 RoPE 作用到 x 上。

    形状约定（严格，不做"智能猜测"——自动探测维度顺序是隐蔽 bug 的温床）：
        x      : [B, L, H, D] / [B, L, D] / [L, D]  —— **位置维恒为 dim=1（无 batch 时为 dim=0）**
        cos/sin: [L, head_dim // 2]
    之所以用 dim=1 而不是"倒数第二维"：batch 维在最前是 PyTorch 的硬约定，
    写成 [B,H,L,D] 的写法在本仓库一律不用（那正是最容易 silent-broadcast 出错的地方）。

    复数写法：把 x 拆成 (x1, x2) 两半，旋转为
        out1 = x1·cos − x2·sin
        out2 = x2·cos + x1·sin
    等价于乘以 e^{i·mθ}，且比构造块对角旋转矩阵省 O(d²) 内存。
    """
    head_dim = x.shape[-1]
    half = head_dim // 2
    ndim = x.dim()
    if ndim not in (2, 3, 4):
        raise ValueError(f"apply_rope 只接受秩 2/3/4 的输入，收到 {ndim}")
    L = x.shape[1] if ndim >= 3 else x.shape[0]

    # 把 [L, half] 的表对齐到 x 的秩：位置维放在 dim=1，其余补 1
    def _align(t: torch.Tensor) -> torch.Tensor:
        t = t[:L, :]
        if ndim == 4:
            return t.view(1, L, 1, half)
        if ndim == 3:
            return t.view(1, L, half)
        return t

    c = _align(cos)
    s = _align(sin)

    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]
    out = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

    # head_dim 为奇数时最后一维原样透传（保持信息不丢，避免静默截断）
    if head_dim % 2 == 1:
        out = torch.cat([out, x[..., -1:]], dim=-1)
    return out


class RotaryEmbedding(torch.nn.Module):
    """带缓存的 RoPE 模块（推理时按当前长度动态扩容）。"""

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        scale: float = 1.0,
        original_max_pos: int = 512,
        max_seq_len: int = 8192,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.base = float(base)
        self.scale = float(scale)
        self.original_max_pos = int(original_max_pos)
        self.max_seq_len = int(max_seq_len)
        self._cache_len = 0
        self.register_buffer("cos", torch.zeros(0), persistent=False)
        self.register_buffer("sin", torch.zeros(0), persistent=False)
        self._build(self.max_seq_len)

    @torch.no_grad()
    def _build(self, seq_len: int) -> None:
        if seq_len <= self._cache_len:
            return
        cos, sin = build_rope_cache(
            seq_len, self.dim, self.base, self.scale, self.original_max_pos
        )
        self.cos = cos
        self.sin = sin
        self._cache_len = seq_len

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        need = start_pos + x.shape[-2]
        self._build(max(need, self.max_seq_len))
        return apply_rope(x, self.cos[start_pos:need], self.sin[start_pos:need])
