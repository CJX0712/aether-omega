"""
Newton-Schulz 正交化：原版 与 Gram 版（2026 最新成果）
=======================================================

问题
----
Muon 优化器需要把动量矩阵 G ∈ R^{m×n} 变成"最接近的半正交矩阵"，
即求 polar 分解 G = U P 中的正交因子 U（UᵀU = I）。
精确做法：SVD，G = UΣVᵀ，取 U Vᵀ。但 SVD 每次 step 都做一遍太贵，
且 SVD 在 GPU 上效率低下。

Newton-Schulz（Higham 1986 / Björck & Bowie 1971）
--------------------------------------------------
用迭代代替分解。核心迭代（作用在 X 上，X 已缩放到谱半径 < 1）：

    X ← a·X + b·(XXᵀ)X + c·(XXᵀ)²X        ……即 X ← (aI + bG + cG²)·X，G = XXᵀ

系数 (a,b,c) = (3.4445, −4.7750, 2.0315) 由 Keller Jordan 团队调优：
这个五次多项式能在 **5 步内**于 bf16 下稳定收敛，而 SVD 需要显式分解。
原理：该迭代是求 G^{-1/2} 的牛顿法变体，X_∞ = X (XᵀX)^{-1/2}，正好是 polar 分解的正交因子。

为什么除以 Frobenius 范数：保证 ‖X‖₂ ≤ ‖X‖_F = 1，即所有奇异值落在 [0,1]，
多项式迭代在这个区间内单调地把奇异值推向 1（小奇异值被快速拉升）。

**Gram Newton-Schulz（本文件的核心，2026 年最新改进）**
------------------------------------------------------
观察：上面的迭代每步都要做两次**矩形**矩阵乘（m×n 与 n×m 或 m×n）：
    A = X Xᵀ        O(m²n)
    X ← a X + B X   O(m²n)   （B 是 m×m）
5 步就是 10 次 O(m²n)。在大模型里 m·n 巨大，这是 Muon 单步开销的主要来源。

但注意迭代的代数结构：令 P_k = aI + b G_k + c G_k²，G_k = X_k X_kᵀ，则

    X_{k+1} = P_k X_k
    G_{k+1} = X_{k+1} X_{k+1}ᵀ = P_k G_k P_kᵀ = P_k² G_k      （P_k 是 G_k 的多项式 ⇒ 对称且可交换）

**整个迭代可以在 m×m 的小对称 Gram 矩阵上完成**，与 n 完全无关！
m 通常远小于 n（例如 4096×11008 的 FFN，m=4096, n=11008），
m×m 的乘法比 m×n 便宜一个量级。最后只需一次 P_total X_0 把结果拉回原空间：

    原版：  5 × (1 次 X Xᵀ + 1 次 B X) = 10 次 O(m²n) 矩形乘
    Gram：  1 次 X Xᵀ + 5 次 m×m 小乘 + 1 次 P X = 2 次 O(m²n) 矩形乘

这就是论文报告的 40–50% 正交化提速的来源，而且**数学上严格等价**
（不只是"差不多"，是代数恒等）——本文件由 cert.equivalence 机器验证。

数值稳定性与 Gram Reset
-----------------------
Gram 版迭代的是 G 的高次多项式（G⁵ 量级），在中精度（bf16/fp16）下
误差会被多项式放大，比原版更容易漂移。论文给出的对策是 **Gram reset**：
每隔若干步用当前的 X 重新计算 G = X Xᵀ，把累积误差清零。
本实现把它做成 `gram_reset_every` 参数，并保留 stabilized 版本。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = [
    "NS_COEFFS",
    "newton_schulz_original",
    "gram_newton_schulz",
    "zeropower",
    "orthogonality_residual",
    "gram_vs_original_residual",
]

# Keller Jordan 调优的五次多项式系数（bf16 下 5 步收敛）
NS_COEFFS: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315)


def _prep(G: torch.Tensor, eps: float):
    """统一处理：转置使行数 ≤ 列数（正交化只需处理"高"矩阵），并缩放到谱半径 ≤ 1。"""
    transposed = G.size(-2) > G.size(-1)
    X = G.mT if transposed else G
    X = X / (X.norm(dim=(-2, -1), keepdim=True).clamp_min(eps))
    return X, transposed


def newton_schulz_original(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coeffs: Tuple[float, float, float] = NS_COEFFS,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """原版 Newton-Schulz：在 X 上迭代。作为 Gram 版的黄金参照实现。

    dtype=None 表示沿用输入 dtype（双精度输入不得被静默降为单精度）；
    Muon 调用时显式指定 float32，以保证半精度训练下正交化的数值稳定。

    返回与 G 同形状、且满足 XXᵀ ≈ I（行正交）的矩阵。
    """
    a, b, c = coeffs
    X, transposed = _prep(G, eps)
    X = X if dtype is None else X.to(dtype)

    for _ in range(int(steps)):
        A = X @ X.mT                    # [m,m] Gram
        B = b * A + c * (A @ A)         # 二次多项式
        X = a * X + B @ X               # [m,n]

    return X.mT if transposed else X


def gram_newton_schulz(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coeffs: Tuple[float, float, float] = NS_COEFFS,
    gram_reset_every: int = 0,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Gram Newton-Schulz：在 m×m 的对称 Gram 矩阵上迭代，矩形乘次数从 ~10 降到 2。

    数学等价性推导（这就是证书要机器检查的那条恒等式）：
        令 P_k = aI + b G_k + c G_k²，G_{k+1} = P_k G_k P_kᵀ = P_k² G_k
        X_n = (P_{n-1} · … · P_0) X_0
    与原版逐位相等，差别只来自浮点结合律。

    Args:
        gram_reset_every: 每隔多少步用当前 X 重算 Gram（0=不重置）。
            中精度下建议设为 2 或 3，把多项式放大的误差清零。
    """
    a, b, c = coeffs
    X, transposed = _prep(G, eps)
    X = X if dtype is None else X.to(dtype)
    m = X.size(-2)

    eye = torch.eye(m, dtype=X.dtype, device=X.device)
    Gk = X @ X.mT                       # 唯一一次"大"矩形乘（构造 Gram）
    P = eye.clone()

    for k in range(int(steps)):
        if gram_reset_every and k > 0 and (k % gram_reset_every) == 0:
            # 把已累积的变换落回 X，再用真实 X 重算 Gram → 误差清零
            X = P @ X
            P = eye.clone()
            Gk = X @ X.mT

        G2 = Gk @ Gk
        Pk = a * eye + b * Gk + c * G2  # [m,m] 小矩阵
        Gk = Pk @ Gk @ Pk.mT            # = Pk² Gk（Pk 与 Gk 可交换且对称）
        P = Pk @ P

    X = P @ X                           # 第二次（也是最后一次）矩形乘
    return X.mT if transposed else X


def zeropower(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    backend: str = "gram",
    gram_reset_every: int = 0,
    dtype: Optional[torch.dtype] = None,
    coeffs: Tuple[float, float, float] = NS_COEFFS,
) -> torch.Tensor:
    """正交化的统一入口。backend ∈ {"gram"（默认，更快）, "original"（参照实现）}。"""
    if backend == "gram":
        return gram_newton_schulz(
            G, steps, eps, coeffs, gram_reset_every=gram_reset_every, dtype=dtype
        )
    if backend == "original":
        return newton_schulz_original(G, steps, eps, coeffs, dtype=dtype)
    raise ValueError(f"未知 backend {backend!r}，可选 'gram' / 'original'")


@torch.no_grad()
def orthogonality_residual(X: torch.Tensor) -> float:
    """正交性残差 ‖XXᵀ − I‖_max。

    这是 Newton-Schulz 的**收敛证书**：越接近 0 说明越接近真正的半正交矩阵。
    fp32 下 5 步迭代后应落在 1e-2 ~ 1e-3 量级（迭代本身是近似的，
    追求的不是精确实正交，而是"方向均衡"的更新）。
    """
    Xf = X.detach().float()
    if Xf.size(-2) > Xf.size(-1):
        Xf = Xf.mT
    G = Xf @ Xf.mT
    m = G.size(-1)
    return float((G - torch.eye(m, dtype=G.dtype, device=G.device)).abs().max().item())


@torch.no_grad()
def gram_vs_original_residual(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    dtype: Optional[torch.dtype] = None,
) -> float:
    """Gram 版与原版输出的最大绝对差 —— 数学等价性的机器凭证。"""
    o = newton_schulz_original(G, steps=steps, eps=eps, dtype=dtype)
    g = gram_newton_schulz(G, steps=steps, eps=eps, dtype=dtype)
    return float((o.float() - g.float()).abs().max().item())
