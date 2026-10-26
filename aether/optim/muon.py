"""
Muon 优化器 + MuonClip（QK-clip）
=================================

Muon —— MomentUm Orthogonalized by Newton-Schulz
------------------------------------------------
来源：Keller Jordan et al. (2024)，在 NanoGPT speedrun 中诞生，
      Liu et al. (arXiv:2502.16982, Moonlight/Moonshot) 在 3B/16B MoE 上验证可扩展性，
      Kimi K2（1T 参数，15.5T tokens）用它做到零 loss spike。

核心直觉
--------
AdamW 把权重矩阵当成**一堆互不相关的标量**各自缩放，完全无视矩阵的二维结构。
后果：更新会被少数大奇异值方向主导（"一两个方向撑起整个更新"），
矩阵的条件数持续恶化，大量容量被浪费在冗余方向上。

Muon 的做法：先算动量，再对动量矩阵做**正交化**（Newton-Schulz），
把所有奇异值拉平到 1：

    M ← β·M + (1−β)·∇L(W)         动量（平滑方向）
    Ũ ← NewtonSchulz(M)            正交化（方向均衡，条件数归一）
    W ← W − η·Ũ                    步进

正交化的效果等价于"在矩阵的所有方向上均匀发力"。
Moonlight 报告相对于 AdamW 约 2× 的计算效率；更严谨的复现研究
（Wen et al., arXiv:2512.05620）指出在 400M–1.5B 区间该比值约 1.4×，
在 1.2B 约 1.1×——**收益随规模放大而增大，小模型上不明显**。
这是选型时必须诚实说明的一点。

适用范围（重要，别用错）
------------------------
Muon 只对**二维隐藏层权重矩阵**有意义（矩阵才有"方向"可正交化）。
embedding、LayerNorm/RMSNorm 缩放、bias、输出头必须留在 AdamW。
所以真实配方永远是 **Muon + AdamW 混合**，由 aether.optim.hybrid 自动路由。

MuonClip（Kimi K2, arXiv:2507.20534）
-------------------------------------
大规模 Muon 训练的一个真实风险：注意力 logit 在训练中后期**无界增长**
（q/k 的范数越训越大），最终炸掉 softmax 的熵，表现为 loss spike。
MuonClip 的解法极其简洁——给 logit 加硬上界 τ：

    若 S_max = max|logit| > τ，则对产生 q/k 的投影各乘 √γ，γ = τ / S_max
    （两侧各乘 √γ ⇒ logit 恰好乘 γ ⇒ 新的 S_max = τ）

它不改梯度、不改目标函数，只是在参数空间做一次**保方向的缩放**，
因此不会像 aux loss 那样污染优化方向。Kimi K2 用它跑完 15.5T tokens 零 spike。
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

import torch

from aether.optim.newton_schulz import NS_COEFFS, zeropower

__all__ = ["Muon", "MuonClipController"]


class Muon(torch.optim.Optimizer):
    """Muon：动量的 Newton-Schulz 正交化更新。

    Args:
        params: 待优化参数（应为二维矩阵）。
        lr: 学习率。
        momentum: 动量系数 β（默认 0.95，Muon 惯用值高于 AdamW）。
        nesterov: 是否用 Nesterov 式动量（实际 Muon 实现普遍开启）。
        ns_steps: Newton-Schulz 迭代步数（5 为调优默认值）。
        backend: "gram"（默认，2026 最新，更快）或 "original"（参照实现）。
        gram_reset_every: Gram 矩阵重置间隔（中精度下建议 2~3）。
        weight_decay: 解耦权重衰减（AdamW 风格，直接衰减参数本身）。
        scale_mode: "spectral"（按 m/n 缩放，Keller 原版）或 "none"。
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        backend: str = "gram",
        gram_reset_every: int = 0,
        weight_decay: float = 0.0,
        eps: float = 1e-7,
        coeffs: tuple = NS_COEFFS,
        scale_mode: str = "spectral",
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"学习率必须非负，收到 {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum 必须在 [0,1)，收到 {momentum}")
        if backend not in ("gram", "original"):
            raise ValueError(f"未知 backend {backend!r}")

        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            backend=backend, gram_reset_every=gram_reset_every,
            weight_decay=weight_decay, eps=eps, coeffs=coeffs, scale_mode=scale_mode,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state: dict) -> None:  # pragma: no cover
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", True)
            group.setdefault("backend", "gram")
            group.setdefault("gram_reset_every", 0)
            group.setdefault("scale_mode", "spectral")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if g.dim() != 2:
                    # 静默跳过非矩阵参数：Muon 对它们无意义，应由 AdamW 接管
                    continue

                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(g)
                    state["momentum_buffer"] = buf
                buf.mul_(mu).add_(g)  # 先存原始动量

                u = buf if not group["nesterov"] else g.add(buf, alpha=mu)

                # ---- 正交化（本优化器的灵魂）----
                u = zeropower(
                    u,
                    steps=group["ns_steps"],
                    eps=group["eps"],
                    backend=group["backend"],
                    gram_reset_every=group["gram_reset_every"],
                    dtype=torch.float32,  # 正交化在 fp32 做：半精度下高次多项式会放大误差
                    coeffs=group["coeffs"],
                )

                if group["scale_mode"] == "spectral":
                    # 对齐 AdamW 的更新 RMS 量级：使 lr 语义与 AdamW 可比
                    u = u * max(1.0, p.size(-2) / p.size(-1)) ** 0.5

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)  # 解耦衰减：不经过梯度/动量，AdamW 同款
                p.add_(u.to(p.dtype), alpha=-lr)

        return loss


class MuonClipController:
    """QK-clip 控制器（Kimi K2 的 MuonClip 机制）。

    每隔 `every` 步测量各注意力层的 max|logit|，超阈值 τ 就对 q/k 投影做保方向缩放。

    为什么测的是"投影矩阵"而不是直接 clamp logit：
    直接 clamp logit 会破坏 softmax 的可微性与信息量，等于给模型喂错误的注意力分布；
    而缩放产生 q/k 的**权重矩阵**是参数空间的等价变换（缩放因子吸收进权重），
    前向数值变了，但函数类的表达能力不变，且不引入梯度噪声。

    Args:
        model: AetherModel。
        threshold: logit 绝对值上界 τ。
        every: 触发间隔步数。
    """

    def __init__(self, model: torch.nn.Module, threshold: float = 64.0, every: int = 50) -> None:
        self.model = model
        self.threshold = float(threshold)
        self.every = int(every)
        self.history: List[dict] = []

    @torch.no_grad()
    def measure(self, x_norm: torch.Tensor) -> List[float]:
        """测量每层注意力的 max|logit|（x_norm 为该层注意力输入）。"""
        out = []
        for layer in self.model.layers:
            xn = layer.attn_norm(x_norm) if hasattr(layer, "attn_norm") else x_norm
            out.append(layer.attn.max_abs_logit(xn))
            # 逐层推进残差代价太高，这里用同一输入近似 —— 记录为近似测量
        return out

    @torch.no_grad()
    def clip(self, per_layer_smax: List[float]) -> dict:
        """按测得的 S_max 执行缩放。返回本次 clip 的审计记录。"""
        rec: dict = {"n_clipped": 0, "max_smax": 0.0, "gammas": []}
        if not per_layer_smax:
            return rec
        rec["max_smax"] = float(max(per_layer_smax))

        for layer, smax in zip(self.model.layers, per_layer_smax):
            if smax <= self.threshold or smax <= 0:
                rec["gammas"].append(1.0)
                continue
            gamma = self.threshold / smax
            side = gamma ** 0.5  # q 侧、k 侧各乘 √γ ⇒ logit 恰好乘 γ
            attn = layer.attn
            # q 侧：w_uq（非位置）+ w_qr（RoPE 位置）
            attn.w_uq.weight.mul_(side)
            attn.w_qr.weight.mul_(side)
            # k 侧：w_uk（从潜在 c^KV 上投影）+ w_kr（解耦 RoPE 键）
            attn.w_uk.weight.mul_(side)
            attn.w_kr.weight.mul_(side)
            rec["n_clipped"] += 1
            rec["gammas"].append(float(gamma))

        self.history.append(rec)
        return rec
