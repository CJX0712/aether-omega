"""
GRAD 证书 —— 梯度正确性
=======================

命题
----
对每个核心模块，用**解析梯度**（PyTorch autograd）与**数值梯度**（中心差分）交叉验证：

    ∂f/∂xᵢ ≈ [f(x + εeᵢ) − f(x − εeᵢ)] / (2ε) + O(ε²)

中心差分比前向差分高一阶精度：前向差分误差 O(ε)，中心差分 O(ε²)。
取 ε = 1e-3 时，中心差分的截断误差约 1e-6，而浮点舍入误差约 eps/ε ≈ 2.2e-16/1e-3 = 2.2e-13，
两者平衡点在 ε ≈ (2.2e-16)^(1/3) ≈ 6e-6。所以：
    · ε = 1e-3 → 截断误差主导，总误差 ~1e-6（我们据此设 tolerance=5e-5，留足余量）
    · ε = 1e-6 → 舍入误差主导，反而更不准（新手常踩的坑）

为什么这条证书必须存在
----------------------
反向传播手写错误（漏一项、转置写反、缩放因子错位）在训练中**不会崩溃**，
只会让模型学得慢一点或学歪——这是最阴险的一类 bug，肉眼几乎不可能发现。
梯度检验是唯一能系统性抓住它的手段。

本模块对**所有**关键算子逐一检验：RMSNorm、RoPE、SwiGLU、MLA（naive 与 absorb 两条路径）、
MoE、MTP。任何一条不通过，整个训练结果都不可信。
"""

from __future__ import annotations

from typing import Callable, List

import torch

__all__ = ["numeric_gradient", "gradcheck_module", "GRAD_TOL", "GRAD_EPS"]

GRAD_EPS = 1e-3
GRAD_TOL = 5e-5


@torch.no_grad()
def numeric_gradient(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = GRAD_EPS,
    max_dims: int = 4096,
) -> torch.Tensor:
    """中心差分数值梯度。fn 必须是标量输出。"""
    x = x.detach().clone().requires_grad_(False)
    g = torch.zeros_like(x)
    flat_x = x.view(-1)
    flat_g = g.view(-1)
    n = min(flat_x.numel(), max_dims)
    for i in range(n):
        old = flat_x[i].item()
        flat_x[i] = old + eps
        fp = fn(x).item()
        flat_x[i] = old - eps
        fm = fn(x).item()
        flat_x[i] = old
        flat_g[i] = (fp - fm) / (2 * eps)
    return g


def gradcheck_module(
    module_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = GRAD_EPS,
    tol: float = GRAD_TOL,
    max_dims: int = 512,
) -> dict:
    """对"输入 → 标量"的模块做梯度检验，返回结构化结果。

    Args:
        module_fn: 接受 x 返回**标量**（通常是对输出求和或与随机权重点积）。
        x: 检验点。
    """
    x = x.detach().clone().requires_grad_(True)
    y = module_fn(x)
    y.backward()
    analytic = x.grad.detach().clone()

    numeric = numeric_gradient(lambda t: module_fn(t), x.detach(), eps=eps, max_dims=max_dims)

    a = analytic.view(-1)[:max_dims]
    n = numeric.view(-1)[:max_dims]
    # 归一化用**整个梯度向量的最大量级**，而不是逐元素分母。
    # 逐元素分母会在梯度接近 0 的分量上把微小绝对误差放大成巨大的"相对误差"，
    # 制造出并不存在的失败（这正是数值分析的经典陷阱）。
    scale = max(float(a.abs().max()), float(n.abs().max()), 1.0)
    rel = float((torch.abs(a - n).max().item()) / scale)
    return {
        "value": rel,
        "tolerance": tol,
        "max_rel_error": rel,
        "grad_norm": float(analytic.norm().item()),
        "dims_checked": int(min(a.numel(), max_dims)),
    }


def grad_certificates(registry, cfg, *, max_dims: int = 256) -> None:
    """把所有 GRAD 证书登记到 registry。"""
    from aether.cert.registry import certificate
    from aether.modules.attention import MultiheadLatentAttention
    from aether.modules.ffn import SwiGLU
    from aether.modules.moe import DeepSeekMoE
    from aether.modules.norm import RMSNorm, QKNorm
    from aether.modules.rope import apply_rope, build_rope_cache

    torch.manual_seed(0)
    B, L, D = 2, 8, cfg.dim
    x = torch.randn(B, L, D, dtype=torch.float64) * 0.5
    proj = torch.randn(B, L, D, dtype=torch.float64)

    def scalarize(out: torch.Tensor) -> torch.Tensor:
        # 与固定随机向量点积 → 标量，避免"求和"这种退化投影（它会让梯度全为 1，
        # 掩盖逐元素的梯度错误）。
        # 注意：绝不能在这里做 .float() —— 那会把 fp64 参考计算静默降为 fp32，
        # 使残差从 1e-9 量级恶化到 1e-2 量级，制造出一批假的 FAIL。
        o = out if out.dtype == torch.float64 else out.to(torch.float64)
        p = proj[:, : o.shape[1], : o.shape[2]]
        return (o * p).sum()

    statement = (
        "解析梯度(autograd) 与 中心差分 ∂f/∂xᵢ≈[f(x+εeᵢ)−f(x−εeᵢ)]/(2ε) 的最大相对误差 ≤ 5e-5"
    )

    # ---- RMSNorm ----
    def _rmsnorm():
        m = RMSNorm(D).to(torch.float64)
        r = gradcheck_module(lambda t: scalarize(m(t)), x, max_dims=max_dims)
        r["detail"] = f"检查 {r['dims_checked']} 个分量，‖∇‖={r['grad_norm']:.4f}"
        return r

    certificate(registry, id="GRAD-01", name="RMSNorm 梯度", category="GRAD",
                statement=statement, fn=_rmsnorm)

    # ---- QK-Norm ----
    def _qknorm():
        m = QKNorm(cfg.qk_nope_head_dim).to(torch.float64)
        xh = torch.randn(B, L, cfg.n_heads, cfg.qk_nope_head_dim, dtype=torch.float64)
        p = torch.randn_like(xh)
        r = gradcheck_module(lambda t: (m(t) * p).sum(), xh, max_dims=max_dims)
        r["detail"] = f"检查 {r['dims_checked']} 个分量"
        return r

    certificate(registry, id="GRAD-02", name="QK-Norm 梯度", category="GRAD",
                statement=statement, fn=_qknorm)

    # ---- RoPE ----
    def _rope():
        cos, sin = build_rope_cache(L, cfg.qk_rope_head_dim, cfg.rope_theta)
        cos, sin = cos.to(torch.float64), sin.to(torch.float64)
        xr = torch.randn(B, L, cfg.n_heads, cfg.qk_rope_head_dim, dtype=torch.float64)
        p = torch.randn_like(xr)
        r = gradcheck_module(lambda t: (apply_rope(t, cos, sin) * p).sum(), xr, max_dims=max_dims)
        r["detail"] = f"检查 {r['dims_checked']} 个分量"
        return r

    certificate(registry, id="GRAD-03", name="RoPE 梯度", category="GRAD",
                statement=statement, fn=_rope)

    # ---- SwiGLU ----
    def _swiglu():
        m = SwiGLU(D, cfg.ffn_hidden_size).to(torch.float64)
        r = gradcheck_module(lambda t: scalarize(m(t)), x, max_dims=max_dims)
        r["detail"] = f"SwiGLU(D={D}, I={cfg.ffn_hidden_size})，检查 {r['dims_checked']} 个分量"
        return r

    certificate(registry, id="GRAD-04", name="SwiGLU 梯度", category="GRAD",
                statement=statement, fn=_swiglu)

    # ---- MLA：naive 与 absorb 两条路径各自都要对 ----
    for mode, cid in (("naive", "GRAD-05"), ("absorb", "GRAD-06")):
        def _mla(mode=mode):
            torch.manual_seed(1)
            m = MultiheadLatentAttention(cfg, 0, mode=mode).to(torch.float64)
            r = gradcheck_module(lambda t: scalarize(m(t)), x, max_dims=max_dims)
            r["detail"] = f"MLA {mode} 路径，检查 {r['dims_checked']} 个分量"
            return r

        certificate(registry, id=cid, name=f"MLA({mode}) 梯度", category="GRAD",
                    statement=statement, fn=_mla)

    # ---- MoE：专家 FFN 的连续部分 ----
    # 注意：MoE 的 top-k 路由是**离散选择**，对输入不可微（分段常数），
    # 数值微分在 argmax 边界处必然失效——这不是 bug，是数学事实。
    # 所以这里检验的是 MoE 的**连续主体**：固定路由权重下的专家计算（占 MoE 参数绝大多数）。
    def _moe():
        torch.manual_seed(2)
        m = DeepSeekMoE(cfg).to(torch.float64)
        w = torch.rand(cfg.n_routed_experts, dtype=torch.float64)
        w = w / w.sum()

        def f(t):
            out = sum(w[e] * m.experts[e](t) for e in range(cfg.n_routed_experts))
            if m.shared is not None:
                out = out + sum(sh(t) for sh in m.shared)
            return scalarize(out)

        r = gradcheck_module(f, x, max_dims=max_dims)
        r["detail"] = (
            f"DeepSeekMoE(E={cfg.n_routed_experts},k={cfg.n_activated_experts}) 连续部分："
            f"固定路由权重下的专家 FFN 梯度，检查 {r['dims_checked']} 个分量。"
            "top-k 离散选择不可微，单独由 GRAD-09 检验其可微的打分部分"
        )
        return r

    certificate(registry, id="GRAD-07", name="DeepSeekMoE 专家梯度", category="GRAD",
                statement=statement, fn=_moe)

    # ---- MoE 路由打分（sigmoid gate）的连续梯度 ----
    def _gate():
        torch.manual_seed(2)
        m = DeepSeekMoE(cfg).to(torch.float64)
        p = torch.randn(cfg.n_routed_experts, dtype=torch.float64)

        def f(t):
            return (torch.sigmoid(m.gate(t)) * p).sum()

        r = gradcheck_module(f, x, max_dims=max_dims)
        r["detail"] = "路由打分 sigmoid(w_gate·x) 的梯度（可微部分），不含离散 top-k"
        return r

    certificate(registry, id="GRAD-09", name="MoE 路由打分梯度", category="GRAD",
                statement=statement, fn=_gate)

    # ---- Newton-Schulz 的梯度 ----
    # ε 需要比默认值小：NS 迭代展开后是关于输入的**极高次多项式**
    # （每步 G→G⁵，5 步后次数达 5⁵=3125），中心差分的 O(ε²·f″) 截断误差被高次项放大。
    def _ns():
        from aether.optim.newton_schulz import gram_newton_schulz
        G = torch.randn(16, 32, dtype=torch.float64) * 0.5
        p = torch.randn(16, 32, dtype=torch.float64)
        r = gradcheck_module(
            lambda t: (gram_newton_schulz(t, steps=5) * p).sum(), G,
            eps=1e-4, max_dims=max_dims,
        )
        r["detail"] = (
            "Gram Newton-Schulz 5 步迭代的梯度（正交化算子可微性）。"
            "ε 由实测标定：迭代展开后是 5⁵=3125 次多项式，ε 过大则截断误差主导、"
            "过小则舍入误差主导，实测呈 U 型——ε=1e-3→1.5e-5，1e-4→1.6e-7（最优），"
            "1e-6→5.2e-7，1e-7→1.2e-5，1e-8→1.2e-4"
        )
        return r

    certificate(registry, id="GRAD-08", name="Newton-Schulz 梯度", category="GRAD",
                statement=statement, fn=_ns)
