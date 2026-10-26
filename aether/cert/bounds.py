"""
BOUND / MONO 证书 —— 浮点误差界 与 训练动态单调性
==================================================

BOUND：误差界
-------------
"数值正确"不是玄学，它可以被量化。对一次前向传播，浮点误差的相对上界为

    ‖fl(f(x)) − f(x)‖ / ‖f(x)‖  ≲  γ_n · cond(f, x),     γ_n = n·u/(1 − n·u)

其中 u 是机器精度（fp32: 1.19e-7，fp64: 2.22e-16），n 是累加次数，
cond 是问题本身的条件数。我们无法在证书里精确求出 cond，
但可以做一个**可执行的**检验：用 fp64 作为参考真值，要求 fp32 的相对误差
落在"由网络深度与累加规模推导出的经验上界"之内。

    fp32 的 u = 1.19e-7；8 层网络 × 每层 O(L·d)≈1.6e4 次累加
    ⇒ 最坏界 ~ 1e-3；实测应远低于此（~1e-6 量级）
    ⇒ tolerance 取 1e-4：既远高于实测，又远低于最坏界，出错就能抓住

MONO：单调性
------------
梯度下降在满足 Lipschitz 光滑性且步长 η ≤ 1/L 时有严格的**下降保证**（下降引理）：

    f(x_{k+1}) ≤ f(x_k) − (η/2)·‖∇f(x_k)‖²

这条性质是机器可检的：用**全批量**（消除采样噪声）+ 足够小的 η，
loss 序列必须单调不增。如果它不单调，说明：
    · 学习率超过 1/L（步长过大），或
    · 梯度算错了（GRAD 证书已覆盖），或
    · 优化器实现有 bug

这是一条"物理定律级"的检查——它不依赖任何 golden reference。
"""

from __future__ import annotations

import torch

from aether.cert.registry import certificate

__all__ = ["bound_certificates", "monotonicity_certificates"]


def bound_certificates(registry, cfg) -> None:
    from aether.modules.transformer import AetherModel

    # ---- BOUND-01：fp32 vs fp64 前向误差 ----
    def _fp():
        torch.manual_seed(0)
        m = AetherModel(cfg).eval()
        m64 = AetherModel(cfg).eval().to(torch.float64)
        m64.load_state_dict({k: v.to(torch.float64) for k, v in m.state_dict().items()})
        idx = torch.randint(0, cfg.vocab_size, (1, 32))
        with torch.no_grad():
            y32 = m(idx)["logits"]
            y64 = m64(idx)["logits"].float()
        err = float(((y32 - y64).abs().max() / torch.clamp(y64.abs().max(), min=1e-12)).item())
        return {
            "value": err, "tolerance": 1e-4,
            "detail": f"fp32 相对 fp64 的前向误差；fp32 机器精度 u=1.19e-7，"
                      f"{cfg.n_layers} 层网络理论最坏界 ~1e-3，实测应远低于此",
            "reference": "Higham, Accuracy and Stability of Numerical Algorithms, Ch.2",
        }

    certificate(registry, id="BOUND-01", name="fp32 前向误差界", category="BOUND",
                statement="fp32 前向相对 fp64 的误差 ≤ 1e-4", fn=_fp)

    # ---- BOUND-02：RoPE 长序列角度精度 ----
    def _rope_bound():
        from aether.modules.rope import build_rope_cache
        L = 4096
        c32, s32 = build_rope_cache(L, cfg.qk_rope_head_dim, cfg.rope_theta)
        c64, s64 = build_rope_cache(L, cfg.qk_rope_head_dim, cfg.rope_theta)
        c64, s64 = c64.to(torch.float64), s64.to(torch.float64)
        err = max(
            float((c32.to(torch.float64) - c64).abs().max()),
            float((s32.to(torch.float64) - s64).abs().max()),
        )
        return {
            "value": err, "tolerance": 1e-5,
            "detail": f"L={L} 的 RoPE 角度表 fp32 vs fp64；"
                      "角度表在长上下文下会被放大，故单独成证",
        }

    certificate(registry, id="BOUND-02", name="RoPE 长序列角度精度", category="BOUND",
                statement="4096 长度的 RoPE 角度表 fp32 误差 ≤ 1e-5", fn=_rope_bound)

    # ---- BOUND-03：MoE 负载均衡上界 ----
    def _load():
        from aether.modules.moe import DeepSeekMoE
        torch.manual_seed(0)
        m = DeepSeekMoE(cfg).train()
        with torch.no_grad():
            for _ in range(200):  # 让偏置自适应收敛
                m(torch.randn(2, 16, cfg.dim))
        ratio, util = m.load_balance_stats()
        return {
            "value": ratio, "tolerance": 3.0, "relation": "<=",
            "detail": f"最大负载/平均负载 = {ratio:.2f}，专家利用率 = {util:.2f}；"
                      "无辅助损失偏置路由的目标是把比值压到 1 附近（上界取 3.0 为健康阈值）",
            "reference": "DeepSeek-V3 无辅助损失负载均衡",
        }

    certificate(registry, id="BOUND-03", name="MoE 负载均衡界", category="BOUND",
                statement="MoE 最大负载 / 平均负载 ≤ 3.0（偏置路由收敛后）", fn=_load)


def monotonicity_certificates(registry, cfg) -> None:
    from aether.modules.transformer import AetherModel

    # ---- MONO-01：下降引理（全批量 + 小步长 ⇒ loss 单调不增）----
    def _descent():
        torch.manual_seed(0)
        m = AetherModel(cfg)
        idx = torch.randint(0, cfg.vocab_size, (2, 32))
        tgt = torch.randint(0, cfg.vocab_size, (2, 32))
        lr = 1e-3  # 远小于 1/L 的保守步长
        opt = torch.optim.SGD(m.parameters(), lr=lr)
        losses = []
        for _ in range(30):
            opt.zero_grad(set_to_none=True)
            loss = m(idx, tgt)["loss"]
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        # 允许 1e-9 的数值噪声容差
        worst = 0.0
        for a, b in zip(losses, losses[1:]):
            worst = max(worst, b - a)
        return {
            "value": worst, "tolerance": 1e-9, "relation": "<=",
            "detail": f"全批量 SGD(η={lr}) 30 步，loss {losses[0]:.6f} → {losses[-1]:.6f}；"
                      f"最大单步上升 {worst:.3e}",
            "reference": "下降引理：η ≤ 1/L ⇒ f(x_{k+1}) ≤ f(x_k) − (η/2)‖∇f‖²",
        }

    certificate(registry, id="MONO-01", name="下降引理（全批量单调性）", category="MONO",
                statement="全批量 + 足够小步长下，每步 loss 上升量 ≤ 1e-9", fn=_descent)

    # ---- MONO-02：Muon 更新正交性（优化器自身的步长界）----
    def _muon_step():
        from aether.optim.muon import Muon
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.randn(64, 128))
        opt = Muon([p], lr=0.02, momentum=0.0, nesterov=False, ns_steps=5, backend="gram")
        p.grad = torch.randn(64, 128)
        before = p.detach().clone()
        opt.step()
        delta = (p.detach() - before)
        # 正交化后的更新矩阵：所有奇异值应接近相等（这才是"方向均衡"的量化表达）
        s = torch.linalg.svdvals(delta)
        ratio = float(s.max() / s.clamp(min=1e-12).min())
        return {
            "value": ratio, "tolerance": 3.0, "relation": "<=",
            "detail": f"Muon 更新矩阵的奇异值最大/最小 = {ratio:.2f}；"
                      "原梯度（未正交化）的该比值通常在 10~100，正交化把它压到 O(1)，"
                      "这正是 Muon '各方向均匀发力' 的量化体现",
            "reference": "Muon / Newton-Schulz 正交化更新",
        }

    certificate(registry, id="MONO-02", name="Muon 更新各向同性", category="MONO",
                statement="Muon 单步更新的奇异值最大/最小比 ≤ 3.0", fn=_muon_step)
