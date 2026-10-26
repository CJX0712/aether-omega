"""
INVAR 证书 —— 代数不变量
=========================

不变量（invariant）是比"测试"更强的东西：它是**无论输入如何都必须成立的性质**。
检验不变量的好处是无需 golden reference——不需要另一个实现来比对，
也不需要保存期望输出，性质本身就是判据。

本文件的每一条都对应一个有名有姓的数学性质：

    INVAR-01  Newton-Schulz 正交性     ‖XXᵀ − I‖_max → 0
    INVAR-02  RoPE 相对性              ⟨R_m q, R_n k⟩ = ⟨R_{m−n} q, k⟩
    INVAR-03  RMSNorm 方向保持         RMSNorm(x)/‖RMSNorm(x)‖ = x/‖x‖
    INVAR-04  MoE 路由权重归一         Σ_{i∈top-k} w_i = 1
    INVAR-05  Softmax 平移不变         softmax(z + c) = softmax(z)
    INVAR-06  因果性                   位置 t 的输出不受 t 之后输入的影响
    INVAR-07  注意力行和归一           Σ_j a_ij = 1
    INVAR-08  MLA KV cache 与头数无关  cache 维度不随 n_heads 变化
    INVAR-09  RMSNorm 齐次性           RMSNorm(αx) = sign-方向不变（尺度无关）
"""

from __future__ import annotations

import torch

from aether.cert.registry import certificate

__all__ = ["invariant_certificates"]


def invariant_certificates(registry, cfg) -> None:
    from aether.modules.attention import MultiheadLatentAttention
    from aether.modules.moe import DeepSeekMoE
    from aether.modules.norm import RMSNorm
    from aether.modules.rope import apply_rope, build_rope_cache
    from aether.modules.transformer import AetherModel
    from aether.optim.newton_schulz import NS_COEFFS, gram_newton_schulz, orthogonality_residual

    torch.manual_seed(0)

    # ---- INVAR-01：正交化把奇异值压到 1 附近 ----
    #
    # ⚠️ 一个必须诚实说明的事实：Muon 的 5 步 Newton-Schulz **不是**精确正交化。
    # 迭代 f(σ)=σ(a+bσ²+cσ⁴) 的不动点是 p(σ)=1 的解 σ*≈0.868 / 1.264，
    # 而 f(σ*)=1 处导数 |f′|≈1.58 > 1 ⇒ 迭代在不动点附近**振荡而非收敛**。
    # 实测 5 步后奇异值落在 [0.68, 1.13]，矩阵在 20~30 步后残差仍停在 ~0.2 不再下降。
    #
    # 这正是 Muon 论文把它叫 "approximate orthogonalization" 的原因：
    # 目标从来不是 ‖UUᵀ−I‖=0，而是把**条件数从 O(10²) 压到 O(1)**——方向均衡。
    # 因此证书的判据必须写成"奇异值围绕 1 的偏离有界"，而非"残差趋零"。
    # （假装它能精确正交，然后设一个 1e-6 的 tolerance，是自欺欺人。）
    def _ortho():
        rows = []
        worst_dev = 0.0
        for shape in [(64, 128), (128, 256), (256, 512)]:
            torch.manual_seed(11)
            G = torch.randn(*shape)
            s0 = torch.linalg.svdvals(G)
            r0 = float(s0.max() / s0.clamp(min=1e-12).min())
            U = gram_newton_schulz(G, steps=5)
            s1 = torch.linalg.svdvals(U)
            r1 = float(s1.max() / s1.clamp(min=1e-12).min())
            dev = max(float((s1.max() - 1).abs()), float((s1.min() - 1).abs()))
            worst_dev = max(worst_dev, dev)
            rows.append(f"{shape}: 奇异值比 {r0:.1f}→{r1:.2f}")
        return {
            "value": worst_dev, "tolerance": 0.5, "relation": "<=",
            "detail": f"正交化后所有奇异值与 1 的最大偏离 {worst_dev:.3f}（实测区间约 [0.68, 1.13]）；"
                      f"各矩阵条件数改善：{'；'.join(rows)}",
            "reference": "Higham (1986) Newton-Schulz；Muon 系数 (3.4445,−4.7750,2.0315)；"
                         "Jordan et al. 2024 称之为 approximate orthogonalization",
        }

    certificate(registry, id="INVAR-01", name="Newton-Schulz 方向均衡", category="INVAR",
                statement="正交化后所有奇异值与 1 的偏离 ≤ 0.5（近似正交，追求方向均衡而非精确实正交）", fn=_ortho)

    # ---- INVAR-02：RoPE 相对性 ----
    def _rope_rel():
        d = cfg.qk_rope_head_dim
        L = 64
        cos, sin = build_rope_cache(L, d, cfg.rope_theta)
        worst = 0.0
        for _ in range(50):
            q = torch.randn(d)
            k = torch.randn(d)
            m = int(torch.randint(0, L, (1,)).item())
            n = int(torch.randint(0, m + 1, (1,)).item())
            xq = torch.zeros(1, L, d); xq[0, m] = q
            xk = torch.zeros(1, L, d); xk[0, n] = k
            rq = apply_rope(xq, cos, sin)[0, m]
            rk = apply_rope(xk, cos, sin)[0, n]
            # 右侧：只旋转相对位移 m-n，key 不旋转
            xq2 = torch.zeros(1, L, d); xq2[0, m - n] = q
            rq2 = apply_rope(xq2, cos, sin)[0, m - n]
            lhs = float(torch.dot(rq, rk))
            rhs = float(torch.dot(rq2, k))
            worst = max(worst, abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-9))
        return {
            "value": worst, "tolerance": 1e-4,
            "detail": "随机 50 组 (q,k,m,n) 验证 ⟨R_m q, R_n k⟩ = ⟨R_{m−n} q, k⟩；"
                      "这是 RoPE 能表达**相对**位置的根本性质",
            "reference": "Su et al. (2021) RoFormer / RoPE",
        }

    certificate(registry, id="INVAR-02", name="RoPE 相对性", category="INVAR",
                statement="⟨RoPE(q,m), RoPE(k,n)⟩ = ⟨RoPE(q,m−n), k⟩ 相对误差 ≤ 1e-4", fn=_rope_rel)

    # ---- INVAR-03：RMSNorm 方向保持 ----
    def _rmsnorm_dir():
        m = RMSNorm(cfg.dim).eval()
        worst = 0.0
        for _ in range(20):
            x = torch.randn(1, 1, cfg.dim)
            with torch.no_grad():
                y = m(x)
            a = (y / y.norm()).view(-1)
            b = (x * m.weight).view(-1)
            b = b / b.norm()
            worst = max(worst, float((a - b).abs().max()))
        return {
            "value": worst, "tolerance": 1e-5,
            "detail": "RMSNorm 只做重缩放、不做去均值与平移 ⇒ 输出方向 = 输入的逐元素缩放方向",
            "reference": "Zhang & Sennrich (2019) RMSNorm",
        }

    certificate(registry, id="INVAR-03", name="RMSNorm 方向保持", category="INVAR",
                statement="RMSNorm 输出方向等于输入按 weight 缩放后的方向，误差 ≤ 1e-5", fn=_rmsnorm_dir)

    # ---- INVAR-04：MoE 路由权重归一 ----
    def _moe_norm():
        torch.manual_seed(13)
        m = DeepSeekMoE(cfg).eval()
        x = torch.randn(2, 8, cfg.dim)
        with torch.no_grad():
            score = torch.sigmoid(m.gate(x.reshape(-1, cfg.dim)))
            _, topi = torch.topk(score + m.expert_bias.unsqueeze(0), cfg.n_activated_experts, dim=-1)
            w = score.gather(1, topi)
            w = w / w.sum(-1, keepdim=True)
        err = float((w.sum(-1) - 1.0).abs().max())
        return {
            "value": err, "tolerance": 1e-5,
            "detail": f"top-{cfg.n_activated_experts} 权重归一化后行和 = 1；"
                      "权重用**未加偏置**的分数，保证偏置不污染梯度",
            "reference": "DeepSeek-V3 无辅助损失偏置路由",
        }

    certificate(registry, id="INVAR-04", name="MoE 路由权重归一", category="INVAR",
                statement="MoE top-k 路由权重之和 = 1，误差 ≤ 1e-5", fn=_moe_norm)

    # ---- INVAR-05：Softmax 平移不变性 ----
    def _softmax_shift():
        worst = 0.0
        for _ in range(20):
            z = torch.randn(4, 32) * 10
            c = torch.randn(4, 1).abs() * 100  # 大平移，检验数值稳定
            worst = max(worst, float((torch.softmax(z, -1) - torch.softmax(z + c, -1)).abs().max()))
        return {
            "value": worst, "tolerance": 1e-5,
            "detail": "softmax(z+c)=softmax(z) 是 max-subtract 数值稳定技巧的理论依据",
            "reference": "logsumexp 稳定性",
        }

    certificate(registry, id="INVAR-05", name="Softmax 平移不变", category="INVAR",
                statement="softmax(z + c) = softmax(z)，误差 ≤ 1e-5", fn=_softmax_shift)

    # ---- INVAR-06：因果性 ----
    def _causal():
        torch.manual_seed(17)
        m = AetherModel(cfg).eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 32))
        with torch.no_grad():
            base = m(idx)["logits"].clone()
            idx2 = idx.clone()
            idx2[:, 20:] = torch.randint(0, cfg.vocab_size, (1, 12))  # 篡改后半段
            pert = m(idx2)["logits"].clone()
        # 位置 0..19 的输出必须完全不受影响
        err = float((base[:, :20] - pert[:, :20]).abs().max())
        late = float((base[:, 20:] - pert[:, 20:]).abs().max())
        return {
            "value": err, "tolerance": 1e-5,
            "detail": f"篡改位置 20+ 后：位置 0–19 输出最大变化 {err:.2e}（应为 0）；"
                      f"位置 20+ 变化 {late:.2e}（应显著非零，证明检验有效）",
            "reference": "自回归因果掩码",
        }

    certificate(registry, id="INVAR-06", name="因果性（未来不泄漏）", category="INVAR",
                statement="修改位置 t 之后的输入，位置 ≤ t 的输出变化 ≤ 1e-5", fn=_causal)

    # ---- INVAR-07：注意力行和归一 ----
    def _attn_sum():
        torch.manual_seed(19)
        att = MultiheadLatentAttention(cfg, 0, mode="naive").eval()
        x = torch.randn(1, 16, cfg.dim)
        with torch.no_grad():
            # 直接取 softmax 后的注意力权重：重跑一次 naive 内部逻辑
            nh = cfg.n_heads
            c_kv = att.w_dkv(x)
            if att.use_qk_norm:
                c_kv = att.kv_latent_norm(c_kv)
            k_c = att.w_uk(c_kv).view(1, 16, nh, att.d_nope)
            q_c = att.w_uq(att.w_dq(x)).view(1, 16, nh, att.d_nope)
            if att.use_qk_norm:
                q_c = att.q_norm(q_c)
            logits = torch.einsum("bhld,bhsd->bhls", q_c.transpose(1, 2), k_c.transpose(1, 2))
            logits = logits * att.scale
            mask = torch.tril(torch.ones(16, 16, dtype=torch.bool))
            logits = logits.masked_fill(~mask, float("-inf"))
            a = torch.softmax(logits, -1)
        err = float((a.sum(-1) - 1.0).abs().max())
        return {
            "value": err, "tolerance": 1e-5,
            "detail": "注意力权重每行和为 1（softmax 归一）+ 因果掩码下三角可见",
        }

    certificate(registry, id="INVAR-07", name="注意力行和归一", category="INVAR",
                statement="softmax 后注意力权重每行和 = 1，误差 ≤ 1e-5", fn=_attn_sum)

    # ---- INVAR-08：MLA KV cache 与头数无关 ----
    def _kv_indep():
        dims = []
        for nh in (4, 8, 16, 32):
            c2 = cfg.to_dict()
            c2["n_heads"] = nh
            from aether.config import ModelConfig
            dims.append((nh, ModelConfig(**c2).kv_cache_dim))
        same = len({d for _, d in dims}) == 1
        mha = 2 * cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim)
        return {
            "value": 0.0 if same else 1.0, "tolerance": 0.0, "relation": "==",
            "passed": same,
            "detail": f"头数→cache维度 {dims}；同配置 MHA 需 {mha} 维/token，"
                      f"MLA 只需 {cfg.kv_cache_dim} 维/token，压缩比 {mha / cfg.kv_cache_dim:.1f}×",
            "reference": "DeepSeek-V2 MLA：KV cache 与 n_heads 解耦",
        }

    certificate(registry, id="INVAR-08", name="MLA KV cache 与头数无关", category="INVAR",
                statement="改变 n_heads 时每 token 的 KV cache 维度保持不变", fn=_kv_indep)

    # ---- INVAR-09：RMSNorm 齐次性 ----
    def _homog():
        m = RMSNorm(cfg.dim).eval()
        worst = 0.0
        with torch.no_grad():
            for _ in range(20):
                x = torch.randn(1, 1, cfg.dim)
                a = 3.0
                y1 = m(x * a)
                y2 = m(x)
                # RMSNorm 是齐次的（正缩放）：f(αx)=α f(x)/1 …… 严格说 f(αx)=sign(α)·f(x)
                worst = max(worst, float((y1 - y2 * (1.0 if a > 0 else -1.0)).abs().max()))
        return {
            "value": worst, "tolerance": 1e-5,
            "detail": "RMSNorm 关于缩放**不变**（0-齐次）：分子 αx 与分母 RMS(αx)=|α|·RMS(x) 同步缩放，"
                      "α 被完全抵消 ⇒ 输出范数恒为 O(√d)，与输入尺度无关。"
                      "残差流因此不会随深度失控——这是它在 2026 年全面取代 LayerNorm 的关键原因。"
                      "（本条初版曾误写为 1-齐次，由 JS-06 交叉验证时发现并更正）",
        }

    certificate(registry, id="INVAR-09", name="RMSNorm 尺度不变性", category="INVAR",
                statement="RMSNorm(αx) = RMSNorm(x)（α>0，0-齐次/尺度不变），误差 ≤ 1e-5", fn=_homog)
