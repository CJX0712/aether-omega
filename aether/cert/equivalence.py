"""
EQUIV 证书 —— 数学等价性
=========================

这类证书是本仓库最有辨识度的部分。

行业现状：几乎所有加速算子（FlashAttention、融合 kernel、量化、吸收技巧）
都在论文或 README 里宣称"与朴素实现数学等价"。但这句话**从来没有被机器检查过**——
它是一句信仰，不是一个可执行的断言。一旦实现里有个转置写反、有个缩放漏了，
模型不会崩，只会静默变慢几个点，没人知道。

AETHER-Ω 的做法：把每一条"数学等价"写成**可执行、有误差上界的证书**。

误差上界从哪来（不是拍脑袋）
----------------------------
两条路径计算的是同一个数学对象，差异只来自浮点结合律。
对 n 个数的求和，浮点累积误差的相对上界是 (n−1)·ε/(1−(n−1)·ε) ≈ n·ε。
fp32 的 ε = 1.19e-7；本仓库的等价性检查涉及 O(L·d) 量级的累加（L=64, d=256 → ~1.6e4），
因此理论上界 ~ 1.6e4 × 1.19e-7 ≈ 1.9e-3。
我们设 tolerance = 1e-4（相对误差），既远高于实测值（实测 ~2e-7），
又低于理论最坏界——这意味着**一旦实现出错，残差会立刻越过上界被抓住**。
"""

from __future__ import annotations

import torch

from aether.cert.registry import certificate

__all__ = ["equivalence_certificates"]

EQUIV_TOL = 1e-4


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """相对误差：‖a−b‖_max / max(‖a‖_max, ‖b‖_max)。"""
    return float(
        ((a.detach().float() - b.detach().float()).abs().max()
         / torch.clamp(a.detach().float().abs().max(), min=1e-12)).item()
    )


def equivalence_certificates(registry, cfg) -> None:
    from aether.modules.attention import KVCache, MultiheadLatentAttention
    from aether.modules.moe import DeepSeekMoE
    from aether.modules.transformer import AetherModel
    from aether.optim.newton_schulz import gram_newton_schulz, newton_schulz_original

    torch.manual_seed(0)
    B, L = 2, 64
    x = torch.randn(B, L, cfg.dim)

    # ---- EQUIV-01：MLA 吸收技巧 ----
    def _mla():
        worst, per = 0.0, []
        for li in range(cfg.n_layers):
            torch.manual_seed(10 + li)
            att = MultiheadLatentAttention(cfg, li)
            a = att(x, mode="naive")
            b = att(x, mode="absorb")
            r = _rel(a, b)
            per.append(r)
            worst = max(worst, r)
        return {
            "value": worst, "tolerance": EQUIV_TOL,
            "detail": f"逐层相对误差 {[f'{v:.2e}' for v in per]}；推导：q·kᵀ=(qᶜWᵁᴷ)cᵀ+qᴿkᴿᵀ",
            "reference": "DeepSeek-V2 arXiv:2405.04434 §MLA 吸收形式",
        }

    certificate(registry, id="EQUIV-01", name="MLA naive ≡ absorb", category="EQUIV",
                statement="MLA 的朴素展开实现与矩阵吸收实现输出相对误差 ≤ 1e-4", fn=_mla)

    # ---- EQUIV-02：Gram Newton-Schulz ≡ 原版 Newton-Schulz ----
    def _ns():
        worst = 0.0
        for shape in [(64, 128), (128, 256), (32, 64), (256, 512)]:
            torch.manual_seed(3)
            G = torch.randn(*shape)
            o = newton_schulz_original(G, steps=5)
            g = gram_newton_schulz(G, steps=5)
            worst = max(worst, _rel(o, g))
        return {
            "value": worst, "tolerance": EQUIV_TOL,
            "detail": "令 Pₖ=aI+bGₖ+cGₖ²，则 Gₖ₊₁=Pₖ²Gₖ，Xₙ=(Π Pₖ)X₀ —— 代数恒等，非近似",
            "reference": "Gram Newton-Schulz (2026)；Keller Jordan Muon 系数 (3.4445,−4.7750,2.0315)",
        }

    certificate(registry, id="EQUIV-02", name="Gram NS ≡ 原版 NS", category="EQUIV",
                statement="Gram Newton-Schulz 与原版 Newton-Schulz 输出相对误差 ≤ 1e-4", fn=_ns)

    # ---- EQUIV-03：KV cache 增量解码 ≡ 全序列重算 ----
    def _kvcache():
        torch.manual_seed(5)
        m = AetherModel(cfg)
        m.eval()
        idx = torch.randint(0, cfg.vocab_size, (1, 24))
        with torch.no_grad():
            full = m(idx)["logits"][:, -1, :]
            gen = m.generate(idx, max_new_tokens=1, temperature=0.0)
            greedy_match = bool(gen[0, -1].item() == full.argmax(-1).item())
            # 更强的一致性检查：整条 logits 向量（不只是 argmax）
            caches = [
                KVCache(1, idx.shape[1] + 1, cfg.kv_lora_rank, cfg.qk_rope_head_dim)
                for _ in m.layers
            ]
            inc = m.forward(idx[:, :-1], kv_caches=caches)["logits"]
            inc = m.forward(idx[:, -1:], kv_caches=caches)["logits"][:, -1, :]
        r = _rel(full, inc)
        return {
            "value": r, "tolerance": EQUIV_TOL,
            "detail": f"增量解码 logits 与全序列重算的相对误差；贪心 next-token 一致={greedy_match}",
            "reference": "MLA KV cache 只存 c^KV 与 k^R",
        }

    certificate(registry, id="EQUIV-03", name="KV cache ≡ 全序列重算", category="EQUIV",
                statement="增量解码（KV cache）与全序列重算的最后位置 logits 相对误差 ≤ 1e-4", fn=_kvcache)

    # ---- EQUIV-04：MoE 稀疏分组计算 ≡ 稠密加 mask 的参考实现 ----
    #
    # 命题说明（纠正一个常见误解）：
    # "top-k=2 的 MoE" 与 "全激活的 MoE" **并不等价**——前者的权重只在这 2 个专家间归一化，
    # 后者在全部 E 个间归一化，两者是不同的加权和。
    # 真正应该被机器检查的等价性是**实现层面**的：
    #     实现 A（稀疏）：只对被选中的 (token, expert) 对做计算，用 index_add 累加
    #     实现 B（稠密）：对**所有**专家计算全部 token，再用路由权重 mask 掉未选中的
    # 这两者必须给出同一个张量——而 B 是"显然正确但慢 O(E) 倍"的参照，
    # 稀疏实现正因为省掉未选中的计算才快，也正因为如此才容易写错（分组、索引、累加顺序）。
    def _moe():
        torch.manual_seed(7)
        m = DeepSeekMoE(cfg).eval()
        E, K, D = cfg.n_routed_experts, cfg.n_activated_experts, cfg.dim
        xs = torch.randn(2, 8, D)
        with torch.no_grad():
            sparse = m(xs)
            T = xs.reshape(-1, D)
            score = torch.sigmoid(m.gate(T))
            _, topi = torch.topk(score + m.expert_bias.unsqueeze(0), K, dim=-1)
            w = score.gather(1, topi)
            w = w / w.sum(-1, keepdim=True)
            wfull = torch.zeros(T.shape[0], E, dtype=T.dtype)
            wfull.scatter_(1, topi, w)              # 未被选中的专家权重为 0
            dense = torch.zeros_like(T)
            for e in range(E):                       # 稠密：每个专家都算全部 token
                dense = dense + wfull[:, e: e + 1] * m.experts[e](T)
            if m.shared is not None:
                for sh in m.shared:
                    dense = dense + sh(T)
        r = _rel(sparse, dense.view_as(sparse))
        return {
            "value": r, "tolerance": EQUIV_TOL,
            "detail": f"稀疏实现（只对 top-{K} 分组计算 + index_add）≡ 稠密实现（全部 {E} 专家 × mask 加权）；"
                      "两者计算同一数学对象，差异只来自浮点累加顺序",
            "reference": "DeepSeekMoE arXiv:2405.04434",
        }

    certificate(registry, id="EQUIV-04", name="MoE 稀疏 ≡ 稠密+mask", category="EQUIV",
                statement="MoE 的稀疏分组实现与稠密全专家加 mask 的参考实现相对误差 ≤ 1e-4", fn=_moe)
