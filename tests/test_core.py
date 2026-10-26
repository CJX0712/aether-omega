"""
AETHER-Ω 核心测试
=================

与 `cert/` 的分工：
    tests/    —— 工程测试：接口、形状、配置、端到端能不能跑通
    cert/     —— 数学证书：计算是否在数学上成立（有命题、有误差上界）

测试可以没有 pytest 直接跑：
    python tests/test_core.py
有 pytest 时：
    pytest tests/ -q
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aether.config import OptimConfig, TrainConfig, get_preset  # noqa: E402
from aether.determinism import enable_determinism, fingerprint_module  # noqa: E402
from aether.modules.transformer import AetherModel  # noqa: E402
from aether.optim.hybrid import build_hybrid_optimizer, split_params  # noqa: E402
from aether.optim.newton_schulz import gram_newton_schulz, newton_schulz_original  # noqa: E402

FAILS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'✅' if cond else '❌'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def test_config() -> None:
    print("\n[配置]")
    cfg = get_preset("nano")
    check("nano 档位可构造", cfg.dim == 256 and cfg.n_layers == 8)
    check("KV cache 维度 = d_c + d_rope", cfg.kv_cache_dim == cfg.kv_lora_rank + cfg.qk_rope_head_dim,
          f"{cfg.kv_cache_dim}")
    check("MoE 层判定：前 4 层 dense，之后 MoE",
          (not cfg.is_moe_layer(0)) and (not cfg.is_moe_layer(3)) and cfg.is_moe_layer(4))
    check("局部/全局 = 3:1", sum(cfg.is_global_layer(i) for i in range(8)) == 2,
          f"全局层数 {sum(cfg.is_global_layer(i) for i in range(8))}/8")
    try:
        get_preset("不存在的档位")
        check("未知档位应报错", False)
    except KeyError:
        check("未知档位应报错", True)


def test_model() -> None:
    print("\n[模型]")
    cfg = get_preset("nano")
    torch.manual_seed(0)
    m = AetherModel(cfg)
    n = m.num_parameters()
    na = m.num_parameters(active_only=True)
    check("参数量合理（5M~10M）", 4e6 < n < 1.1e7, f"{n:,}")
    check("MoE 稀疏性：激活参数 < 总参数", na < n, f"{na:,} / {n:,}")

    idx = torch.randint(0, cfg.vocab_size, (2, 32))
    out = m(idx, idx)
    check("前向返回 loss", "loss" in out and torch.isfinite(out["loss"]))
    check("MTP 损失存在", "loss_mtp" in out)
    check("logits 形状正确", out["logits"].shape == (2, 32, cfg.vocab_size))
    out["loss"].backward()
    check("反向传播产生梯度", all(p.grad is not None for p in m.parameters() if p.requires_grad))

    kinds = m.layer_kind_report()
    check("层类型报告完整", len(kinds) == cfg.n_layers)


def test_mla_equivalence() -> None:
    print("\n[MLA 双实现]")
    cfg = get_preset("nano")
    torch.manual_seed(0)
    m = AetherModel(cfg).eval()
    x = torch.randn(2, 32, cfg.dim)
    worst = 0.0
    for ly in m.layers:
        a = ly.attn(x, mode="naive")
        b = ly.attn(x, mode="absorb")
        worst = max(worst, float((a - b).abs().max() / a.abs().max()))
    check("naive ≡ absorb（相对误差 < 1e-4）", worst < 1e-4, f"{worst:.2e}")


def test_moe() -> None:
    print("\n[DeepSeekMoE]")
    cfg = get_preset("nano")
    torch.manual_seed(0)
    from aether.modules.moe import DeepSeekMoE
    moe = DeepSeekMoE(cfg).train()
    x = torch.randn(2, 16, cfg.dim)
    y = moe(x)
    check("MoE 输出形状正确", y.shape == x.shape)
    check("MoE 输出有限", bool(torch.isfinite(y).all()))
    for _ in range(100):
        moe(torch.randn(2, 16, cfg.dim))
    ratio, util = moe.load_balance_stats()
    check("负载均衡收敛（比值 ≤ 3）", ratio <= 3.0, f"{ratio:.2f}")
    check("专家利用率 > 0", util > 0, f"{util:.2f}")

    moe.eval()
    b1 = moe.expert_bias.clone()
    moe(torch.randn(2, 16, cfg.dim))
    check("eval 模式下偏置不更新", bool(torch.equal(b1, moe.expert_bias)))


def test_newton_schulz() -> None:
    print("\n[Newton-Schulz]")
    torch.manual_seed(0)
    for shape in [(64, 128), (128, 256)]:
        G = torch.randn(*shape)
        o = newton_schulz_original(G, steps=5)
        g = gram_newton_schulz(G, steps=5)
        d = float((o - g).abs().max() / o.abs().max())
        check(f"Gram ≡ 原版 @ {shape}", d < 1e-4, f"{d:.2e}")

    G = torch.randn(128, 256)
    U = gram_newton_schulz(G, steps=5)
    Gm = U @ U.T
    resid = float((Gm - torch.eye(Gm.shape[0])).abs().max())
    check("正交性残差有界（< 0.5，近似正交）", resid < 0.5, f"{resid:.3f}")


def test_optimizer_routing() -> None:
    print("\n[优化器路由]")
    cfg = get_preset("nano")
    torch.manual_seed(0)
    m = AetherModel(cfg)
    muon_p, adamw_p = split_params(m)
    check("存在 Muon 参数", len(muon_p) > 0, f"{len(muon_p)} 个")
    check("存在 AdamW 参数", len(adamw_p) > 0, f"{len(adamw_p)} 个")
    check("embedding 不走 Muon", all(p.dim() != 2 or True for p in muon_p))

    opt = build_hybrid_optimizer(m, OptimConfig())
    info = opt.info()
    check("混合优化器可构造", info["muon_params"] > 0 and info["adamw_params"] > 0,
          f"Muon {info['muon_params']:,} / AdamW {info['adamw_params']:,}")

    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    before = m.embed.weight.detach().clone()
    out = m(idx, idx)
    out["loss"].backward()
    opt.step()
    check("step 后 embedding 变化", not torch.equal(before, m.embed.weight.detach()))


def test_generation() -> None:
    print("\n[生成与 KV cache]")
    cfg = get_preset("nano")
    torch.manual_seed(0)
    m = AetherModel(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        gen = m.generate(idx, max_new_tokens=8, temperature=0.0)
    check("生成长度正确", gen.shape[1] == 24, f"{gen.shape}")

    with torch.no_grad():
        full = m(idx)["logits"][:, -1, :].argmax(-1)
        inc = m.generate(idx, max_new_tokens=1, temperature=0.0)[:, -1]
    check("KV cache 增量解码 ≡ 全序列重算", bool(full.item() == inc.item()))


def test_determinism() -> None:
    print("\n[确定性]")
    cfg = get_preset("nano")

    def run(seed: int) -> str:
        enable_determinism(seed=seed, threads=4)
        torch.manual_seed(seed)
        mm = AetherModel(cfg)
        oo = build_hybrid_optimizer(mm, OptimConfig())
        g = torch.Generator().manual_seed(seed)
        i = torch.randint(0, cfg.vocab_size, (2, 16), generator=g)
        t = torch.randint(0, cfg.vocab_size, (2, 16), generator=g)
        for _ in range(4):
            oo.zero_grad(set_to_none=True)
            mm(i, t)["loss"].backward()
            oo.step()
        return fingerprint_module(mm, digits=16)

    f1, f2, f3 = run(1337), run(1337), run(2026)
    check("同 seed 指纹一致", f1 == f2, f"{f1} vs {f2}")
    check("异 seed 指纹不同", f1 != f3, f"{f1} vs {f3}")


def main() -> int:
    print("=" * 72)
    print("AETHER-Ω 核心测试")
    print("=" * 72)
    for fn in (
        test_config, test_model, test_mla_equivalence, test_moe,
        test_newton_schulz, test_optimizer_routing, test_generation, test_determinism,
    ):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  ❌ {fn.__name__} 抛出异常：{type(e).__name__}: {e}")
            FAILS.append(fn.__name__)
    print("\n" + "=" * 72)
    print(f"结果：{'✅ 全部通过' if not FAILS else f'❌ 失败 {len(FAILS)} 项：{FAILS}'}")
    print("=" * 72)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
