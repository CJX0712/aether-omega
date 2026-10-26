#!/usr/bin/env python
"""性能基准 —— 把"更快"变成可测量的数字，而不是 README 里的一句宣称。

    python scripts/benchmark.py                  # 全部基准
    python scripts/benchmark.py --scan-threads   # CPU 线程数校准
    python scripts/benchmark.py --ns-only        # 只测 Newton-Schulz

重点测量三件别人只写在论文里的事：
    1. Gram Newton-Schulz 相对原版 NS 的**真实**提速（论文称 40–50%）
    2. MLA 相对 MHA/GQA 的 KV cache 压缩比（可解析计算 + 实测内存）
    3. 不同 CPU 线程数下的吞吐（线程不是越多越好）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from aether.config import get_preset  # noqa: E402
from aether.optim.newton_schulz import (  # noqa: E402
    gram_newton_schulz,
    newton_schulz_original,
)


def bench(fn, n: int = 20, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000.0  # ms


def bench_ns(verbose: bool = True) -> list:
    """Newton-Schulz：Gram 版 vs 原版。

    重复次数按矩阵规模自适应：单次 NS 的成本是 O(m²n)，
    对 4096×11008 这种真实大模型尺寸，一次就是 3.7e11 FLOPs，
    重复 20 次要在 CPU 上跑半小时——基准本身必须跑得完才有意义。
    """
    rows = []
    shapes = [(256, 512), (512, 1024), (768, 3072), (1024, 4096)]
    print("\n【Newton-Schulz 正交化】Gram 版 vs 原版")
    print(f"{'矩阵形状':>16s} {'原版(ms)':>10s} {'Gram(ms)':>10s} {'提速':>9s} {'最大偏差':>12s}")
    print("-" * 62)
    for shape in shapes:
        torch.manual_seed(0)
        G = torch.randn(*shape)
        m, n = shape
        reps = 20 if m * n < 6e5 else (8 if m * n < 3e6 else 3)
        t_o = bench(lambda: newton_schulz_original(G, steps=5), n=reps)
        t_g = bench(lambda: gram_newton_schulz(G, steps=5), n=reps)
        o = newton_schulz_original(G, steps=5)
        g = gram_newton_schulz(G, steps=5)
        dev = float((o - g).abs().max().item())
        speedup = (t_o - t_g) / t_o * 100
        rows.append({"shape": list(shape), "original_ms": round(t_o, 3), "gram_ms": round(t_g, 3),
                     "speedup_pct": round(speedup, 1), "max_abs_dev": dev})
        print(f"{str(shape):>16s} {t_o:10.3f} {t_g:10.3f} {speedup:8.1f}% {dev:12.2e}")
    return rows


def bench_kv(verbose: bool = True) -> list:
    """MLA vs MHA/GQA 的 KV cache 占用（解析计算，与实现无关）。"""
    print("\n【KV cache 每 token 占用】MLA vs GQA vs MHA（解析计算）")
    rows = []
    for preset in ("nano", "micro", "base"):
        cfg = get_preset(preset)
        mla = cfg.kv_cache_dim
        mha = 2 * cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim)
        gqa = 2 * max(1, cfg.n_heads // 4) * (cfg.qk_nope_head_dim + cfg.v_head_dim)
        rows.append({
            "preset": preset, "n_heads": cfg.n_heads,
            "mla_per_token": mla, "gqa_per_token": gqa, "mha_per_token": mha,
            "mla_vs_mha": round(mha / mla, 2), "mla_vs_gqa": round(gqa / mla, 2),
        })
        print(f"  {preset:6s} heads={cfg.n_heads:3d}  MLA {mla:5d}  GQA {gqa:5d}  MHA {mha:5d}"
              f"   压缩比 MHA {mha / mla:.1f}× / GQA {gqa / mla:.1f}×")
    return rows


def scan_threads(steps: int = 3) -> list:
    """CPU 线程数扫描：找到这台机器上的最优值。"""
    from aether.modules.transformer import AetherModel
    cfg = get_preset("nano")
    torch.manual_seed(0)
    m = AetherModel(cfg).eval()
    idx = torch.randint(0, 256, (4, 256))
    rows = []
    print("\n【CPU 线程数扫描】（越多≠越快，小矩阵上调度与缓存争用会反噬）")
    best = (None, 1e9)
    for th in (1, 2, 4, 6, 8, 12, 16):
        torch.set_num_threads(th)
        for _ in range(2):
            with torch.no_grad():
                m(idx)
        t = time.perf_counter()
        with torch.no_grad():
            for _ in range(steps):
                m(idx)
        dt = (time.perf_counter() - t) / steps * 1000
        rows.append({"threads": th, "ms": round(dt, 1)})
        flag = ""
        if dt < best[1]:
            best = (th, dt)
            flag = "  ← 当前最优"
        print(f"  threads={th:2d}  {dt:8.1f} ms{flag}")
    print(f"  ⇒ 最优线程数 {best[0]}（{best[1]:.1f} ms）")
    return rows


def bench_infer() -> dict:
    """推理吞吐：prefill + decode。"""
    from aether.modules.transformer import AetherModel
    from aether.determinism import enable_determinism
    enable_determinism(seed=0, threads=6)
    cfg = get_preset("nano")
    torch.manual_seed(0)
    m = AetherModel(cfg).eval()
    idx = torch.randint(0, 256, (1, 128))
    with torch.no_grad():
        t = time.perf_counter()
        m(idx)
        prefill = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        m.generate(idx, max_new_tokens=32, temperature=0.0)
        gen = time.perf_counter() - t
    print("\n【推理吞吐】")
    print(f"  prefill 128 tokens : {prefill:.1f} ms")
    print(f"  decode  32 tokens  : {gen * 1000:.1f} ms  ({gen / 32 * 1000:.1f} ms/token)")
    return {"prefill_ms": round(prefill, 1), "decode_32_ms": round(gen * 1000, 1),
            "ms_per_token": round(gen / 32 * 1000, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-threads", action="store_true")
    ap.add_argument("--ns-only", action="store_true")
    ap.add_argument("--out", default="experiments/benchmark.json")
    args = ap.parse_args()

    print("=" * 72)
    print("AETHER-Ω 性能基准 · torch", torch.__version__, "· device",
          "cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)

    out: dict = {"torch_version": torch.__version__,
                 "device": "cuda" if torch.cuda.is_available() else "cpu"}
    out["newton_schulz"] = bench_ns()
    out["kv_cache"] = bench_kv()
    if not args.ns_only:
        if args.scan_threads:
            out["thread_scan"] = scan_threads()
        out["inference"] = bench_infer()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n报告：{args.out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
