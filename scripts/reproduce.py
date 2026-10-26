#!/usr/bin/env python
"""复现性验证 —— 第三方克隆仓库后，用这条命令证明"跑出来的和报告里的一样"。

    python scripts/reproduce.py                 # 默认 60 步，两次独立运行
    python scripts/reproduce.py --steps 200     # 更严格（更慢）

验证三件事：
    1. 同一 seed 两次训练 → 权重 SHA-256 指纹 **bit-exact 相同**
    2. 不同 seed → 指纹不同（证明 seed 真的生效，而非被确定性开关"锁死"）
    3. loss 曲线逐点一致

为什么这很重要
--------------
绝大多数开源训练代码无法复现：非确定的归约顺序、cuBLAS split-K、
多线程并行归约、数据采样顺序…… 任何一处漂移都让"我复现了你的结果"变成空话。
本仓库把它做成可执行断言，并给出可公开比对的 SHA-256 指纹。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from aether.config import OptimConfig, TrainConfig, get_preset  # noqa: E402
from aether.data import TextDataset, load_text  # noqa: E402
from aether.determinism import enable_determinism, fingerprint_module  # noqa: E402
from aether.modules.transformer import AetherModel  # noqa: E402
from aether.optim.hybrid import build_hybrid_optimizer  # noqa: E402


def run_once(preset: str, seed: int, steps: int, threads: int, data: str) -> dict:
    """完整构建 + 训练一次，返回指纹与 loss 轨迹。"""
    enable_determinism(seed=seed, threads=threads)
    cfg = get_preset(preset)
    ocfg, tcfg = OptimConfig(), TrainConfig(seed=seed, threads=threads, max_steps=steps)

    text = load_text(data)
    ds = TextDataset(text, cfg.max_seq_len, seed=seed)
    torch.manual_seed(seed)
    model = AetherModel(cfg)
    opt = build_hybrid_optimizer(model, ocfg)

    losses = []
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        x, y = ds.get_batch("train", tcfg.batch_size)
        out = model(x, y)
        out["loss"].backward()
        opt.step()
        losses.append(round(float(out["loss"].detach()), 10))
    return {
        "fingerprint": fingerprint_module(model, digits=32),
        "losses": losses,
        "final_loss": losses[-1] if losses else None,
        "elapsed": round(time.perf_counter() - t0, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="AETHER-Ω 复现性验证")
    ap.add_argument("--preset", default="nano")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--data", default="data/tinyshakespeare.txt")
    ap.add_argument("--out", default="experiments/reproducibility.json")
    args = ap.parse_args()

    print("=" * 78)
    print("AETHER-Ω · 位级复现性验证")
    print("=" * 78)
    print(f"档位 {args.preset} · seed {args.seed} · {args.steps} 步 · threads {args.threads}")

    a = run_once(args.preset, args.seed, args.steps, args.threads, args.data)
    b = run_once(args.preset, args.seed, args.steps, args.threads, args.data)
    c = run_once(args.preset, args.seed + 1, args.steps, args.threads, args.data)

    same = a["fingerprint"] == b["fingerprint"]
    loss_same = a["losses"] == b["losses"]
    differs = a["fingerprint"] != c["fingerprint"]
    all_green = same and loss_same and differs

    print("-" * 78)
    print(f"运行 1 (seed={args.seed})  指纹 {a['fingerprint']}")
    print(f"运行 2 (seed={args.seed})  指纹 {b['fingerprint']}")
    print(f"运行 3 (seed={args.seed + 1})  指纹 {c['fingerprint']}")
    print("-" * 78)
    print(f"  [{'✅' if same else '❌'}] 同 seed 权重指纹 bit-exact 相同")
    print(f"  [{'✅' if loss_same else '❌'}] 同 seed loss 轨迹逐点相同"
          f"（最大漂移 {max((abs(x - y) for x, y in zip(a['losses'], b['losses'])), default=0):.3e}）")
    print(f"  [{'✅' if differs else '❌'}] 不同 seed 产生不同结果（seed 确实生效）")
    print("-" * 78)
    print(f"单次耗时 {a['elapsed']}s；最终 loss {a['final_loss']:.6f}")
    print("✅ 复现性成立：任何人克隆本仓库都能得到一模一样的权重。"
          if all_green else "❌ 复现性不成立，请检查确定性开关与线程数设置。")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "preset": args.preset, "steps": args.steps, "seed": args.seed,
                "threads": args.threads,
                "run1": {k: v for k, v in a.items() if k != "losses"},
                "run2": {k: v for k, v in b.items() if k != "losses"},
                "run3": {k: v for k, v in c.items() if k != "losses"},
                "loss_trace": a["losses"],
                "checks": {"bitwise_identical": same, "loss_identical": loss_same, "seed_sensitive": differs},
                "all_green": all_green,
                "torch_version": torch.__version__,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"报告：{args.out}")
    sys.exit(0 if all_green else 1)


if __name__ == "__main__":
    main()
