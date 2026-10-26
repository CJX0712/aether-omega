#!/usr/bin/env python
"""训练 AETHER-Ω（nano 档）。

    python scripts/train_nano.py                      # 默认 1500 步
    python scripts/train_nano.py --steps 300          # 快速冒烟
    python scripts/train_nano.py --preset micro       # 更大模型
    python scripts/train_nano.py --no-muon            # 消融：全部走 AdamW

训练结束后会：
    1. 保存 checkpoint 到 checkpoints/
    2. 输出训练报告到 experiments/
    3. 生成一段文本样本（肉眼验证模型真的学会了什么）
    4. 打印权重 SHA-256 指纹（供第三方复现比对）
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from aether.config import OptimConfig, TrainConfig, get_preset  # noqa: E402
from aether.data import TextDataset, load_text  # noqa: E402
from aether.determinism import enable_determinism, fingerprint_module  # noqa: E402
from aether.modules.transformer import AetherModel  # noqa: E402
from aether.trainer import Trainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="训练 AETHER-Ω")
    p.add_argument("--preset", default="nano", choices=["nano", "micro", "base"])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--lr-muon", type=float, default=None)
    p.add_argument("--lr-adamw", type=float, default=None)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--data", default="data/tinyshakespeare.txt")
    p.add_argument("--no-muon", action="store_true", help="消融：所有参数走 AdamW")
    p.add_argument("--no-gram-ns", action="store_true", help="消融：用原版 Newton-Schulz")
    p.add_argument("--eval-every", type=int, default=150)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--sample", action="store_true", default=True)
    p.add_argument("--out", default="experiments")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = enable_determinism(seed=args.seed, threads=args.threads)

    cfg = get_preset(args.preset)
    if args.seq_len:
        cfg.max_seq_len = args.seq_len
        cfg.local_window = max(32, args.seq_len // 4)

    ocfg = OptimConfig()
    if args.lr_muon:
        ocfg.lr_muon = args.lr_muon
    if args.lr_adamw:
        ocfg.lr_adamw = args.lr_adamw
    if args.no_gram_ns:
        ocfg.use_gram_ns = False

    tcfg = TrainConfig(
        batch_size=args.batch_size,
        max_steps=args.steps,
        seed=args.seed,
        threads=args.threads,
        eval_every=args.eval_every,
        log_every=args.log_every,
        grad_accum_steps=args.grad_accum or TrainConfig().grad_accum_steps,
    )

    text = load_text(args.data)
    ds = TextDataset(text, cfg.max_seq_len, seed=args.seed)

    torch.manual_seed(args.seed)
    model = AetherModel(cfg)

    # 消融：--no-muon 时把所有参数交给 AdamW（用于对比 Muon 的实际收益）
    if args.no_muon:
        from aether.optim.muon import Muon

        class _AllAdamW(Trainer):
            pass

        trainer = Trainer(model, cfg, ocfg, tcfg, ds)
        from torch.optim import AdamW

        trainer.opt.adamw = AdamW(
            list(model.parameters()), lr=ocfg.lr_adamw, betas=ocfg.betas,
            eps=ocfg.eps, weight_decay=ocfg.weight_decay,
        )
        trainer.opt.muon = Muon([], lr=ocfg.lr_muon)  # 空 Muon
    else:
        trainer = Trainer(model, cfg, ocfg, tcfg, ds)

    n_param = model.num_parameters()
    n_active = model.num_parameters(active_only=True)
    print("=" * 72)
    print(f"AETHER-Ω 训练 · 档位 {args.preset}")
    print("=" * 72)
    print(f"  总参数        : {n_param:,}（每 token 激活 {n_active:,}）")
    print(f"  层数/维度/头数: {cfg.n_layers} / {cfg.dim} / {cfg.n_heads}")
    print(f"  MLA KV cache  : {cfg.kv_cache_dim} 维/token"
          f"（同规模 MHA 需 {2 * cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim)} 维）")
    print(f"  优化器        : {'Muon(Gram NS)+AdamW' if not args.no_muon else 'AdamW only（消融）'}"
          f"{'' if args.no_gram_ns else ''}")
    print(f"  数据          : {ds.stats()}")
    print(f"  确定性环境    : seed={env['seed']} threads={env['threads']} "
          f"deterministic={env['deterministic_algorithms']}")
    print("-" * 72)

    result = trainer.run()
    result["config"] = {"model": cfg.to_dict(), "optim": ocfg.to_dict(), "train": tcfg.to_dict()}
    result["dataset"] = ds.stats()
    result["env"] = env
    result["ablation"] = {"no_muon": args.no_muon, "no_gram_ns": args.no_gram_ns}

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(tcfg.checkpoint_dir, f"aether-{args.preset}-s{args.seed}.pt")
    trainer.save(ckpt)

    if args.sample:
        model.eval()
        prompt = "ROMEO:"
        ids = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=200, temperature=0.8, top_k=16, seed=args.seed)
        sample = bytes(out[0].tolist()).decode("utf-8", errors="replace")
        result["sample"] = sample
        print("-" * 72)
        print("生成样本（prompt='ROMEO:'）：")
        print(sample)
        model.train()

    print("-" * 72)
    print(f"  耗时          : {result['elapsed_sec']}s  ({result['steps_per_sec']} steps/s)")
    print(f"  最佳 val loss : {result['final_val_loss']:.4f}  ppl {result['final_perplexity']:.2f}")
    print(f"  QK-clip 触发  : {result['n_clip_events']} 次")
    print(f"  权重指纹      : {result['weight_fingerprint']}")
    print(f"  checkpoint    : {ckpt}")

    tag = "nano" if args.preset == "nano" else args.preset
    suffix = "-adamw" if args.no_muon else ""
    rp = os.path.join(args.out, f"train-{tag}{suffix}-report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  训练报告      : {rp}")
    print("=" * 72)


if __name__ == "__main__":
    main()
