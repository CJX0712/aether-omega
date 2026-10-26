#!/usr/bin/env python
"""推理 —— 加载 checkpoint 并用 MLA KV cache 生成文本。

    python scripts/infer.py --ckpt checkpoints/aether-nano-s1337.pt
    python scripts/infer.py --ckpt ... --prompt "ROMEO:" --tokens 400 --temp 0.8

支持 --verify-cache：额外用"全序列重算"逐 token 校验 KV cache 的正确性
（慢 O(L²)，但能证明增量解码没有静默出错——这是本仓库的一贯作风）。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from aether.config import ModelConfig, get_preset  # noqa: E402
from aether.data import encode  # noqa: E402
from aether.modules.transformer import AetherModel  # noqa: E402


def load_model(ckpt: str, preset: str = "nano") -> AetherModel:
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**sd["cfg"]) if "cfg" in sd else get_preset(preset)
    m = AetherModel(cfg)
    if "model" in sd:
        m.load_state_dict(sd["model"])
    else:
        m.load_state_dict(sd)
    return m.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--preset", default="nano")
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--verify-cache", action="store_true",
                    help="逐 token 用全序列重算校验 KV cache 正确性（慢，但能证伪）")
    args = ap.parse_args()

    torch.set_num_threads(6)
    m = load_model(args.ckpt, args.preset)
    cfg = m.cfg
    n = m.num_parameters()
    print(f"模型加载完成：{n:,} 参数 · {cfg.n_layers} 层 · KV cache {cfg.kv_cache_dim} 维/token")

    ids = torch.tensor([encode(args.prompt)], dtype=torch.long)
    t = time.perf_counter()
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=args.tokens, temperature=args.temp,
                         top_k=args.top_k, seed=args.seed)
    dt = time.perf_counter() - t
    text = bytes(out[0].tolist()).decode("utf-8", errors="replace")
    print("-" * 72)
    print(text)
    print("-" * 72)
    print(f"生成 {args.tokens} tokens 用时 {dt:.2f}s（{dt / args.tokens * 1000:.1f} ms/token）")

    if args.verify_cache:
        print("\n校验：增量解码(KV cache) 是否 ≡ 全序列重算 …")
        with torch.no_grad():
            cur = ids.clone()
            bad = 0
            for i in range(min(args.tokens, 40)):
                full = m(cur)["logits"][:, -1, :].argmax(-1)
                inc = m.generate(cur, max_new_tokens=1, temperature=0.0)[:, -1]
                if int(full) != int(inc):
                    bad += 1
                cur = torch.cat([cur, full.view(1, 1)], dim=1)
        print(f"  比对 {min(args.tokens, 40)} 个 token，不一致 {bad} 个"
              f"  → {'✅ KV cache 正确' if bad == 0 else '❌ KV cache 有误'}")


if __name__ == "__main__":
    main()
