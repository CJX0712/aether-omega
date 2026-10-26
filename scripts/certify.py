#!/usr/bin/env python
"""一键全量验真 —— 这是 AETHER-Ω 的核心入口。

    python scripts/certify.py                 # 跑全部证书，控制台输出
    python scripts/certify.py --md            # 输出 Markdown 报告
    python scripts/certify.py --json report.json
    python scripts/certify.py --preset micro  # 对更大档位验真

它回答的问题不是"代码能不能跑"，而是：
    **这个 AI 系统的每一项计算，是否在数学上成立？**
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aether.cert.runner import run_all  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="AETHER-Ω 自证式证书")
    ap.add_argument("--preset", default="nano", choices=["nano", "micro", "base"])
    ap.add_argument("--md", action="store_true", help="输出 Markdown 报告")
    ap.add_argument("--json", default=None, help="把 JSON 报告写到指定路径")
    ap.add_argument("--out-dir", default="experiments")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("AETHER-Ω · 自证式证书（Self-Certifying Certificates）")
    print("=" * 78)
    t0 = time.perf_counter()
    reg = run_all(args.preset, verbose=not args.quiet)
    s = reg.summary()

    print("-" * 78)
    print(f"总计 {s['total']} 条 · 通过 {s['passed']} · 失败 {s['failed']} · "
          f"耗时 {time.perf_counter() - t0:.1f}s")
    for k, v in s["by_category"].items():
        print(f"  {k:6s}  {v['pass']}/{v['n']}")
    print("-" * 78)

    if not args.quiet:
        print(f"{'编号':<10s} {'类别':<6s} {'命题':<46s} {'残差':>11s} {'状态':>5s}")
        for c in reg.certs:
            v = "—" if c.value is None else f"{c.value:.3e}"
            print(f"{c.id:<10s} {c.category:<6s} {c.statement[:44]:<46s} {v:>11s} "
                  f"{'✅' if c.passed else '❌':>5s}")

    os.makedirs(args.out_dir, exist_ok=True)
    md_path = os.path.join(args.out_dir, f"certificates-{args.preset}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(reg.to_markdown())
    print("-" * 78)
    print(f"Markdown 报告 : {md_path}")
    if args.md:
        print(reg.to_markdown())
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(reg.to_json())
        print(f"JSON 报告     : {args.json}")
    print("=" * 78)
    print("✅ 全部证书通过 —— 系统的每一项计算都在数学上成立。"
          if s["all_green"] else "❌ 存在未通过证书，请查看上方明细。")
    sys.exit(0 if s["all_green"] else 1)


if __name__ == "__main__":
    main()
