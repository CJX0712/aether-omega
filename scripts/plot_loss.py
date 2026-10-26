#!/usr/bin/env python3
"""零依赖：从训练 report.json 生成 loss / perplexity 曲线 SVG。

AETHER-Ω 坚持「零依赖、可复现」原则——本项目不引入 matplotlib 等重型绘图库，
而是用纯标准库构造矢量图，任何人 clone 后无需额外安装即可重新生成图表。

用法:
    python scripts/plot_loss.py [--report PATH] [--out PATH] [--mock]

    --mock  用内置假数据跑通 SVG 生成（CI / 自检用，不读真实 report）
"""
import json
import math
import glob
import os
import sys
import argparse


def find_report(path: str = None) -> str:
    if path:
        return path
    cands = sorted(glob.glob("experiments/train-*-report.json"))
    if not cands:
        cands = sorted(glob.glob("experiments/*report*.json"))
    if not cands:
        raise FileNotFoundError("未找到训练报告 experiments/train-*-report.json")
    return cands[-1]


def _points(history):
    train = [(r["step"], r["train_loss"]) for r in history if r.get("train_loss") is not None]
    val = [(r["step"], r["val_loss"]) for r in history if r.get("val_loss") is not None]
    ppl = [(s, math.exp(min(v, 20))) for s, v in val]
    return train, val, ppl


def build_svg(history, title: str = "AETHER-Ω · 训练曲线 (nano)", width: int = 880, height: int = 480):
    train, val, ppl = _points(history)

    # ---- 画布与边距 ----
    ml, mr, mt, mb = 64, 64, 54, 46          # 左/右/上/下边距
    pw, ph = width - ml - mr, height - mt - mb
    plot_x0, plot_y0 = ml, mt
    plot_x1, plot_y1 = ml + pw, mt + ph

    # ---- 数据范围 ----
    all_steps = [s for s, _ in train] + [s for s, _ in val]
    max_step = max(all_steps) if all_steps else 1
    loss_vals = [v for _, v in train] + [v for _, v in val]
    max_loss = max(loss_vals) if loss_vals else 1.0
    min_loss = min(loss_vals) if loss_vals else 0.0
    # 留 10% 余量
    loss_top = max_loss * 1.08
    loss_bot = max(0.0, min_loss * 0.9)
    max_ppl = max([v for _, v in ppl]) if ppl else 10.0

    def sx(step):
        if max_step == 0:
            return plot_x0
        return plot_x0 + pw * (step / max_step)

    def sy_loss(v):
        if loss_top == loss_bot:
            return plot_y1
        return plot_y1 - ph * ((v - loss_bot) / (loss_top - loss_bot))

    def sy_ppl(v):
        if max_ppl <= 1:
            return plot_y1
        return plot_y1 - ph * ((v - 1.0) / (max_ppl - 1.0))

    # ---- 网格 + 坐标轴 ----
    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                 f'viewBox="0 0 {width} {height}" font-family="Menlo, Consolas, monospace">')
    parts.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    parts.append(f'<text x="{width//2}" y="26" text-anchor="middle" font-size="18" '
                 f'font-weight="bold" fill="#111">AETHER-Ω 训练曲线 · nano 档</text>')

    # y 轴网格（loss，左）
    ny = 5
    for i in range(ny + 1):
        y = plot_y0 + ph * i / ny
        lv = loss_top - (loss_top - loss_bot) * i / ny
        parts.append(f'<line x1="{plot_x0}" y1="{y:.1f}" x2="{plot_x1}" y2="{y:.1f}" '
                     f'stroke="#e3e8ee" stroke-width="1"/>')
        parts.append(f'<text x="{plot_x0-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#444">{lv:.2f}</text>')
    # y 轴网格（ppl，右）
    for i in range(ny + 1):
        y = plot_y0 + ph * i / ny
        pv = max_ppl - (max_ppl - 1.0) * i / ny
        parts.append(f'<text x="{plot_x1+8}" y="{y+4:.1f}" text-anchor="start" font-size="11" fill="#c2410c">{pv:.0f}</text>')

    # x 轴刻度
    nx = 6
    for i in range(nx + 1):
        x = plot_x0 + pw * i / nx
        sv = max_step * i / nx
        parts.append(f'<text x="{x:.1f}" y="{plot_y1+20}" text-anchor="middle" font-size="11" fill="#444">{sv:.0f}</text>')

    # 轴标题
    parts.append(f'<text x="{ml-46}" y="{mt+ph/2}" text-anchor="middle" font-size="12" '
                 f'fill="#444" transform="rotate(-90 {ml-46} {mt+ph/2})">loss</text>')
    parts.append(f'<text x="{plot_x1+46}" y="{mt+ph/2}" text-anchor="middle" font-size="12" '
                 f'fill="#c2410c" transform="rotate(90 {plot_x1+46} {mt+ph/2})">perplexity</text>')
    parts.append(f'<text x="{plot_x0+pw/2}" y="{height-10}" text-anchor="middle" font-size="12" fill="#444">training step</text>')

    def path(pts, ymap):
        if not pts:
            return ""
        d = "M " + " L ".join(f"{sx(s):.1f},{ymap(v):.1f}" for s, v in pts)
        return d

    # ---- 曲线 ----
    # train_loss 仅在数据合理时绘制：旧版报告的 train_loss 是窗口累加和（数值 >> loss 标度，>50），
    # 那是已修复的 bug；此处自动跳过，避免对外展示错误曲线。修正后的 trainer 产出的
    # 平均 loss 通常 < 10，会被正常绘制。
    train_valid = bool(train) and max(v for _, v in train) < 50
    if train_valid:
        parts.append(f'<path d="{path(train, sy_loss)}" fill="none" stroke="#2563eb" stroke-width="2" '
                     f'stroke-linejoin="round" opacity="0.85"/>')
    parts.append(f'<path d="{path(val, sy_loss)}" fill="none" stroke="#dc2626" stroke-width="2.4" '
                 f'stroke-linejoin="round"/>')
    parts.append(f'<path d="{path(ppl, sy_ppl)}" fill="none" stroke="#c2410c" stroke-width="1.6" '
                 f'stroke-dasharray="5 4" stroke-linejoin="round"/>')

    # ---- 数据点 ----
    for s, v in val:
        parts.append(f'<circle cx="{sx(s):.1f}" cy="{sy_loss(v):.1f}" r="3" fill="#dc2626"/>')
    last_s, last_v = val[-1]
    parts.append(f'<text x="{sx(last_s):.1f}" y="{sy_loss(last_v)-8:.1f}" text-anchor="end" '
                 f'font-size="11" fill="#dc2626">val {last_v:.3f}</text>')
    if ppl:
        last_p = ppl[-1]
        parts.append(f'<text x="{sx(last_p[0]):.1f}" y="{sy_ppl(last_p[1])+14:.1f}" text-anchor="end" '
                     f'font-size="11" fill="#c2410c">ppl {last_p[1]:.1f}</text>')

    # ---- 图例 ----
    lx, ly = plot_x0 + 14, plot_y0 + 16
    rows = []
    if train_valid:
        rows.append(("#2563eb", "2", "train loss"))
    rows.append(("#dc2626", "2.4", "val loss"))
    rows.append(("#c2410c", "1.6", "val perplexity"))
    bh = 10 + 18 * len(rows)
    parts.append(f'<rect x="{lx}" y="{ly-10}" width="170" height="{bh}" fill="#ffffff" stroke="#ccc" rx="4"/>')
    yy = ly
    for color, sw, label in rows:
        dash = ' stroke-dasharray="5 4"' if color == "#c2410c" else ""
        parts.append(f'<line x1="{lx+8}" y1="{yy}" x2="{lx+30}" y2="{yy}" stroke="{color}" '
                     f'stroke-width="{sw}"{dash}/>')
        parts.append(f'<text x="{lx+36}" y="{yy+4}" font-size="11" fill="#333">{label}</text>')
        yy += 18

    parts.append('</svg>')
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=None)
    ap.add_argument("--out", default="assets/train-loss-curve.svg")
    ap.add_argument("--mock", action="store_true", help="用内置假数据生成示例图")
    args = ap.parse_args()

    if args.mock:
        history = [
            {"step": 0, "train_loss": 6.2, "val_loss": 6.21},
            {"step": 100, "train_loss": 3.1, "val_loss": 2.92},
            {"step": 200, "train_loss": 2.5, "val_loss": 2.40},
            {"step": 300, "train_loss": 2.1, "val_loss": 2.05},
            {"step": 400, "train_loss": 1.8, "val_loss": 1.80},
            {"step": 500, "train_loss": 1.6, "val_loss": 1.62},
            {"step": 600, "train_loss": 1.45, "val_loss": 1.50},
            {"step": 700, "train_loss": 1.33, "val_loss": 1.41},
            {"step": 800, "train_loss": 1.24, "val_loss": 1.35},
            {"step": 900, "train_loss": 1.17, "val_loss": 1.30},
            {"step": 1000, "train_loss": 1.11, "val_loss": 1.27},
            {"step": 1100, "train_loss": 1.06, "val_loss": 1.25},
            {"step": 1200, "train_loss": 1.02, "val_loss": 1.23},
        ]
    else:
        report = json.load(open(find_report(args.report), encoding="utf-8"))
        history = report["history"]
        title = f"AETHER-Ω · 训练曲线 (nano) · {report.get('elapsed_sec', 0):.0f}s"

    svg = build_svg(history)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"SVG 已写出: {args.out}  ({len(svg)} bytes)")


if __name__ == "__main__":
    main()
