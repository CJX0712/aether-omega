# 实验产物

本目录存放**真实跑出**的结果，每一份都可被第三方复现比对。

| 文件 | 内容 | 生成命令 |
|---|---|---|
| `train-log.txt` | 训练原始日志（loss / 梯度范数 / MoE 负载） | `python scripts/train_nano.py --steps 1200` |
| `train-nano-report.json` | 训练报告（含权重 SHA-256 指纹、完整配置、运行时证书） | 同上自动生成 |
| `certificates-nano.md` | 30 条证书的完整报告（命题/残差/上界） | `python scripts/certify.py` |
| `benchmark.json` | 性能基准（Gram NS 提速、KV cache 压缩比、线程扫描） | `python scripts/benchmark.py --scan-threads` |
| `reproducibility.json` | 位级复现验证（三次运行的指纹） | `python scripts/reproduce.py` |

## 怎么比对

训练报告里的 `weight_fingerprint` 是权重的 SHA-256（前 16 位）。
如果你用相同配置（seed / threads / 步数 / torch 版本）跑，
应当得到**完全相同**的指纹。不同则说明环境有差异——
先看 `env` 字段（torch 版本、线程数、device）。

```bash
python - <<'PY'
import json
r = json.load(open("experiments/train-nano-report.json"))
print("指纹 :", r["weight_fingerprint"])
print("val  :", r["final_val_loss"], "ppl", r["final_perplexity"])
print("环境 :", r["env"])
PY
```

> 更严格的验证请用 `python scripts/reproduce.py`——它跑两次独立训练并直接比对，
> 同时验证"不同 seed 必须产生不同结果"（防止确定性开关把随机性也关掉）。
