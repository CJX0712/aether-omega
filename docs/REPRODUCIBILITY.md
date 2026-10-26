# 复现指南

> 本文的目标：**任何人 clone 这个仓库，都能得到 bit-exact 相同的结果**。
> 如果不能，这是 bug，请提 issue 并附上你的环境信息。

---

## 1. 三条命令自证

```bash
git clone https://github.com/CJX0712/aether-omega.git && cd aether-omega
pip install -r requirements.txt

python scripts/reproduce.py --steps 60      # 位级复现性
python scripts/certify.py                   # 30 条证书
node  verify/aether-verify.js               # 12 条（不需要 Python）
```

`reproduce.py` 验证三件事：

| 检查 | 判据 | 为什么需要 |
|---|---|---|
| 同 seed 两次训练权重 SHA-256 指纹 | **bit-exact 相同** | 核心保证 |
| 同 seed 两次的 loss 轨迹 | **逐点相同** | 排除"权重相同但过程不同"的巧合 |
| 不同 seed | 指纹**必须不同** | 反证：若相同，说明 seed 根本没生效 |

---

## 2. 为什么用 SHA-256 而不是"比较 loss"

loss 相同但权重不同的情况非常常见：

- 对称性导致的等价解（如注意力头的排列对称）
- 浮点微扰后的分岔（两条轨迹在某个临界点分开，loss 曲线却仍很接近）
- 不同的初始化却收敛到相近的 loss

**bit-exact 的权重指纹是唯一不含糊的复现凭据**，且可公开比对——
第三方跑完后可以直接比对 `experiments/reproducibility.json` 里的指纹。

指纹计算方式（`aether/determinism.py`）：

```python
# 参数按名字排序（避免 dict 顺序漂移），逐参数取 SHA-256 后串联
# 同时纳入 buffer，并记录 shape/dtype/layout 到 header
fingerprint_module(model, digits=32)
```

---

## 3. 漂移源与对策（完整清单）

| # | 漂移源 | 表现 | 对策 |
|---|---|---|---|
| 1 | **cuBLAS split-K 归约顺序** | 同机两次结果不同 | `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| 2 | **cuDNN autotune** | 挑选非确定的最快算法 | `cudnn.deterministic=True, benchmark=False` |
| 3 | **原子加（atomicAdd）** | 梯度累加顺序不确定 | `torch.use_deterministic_algorithms(True)` |
| 4 | **OpenMP 并行归约** | 多线程分块顺序漂移 | 固定 `torch.set_num_threads()` |
| 5 | **数据采样顺序** | batch 内容不同 | 独立 RNG 流（与模型初始化解耦） |
| 6 | **Python hash 随机化** | set/dict 顺序漂移 | `PYTHONHASHSEED` |
| 7 | **非确定算子静默回退** | 某个算子偷偷用了非确定实现 | `warn_only=False` ⇒ 直接抛错而非静默 |

第 7 条最关键：`torch.use_deterministic_algorithms(True, warn_only=False)`
会让非确定算子**直接抛错**，而不是静默给你一个不可复现的结果。
生产环境有时需要 `warn_only=True`，但 CI 与实验必须 `False`。

---

## 4. 数据可复现

### 4.1 语料随仓库发布

`data/tinyshakespeare.txt`（1.1MB，莎士比亚公共领域文本）**已入库**。
目的很直接：**网络是复现的头号敌人**。如果训练依赖"下载某个文件"，
那么这个文件变了、链接挂了、或者镜像被污染，实验结果就不可复现。

如需重新拉取（多镜像兜底）：

```bash
python scripts/download_data.py --force
```

镜像列表（`aether/data.py`）：

```
raw.githubusercontent.com
cdn.jsdelivr.net
gh-proxy.com
```

### 4.2 字节级分词，零外部词表

不用 BPE：BPE 需要额外的词表文件与合并规则，引入一条外部依赖。
字节级（UTF-8）分词的好处：

- 词表固定 256，任何机器跑出来完全一致
- 中英混排天然支持
- 与 GPT-2 的 byte-level BPE 在信息论上等价（都是无损编码）

代价是序列变长（同等文本 token 数更多）——对教学规模实验完全够用。

### 4.3 独立 RNG 流

```python
# 数据采样用独立 generator，与模型初始化的 RNG 流互不干扰
self._gen = torch.Generator(device="cpu").manual_seed(seed)
```

**为什么重要**：如果共用全局 RNG，那么"改数据种子"会悄悄改变模型初始化，
ablation 就失去了意义——你以为在对比数据，其实连初始化都变了。

---

## 5. 训练可复现

### 5.1 完整流程

```bash
python scripts/train_nano.py --steps 1200 --seed 1337 --threads 6
```

报告写入 `experiments/train-nano-report.json`，包含：

| 字段 | 用途 |
|---|---|
| `weight_fingerprint` | 权重 SHA-256（前 16 位），供第三方比对 |
| `final_val_loss` / `final_perplexity` | 结果指标 |
| `history` | 训练期运行时证书（梯度范数、MoE 负载…） |
| `config` / `dataset` / `env` | 完整环境快照 |
| `ablation` | 消融开关状态 |

### 5.2 影响可复现性的因素

| 因素 | 是否影响指纹 |
|---|---|
| seed | ✅ 是（设计如此） |
| **线程数** | ⚠️ **是**（并行归约分块） |
| batch size / seq len | ✅ 是 |
| torch 版本 | ⚠️ 可能（算子实现变化） |
| CPU 型号 | ⚠️ 可能（SIMD 指令差异，通常 fp32 一致） |
| GPU 型号 | ✅ 是（不同架构结果不同） |

**结论**：跨硬件的 bit-exact 复现是不现实的承诺。
本仓库承诺的是**同机同配置**下的 bit-exact 复现，
并给出完整的环境快照让差异可归因。

### 5.3 线程数的实际影响

本仓库实测（16 核 CPU，nano 档，seq=512 前向）：

| threads | 1 | 2 | 4 | 6 | 8 | 16 |
|---|---|---|---|---|---|---|
| ms | 2143 | 1526 | 1274 | **1243** | 1195 | 3072 |

**线程数不是越多越好**（超线程争用 + 小矩阵调度开销）。
默认 6，可用 `scripts/benchmark.py --scan-threads` 自行校准。

注意：换线程数会改变指纹——这是**预期行为**，不是 bug，
因为并行归约的分块顺序确实变了。

---

## 6. 如何验证第三方报告的数字

假设某人声称"我用 AETHER-Ω nano 跑出了 val loss 2.34，指纹 `a1b2c3d4...`"：

```bash
# 1. 用相同配置跑
python scripts/train_nano.py --steps 1200 --seed 1337 --threads 6

# 2. 比对指纹
python - <<'PY'
import json
a = json.load(open("experiments/train-nano-report.json"))
print("我的指纹:", a["weight_fingerprint"])
print("我的 val :", a["final_val_loss"])
PY
```

指纹相同 ⇒ 你们的权重 bit-exact 一致，任何下游结论都可直接互认。
指纹不同 ⇒ 先看环境快照（`env` 字段）差异：torch 版本、线程数、device。

---

## 7. 已知的限制（诚实清单）

1. **跨硬件不承诺 bit-exact**。CPU 与 GPU、不同 GPU 架构之间结果必然不同。
   承诺范围是**同机同配置**。
2. **torch 版本可能引入差异**。算子实现的改进会改变浮点结果。
   报告里记录了 `torch.version`，比对时先对齐版本。
3. **`warn_only=True` 会静默放弃确定性**。生产需要性能时可能不得不开，
   但那样就不能声称可复现——本仓库默认 `False`。
4. **bf16/fp16 训练的可复现性更脆弱**。低精度下累积误差更大，
   非确定归约的影响被放大。本仓库默认 fp32。

---

## 8. 提交 bug

如果 `reproduce.py` 在你机器上失败，请附：

```bash
python -c "import torch,platform,sys;print(sys.version);print(platform.platform());print(torch.__version__)"
python scripts/reproduce.py --steps 60 2>&1 | tail -40
cat experiments/reproducibility.json
```

以及 CPU 型号、线程数、操作系统。
