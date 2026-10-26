# AETHER-Ω 数学基础

> 本文只写本仓库**真正用到**的数学，每条都配一条可机器验证的性质（证书编号）。
> 不堆砌与实现无关的理论。

---

## 1. 线性代数

### 1.1 正交化与 polar 分解

任何矩阵 `G ∈ R^{m×n}`（m ≤ n）都有 polar 分解：

```
G = U P        U 的列正交（UᵀU = I），P 对称半正定
```

Muon 需要的正是 `U` —— 与 `G` "方向相同但所有方向等权"的矩阵。

**精确求法**：SVD，`G = UΣVᵀ` ⇒ `U_polar = U Vᵀ`。
**问题**：SVD 每个 step 都做一遍太贵，且 GPU 上效率低。

**Newton-Schulz 迭代**（Higham 1986）：

```
X ← (aI + b·XXᵀ + c·(XXᵀ)²) X
```

对奇异值的作用：`σ' = f(σ) = σ(a + bσ² + cσ⁴)`。

系数 `(a,b,c) = (3.4445, −4.7750, 2.0315)` 由 Keller Jordan 团队调优，
使 5 步迭代在 bf16 下稳定收敛（多项式次数与收敛速度的权衡）。

**必须知道的真相**：这个迭代在不动点处**不收敛**（见 §1.2），是"近似正交化"。

> 证书：INVAR-01（方向均衡）、EQUIV-02（Gram 版等价）、GRAD-08（可微性）

### 1.2 不动点与稳定性分析

```
不动点：f(σ) = σ ⇒ p(σ) := a + bσ² + cσ⁴ = 1
       2.0315σ⁴ − 4.775σ² + 2.4445 = 0
       ⇒ σ* ≈ 0.868 或 1.264

稳定性：f'(σ) = a + 3bσ² + 5cσ⁴
       f'(0.868) ≈ 3.4445 − 10.786 + 5.760 = −1.582
       |f'| > 1  ⇒ 迭代在不动点附近振荡，不收敛
```

实测：5 步后奇异值 `[0.68, 1.13]`；20、30 步后残差仍停在 ~0.2。

**结论**：NS 的价值不在"精确正交"，而在**把条件数从 O(10²) 压到 O(1)**：

| 矩阵 | 正交化前 σ_max/σ_min | 正交化后 |
|---|---|---|
| 64×128 | 8.7 | 1.9 |
| 128×256 | 5.8 | 1.66 |

### 1.3 Gram 矩阵的代数结构

令 `G_k = X_k X_kᵀ`（m×m，对称半正定），`P_k = aI + b G_k + c G_k²`。

因为 `P_k` 是 `G_k` 的多项式：

1. `P_k` 对称（多项式保持对称性）
2. `P_k` 与 `G_k` 可交换（同为 `G_k` 的多项式）

于是：

```
G_{k+1} = X_{k+1} X_{k+1}ᵀ
        = (P_k X_k)(P_k X_k)ᵀ
        = P_k G_k P_kᵀ
        = P_k G_k P_k        （P_k 对称）
        = P_k² G_k           （可交换）
```

**推论**：整个迭代可以在 m×m 空间完成，与 n 无关。
最后 `X_n = (Π P_k) X_0` 一次拉回。这就是 Gram NS 的全部数学。

> 证书：EQUIV-02（实测残差 3.4e-06，JS 版 3.6e-15）、JS-11（Gram 对称半正定）

### 1.4 Eckart–Young 与低秩压缩

MLA 的 KV 压缩本质是：用低秩潜在 `c^{KV} ∈ R^{d_c}` 表示原本的 `n_h × d_h` 维 K/V。

Eckart–Young 定理给出：秩-k 截断 SVD 是 Frobenius / 谱范数意义下的**最优**秩-k 近似。
MLA 不是做 SVD（那样太贵），而是**学习**这个低秩映射（`W^{DKV}` + `W^{UK}`），
因此是有损但任务自适应的压缩——实测比 SVD 更好，因为目标不是重构误差最小，而是下游 loss 最小。

---

## 2. 微积分与优化

### 2.1 下降引理（MONO-01 的依据）

若 `f` 是 L-光滑的（‖∇f(x) − ∇f(y)‖ ≤ L‖x − y‖），则梯度下降步长 `η ≤ 1/L` 时：

```
f(x_{k+1}) ≤ f(x_k) − (η/2)·‖∇f(x_k)‖²
```

**这是严格保证，不是经验规律**。推论：全批量 + 足够小 η 时 loss 必须单调不增。
如果不单调 ⇒ 学习率过大 / 梯度算错 / 优化器有 bug。

> 证书：MONO-01（30 步全批量 SGD，最大单步上升 = 0）

### 2.2 AdamW 的解耦权重衰减

Loshchilov & Hutter (2019) 的关键洞察：L2 正则与自适应学习率耦合后，
衰减量会被梯度尺度调制，导致正则强度失控。解法是**解耦**：

```
# Adam：θ ← θ − η·(m̂/(√v̂+ε) + λθ)     ← 衰减被 √v̂ 调制，错
# AdamW：θ ← θ − η·m̂/(√v̂+ε) − ηλθ      ← 衰减独立于梯度统计，对
```

本仓库的 Muon 同样采用解耦衰减（`p.mul_(1 - lr*wd)`）。

### 2.3 Muon 的更新尺度对齐

正交化后 `Ũ` 的所有奇异值 ≈ 1，其 RMS 为 `1/√(max(m,n))`。
为了让 Muon 的 lr 语义与 AdamW 可比，Keller 的实现乘上：

```
scale = max(1, m/n)^{1/2}
```

这使更新的 RMS 落在 ~0.2（AdamW 的典型量级），
于是同一套 lr 直觉在两个优化器间大致通用。

### 2.4 链式法则与梯度检验

反向传播的全部数学就是链式法则。它之所以需要被验证，不是因为数学难，
而是因为**实现极易出错**（漏项、转置写反、缩放错位），
而这类错误在训练中**不会崩溃**，只会让模型学歪——肉眼几乎不可能发现。

> 证书：GRAD-01 ~ GRAD-09

---

## 3. 数值分析

### 3.1 浮点模型

IEEE 754：`fl(a op b) = (a op b)(1 + δ)`，`|δ| ≤ u`

| | u（单位舍入） | 十进制 |
|---|---|---|
| fp32 | 2⁻²⁴ | ≈ 5.96e-8 |
| fp64 | 2⁻⁵³ | ≈ 1.11e-16 |
| bf16 | 2⁻⁸ | ≈ 3.9e-3 |

### 3.2 累积误差界（所有 tolerance 的来源）

对 n 个数求和，浮点累积的相对误差上界：

```
γ_n = n·u / (1 − n·u) ≈ n·u
```

本仓库等价性检查涉及 `O(L·d) ≈ 1.6e4` 项累加：

```
理论上界 ≈ 1.6e4 × 1.19e-7 ≈ 1.9e-3
实测     ≈ 2e-7
tolerance = 1e-4   ← 夹在两者之间：既不放过真错，也不误报
```

### 3.3 数值稳定的三个标准技巧

| 技巧 | 数学依据 | 证书 |
|---|---|---|
| **max-subtract softmax** | `softmax(z + c) = softmax(z)` | INVAR-05 / JS-05 |
| **logsumexp** | `log Σ e^{xᵢ} = m + log Σ e^{xᵢ−m}` | JS-07 |
| **低精度提升做规约** | fp16 下 `Σx²` 上界越界（65504） | GRAD-01 |

logsumexp 的必要性：

```
朴素：log(Σ exp([1000, 1001, 999])) = log(Infinity) = Infinity   ← 溢出
LSE ：1001 + log(1 + e^-1 + e^-2)  = 1001.407605964              ← 正确
```

### 3.4 中心差分的 ε 权衡

```
误差 ≈ O(ε²·|f'''|)  +  O(u/ε)
        └ 截断误差        └ 舍入误差
```

理论最优 `ε ≈ u^{1/3}`。实测（Newton-Schulz，3125 次多项式）：

| ε | 1e-3 | **1e-4** | 1e-5 | 1e-6 | 1e-7 | 1e-8 |
|---|---|---|---|---|---|---|
| 残差 | 1.5e-5 | **1.6e-7** | 3.0e-8 | 5.2e-7 | 1.2e-5 | 1.2e-4 |

**反直觉**：`ε = 1e-8` 比 `ε = 1e-4` 差三个数量级。

---

## 4. 概率与信息论

### 4.1 交叉熵损失

语言建模的目标是最大化似然 `Π P(y_t | y_{<t})`，等价于最小化 NLL：

```
L = −Σ_t log P(y_t | y_{<t}) = Σ_t CE(softmax(logits_t), y_t)
```

困惑度 `PPL = exp(L)`，是 loss 的可解释版本（"模型平均在多少个 token 间犹豫"）。

### 4.2 熵与 logit 爆炸

注意力 logit `z` 的 softmax 熵：

```
H(a) = −Σ a_i log a_i
```

当 logit 的绝对值无界增长时，softmax 趋近 one-hot ⇒ `H → 0` ⇒
梯度趋零 + 表示能力坍缩。这就是训练中后期 **loss spike** 的机制。

MuonClip 的解法（Kimi K2）：给 logit 加硬上界 τ：

```
S_max = max|logit| > τ  ⇒  q 侧、k 侧各乘 √γ，γ = τ/S_max
                          （logit 恰好乘 γ ⇒ 新的 S_max = τ）
```

**为什么缩放权重而不是 clamp logit**：clamp 会破坏 softmax 的可微性与信息量
（等于喂给模型一个错误的注意力分布）；而缩放产生 q/k 的**权重矩阵**
是参数空间的等价变换——前向数值变了，但函数类表达能力不变，且不引入梯度噪声。

### 4.3 MTP 与监督信号密度

标准 LM：每个位置 1 个监督信号。
MTP (depth=D)：每个位置 D+1 个。

信息论视角：MTP 是在最大化 `Σ_d I(h_t ; y_{t+1+d})`，
迫使 `h_t` 保留"未来 d 步还需要"的信息，而不是只保留"下一步"的信息。

---

## 5. 旋转与位置编码

### 5.1 RoPE 的复数形式

把每两维视为复数 `z = x_{2i} + i·x_{2i+1}`，位置 m 的旋转：

```
z' = z · e^{i·mθ_i},   θ_i = 1/base^{2i/d}
```

实部/虚部展开（即实现中的形式）：

```
out_{2i}   = x_{2i}·cos(mθ_i) − x_{2i+1}·sin(mθ_i)
out_{2i+1} = x_{2i+1}·cos(mθ_i) + x_{2i}·sin(mθ_i)
```

比构造 `d×d` 块对角旋转矩阵省 `O(d²)` 内存。

### 5.2 相对性的证明（INVAR-02）

```
⟨RoPE(q,m), RoPE(k,n)⟩
  = Re( Σ_i (q_i e^{i·mθ_i}) · conj(k_i e^{i·nθ_i}) )
  = Re( Σ_i q_i conj(k_i) e^{i(m−n)θ_i} )
  = ⟨RoPE(q, m−n), k⟩                                    ∎
```

实测残差 4.4e-06（PyTorch）/ 2.3e-15（JS）。

### 5.3 YaRN 的频率分治

```
wavelength_i = 2π · base^{2i/d}
```

- **低频（长波长）**：`wavelength >> 原始上下文` ⇒ 需要插值（否则位置超出训练分布）
- **高频（短波长）**：`wavelength << 原始上下文` ⇒ 不插值（保局部细节分辨率）
- **中间**：斜坡平滑过渡 `γ = clamp((ratio − β_slow)/(β_fast − β_slow), 0, 1)`

再乘注意力温度校正 `√(1/log s)` 补偿插值后分布熵的变化。

---

## 6. 复杂度

| 模块 | 时间 | 空间 |
|---|---|---|
| MLA 全局注意力 | `O(L²·d)` | `O(L·(d_c + d_h^R))` cache |
| MLA 局部注意力 | `O(L·w·d)` | 同上 |
| MLA absorb（推理） | `O(L·d_c)` per query | ← head 维消失，这是关键 |
| DeepSeekMoE | `O(T · k/E · E · I · d)` ≈ `O(T·k·I·d)` | `O(E·I·d)` 参数 |
| SwiGLU | `O(T·I·d)` | |
| Gram NS（m×n 矩阵） | `2·O(m²n) + 5·O(m³)` | `O(m²)` |
| 原版 NS | `10·O(m²n) + 5·O(m³)` | 同 |

`m × n` 越大（`n >> m`），Gram NS 的优势越明显。

---

## 7. 参考

- Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM
- Higham (1986), Newton's method for the matrix square root
- Björck & Bowie (1971), An iterative algorithm for computing the best estimate of an orthogonal matrix
- Jordan et al. (2024), Muon: Momentum Orthogonalized by Newton-Schulz
- Liu et al. (arXiv:2502.16982), Muon at scale（Moonlight）
- Wen et al. (arXiv:2512.05620), Muon 收益随规模的复现研究
- Su et al. (2021), RoFormer / RoPE
- Peng et al. (2023), YaRN
- Zhang & Sennrich (2019), RMSNorm
- Shazeer (2020), GLU Variants / SwiGLU
- Gloeckle et al. (arXiv:2404.19737), Multi-Token Prediction
- DeepSeek-AI (arXiv:2405.04434), DeepSeek-V2: MLA + DeepSeekMoE
- DeepSeek-AI (arXiv:2412.19437), DeepSeek-V3: 无辅助损失负载均衡
- Kimi (arXiv:2507.20534), MuonClip
- Loshchilov & Hutter (2019), Decoupled Weight Decay (AdamW)
