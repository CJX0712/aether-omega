# 证书报告 · AETHER-Ω Certificate Bundle (nano)

**总计 30 条 · 通过 30 · 失败 0 · ✅ 全绿**

| 编号 | 类别 | 命题 | 实测残差 | 允许上界 | 余量 | 状态 |
|---|---|---|---|---|---|---|
| `GRAD-01` | GRAD | 解析梯度(autograd) 与 中心差分 ∂f/∂xᵢ≈[f(x+εeᵢ)−f(x−εeᵢ)]/(2ε) 的最大相对误差 ≤ 5e-5 | 7.616e-09 | 5.000e-05 | 0.00× | ✅ |
| `GRAD-02` | GRAD | 解析梯度(autograd) 与 中心差分 ∂f/∂xᵢ≈[f(x+εeᵢ)−f(x−εeᵢ)]/(2ε) 的最大相对误差 ≤ 5e-5 | 1.493e-08 | 5.000e-05 | 0.00× | ✅ |
| `GRAD-03` | GRAD | 解析梯度(autograd) 与 中心差分 ∂f/∂xᵢ≈[f(x+εeᵢ)−f(x−εeᵢ)]/(2ε) 的最大相对误差 ≤ 5e-5 | 1.883e-12 | 5.000e-05 | 0.00× | ✅ |
| `GRAD-04` | GRAD | 解析梯度(autograd) 与 中心差分 ∂f/∂xᵢ≈[f(x+εeᵢ)−f(x−εeᵢ)]/(2ε) 的最大相对误差 ≤ 5e-5 | 4.461e-10 | 5.000e-05 | 0.00× | ✅ |
| `GRAD-05` | GRAD | 解析梯度(autograd) 与 中心差分 ∂f/∂xᵢ≈[f(x+εeᵢ)−f(x−εeᵢ)]/(2ε) 的最大相对误差 ≤ 5e-5 | 9.490e-09 | 5.000e-05 | 0.00× | ✅ |
| `GRAD-06` | GRAD | 解析梯度(autograd) 与 中心差分 ∂f/∂xᵢ≈[f(x+εeᵢ)−f(x−εeᵢ)]/(2ε) 的最大相对误差 ≤ 5e-5 | 9.490e-09 | 5.000e-05 | 0.00× | ✅ |
| `GRAD-07` | GRAD | 解析梯度(autograd) 与 中心差分 ∂f/∂xᵢ≈[f(x+εeᵢ)−f(x−εeᵢ)]/(2ε) 的最大相对误差 ≤ 5e-5 | 4.601e-10 | 5.000e-05 | 0.00× | ✅ |
| `GRAD-09` | GRAD | 解析梯度(autograd) 与 中心差分 ∂f/∂xᵢ≈[f(x+εeᵢ)−f(x−εeᵢ)]/(2ε) 的最大相对误差 ≤ 5e-5 | 6.274e-12 | 5.000e-05 | 0.00× | ✅ |
| `GRAD-08` | GRAD | 解析梯度(autograd) 与 中心差分 ∂f/∂xᵢ≈[f(x+εeᵢ)−f(x−εeᵢ)]/(2ε) 的最大相对误差 ≤ 5e-5 | 3.002e-09 | 5.000e-05 | 0.00× | ✅ |
| `EQUIV-01` | EQUIV | MLA 的朴素展开实现与矩阵吸收实现输出相对误差 ≤ 1e-4 | 3.073e-07 | 1.000e-04 | 0.00× | ✅ |
| `EQUIV-02` | EQUIV | Gram Newton-Schulz 与原版 Newton-Schulz 输出相对误差 ≤ 1e-4 | 3.380e-06 | 1.000e-04 | 0.03× | ✅ |
| `EQUIV-03` | EQUIV | 增量解码（KV cache）与全序列重算的最后位置 logits 相对误差 ≤ 1e-4 | 1.521e-07 | 1.000e-04 | 0.00× | ✅ |
| `EQUIV-04` | EQUIV | MoE 的稀疏分组实现与稠密全专家加 mask 的参考实现相对误差 ≤ 1e-4 | 2.167e-07 | 1.000e-04 | 0.00× | ✅ |
| `INVAR-01` | INVAR | 正交化后所有奇异值与 1 的偏离 ≤ 0.5（近似正交，追求方向均衡而非精确实正交） | 3.182e-01 | 5.000e-01 | 0.64× | ✅ |
| `INVAR-02` | INVAR | ⟨RoPE(q,m), RoPE(k,n)⟩ = ⟨RoPE(q,m−n), k⟩ 相对误差 ≤ 1e-4 | 4.401e-06 | 1.000e-04 | 0.04× | ✅ |
| `INVAR-03` | INVAR | RMSNorm 输出方向等于输入按 weight 缩放后的方向，误差 ≤ 1e-5 | 2.980e-08 | 1.000e-05 | 0.00× | ✅ |
| `INVAR-04` | INVAR | MoE top-k 路由权重之和 = 1，误差 ≤ 1e-5 | 5.960e-08 | 1.000e-05 | 0.01× | ✅ |
| `INVAR-05` | INVAR | softmax(z + c) = softmax(z)，误差 ≤ 1e-5 | 2.861e-06 | 1.000e-05 | 0.29× | ✅ |
| `INVAR-06` | INVAR | 修改位置 t 之后的输入，位置 ≤ t 的输出变化 ≤ 1e-5 | 4.768e-07 | 1.000e-05 | 0.05× | ✅ |
| `INVAR-07` | INVAR | softmax 后注意力权重每行和 = 1，误差 ≤ 1e-5 | 2.384e-07 | 1.000e-05 | 0.02× | ✅ |
| `INVAR-08` | INVAR | 改变 n_heads 时每 token 的 KV cache 维度保持不变 | 0.000e+00 | 0.000e+00 | — | ✅ |
| `INVAR-09` | INVAR | RMSNorm(αx) = RMSNorm(x)（α>0，0-齐次/尺度不变），误差 ≤ 1e-5 | 1.669e-06 | 1.000e-05 | 0.17× | ✅ |
| `DETERM-01` | DETERM | 同一 seed 下两次独立训练的权重 SHA-256 指纹完全相同（bit-exact） | 0.000e+00 | 0.000e+00 | — | ✅ |
| `DETERM-02` | DETERM | 不同 seed 应产生不同权重指纹（证明 seed 真实生效） | 0.000e+00 | 0.000e+00 | — | ✅ |
| `DETERM-03` | DETERM | deterministic_algorithms=True 且 cudnn.deterministic=True 且 benchmark=False | 0.000e+00 | 0.000e+00 | — | ✅ |
| `BOUND-01` | BOUND | fp32 前向相对 fp64 的误差 ≤ 1e-4 | 2.706e-07 | 1.000e-04 | 0.00× | ✅ |
| `BOUND-02` | BOUND | 4096 长度的 RoPE 角度表 fp32 误差 ≤ 1e-5 | 0.000e+00 | 1.000e-05 | 0.00× | ✅ |
| `BOUND-03` | BOUND | MoE 最大负载 / 平均负载 ≤ 3.0（偏置路由收敛后） | 1.057e+00 | 3.000e+00 | 0.35× | ✅ |
| `MONO-01` | MONO | 全批量 + 足够小步长下，每步 loss 上升量 ≤ 1e-9 | 0.000e+00 | 1.000e-09 | 0.00× | ✅ |
| `MONO-02` | MONO | Muon 单步更新的奇异值最大/最小比 ≤ 3.0 | 1.664e+00 | 3.000e+00 | 0.55× | ✅ |

## 命题详情
### `GRAD-01` · RMSNorm 梯度
- **命题**：解析梯度(autograd) 与 中心差分 ∂f/∂xᵢ≈[f(x+εeᵢ)−f(x−εeᵢ)]/(2ε) 的最大相对误差 ≤ 5e-5
- **判据**：value <= tolerance
- **说明**：检查 256 个分量，‖∇‖=128.7487
- **结果**：PASS

### `GRAD-02` · QK-Norm 梯度
- **命题**：解析梯度(autograd) 与 中心差分 ∂f/∂xᵢ≈[f(x+εeᵢ)−f(x−εeᵢ)]/(2ε) 的最大相对误差 ≤ 5e-5
- **判据**：value <= tolerance
- **说明**：检查 256 个分量
- **结果**：PASS

### `GRAD-03` · RoPE 梯度
- **命题**：解析梯度(autograd) 与 中心差分 ∂f/∂xᵢ≈[f(x+εeᵢ)−f(x−εeᵢ)]/(2ε) 的最大相对误差 ≤ 5e-5
- **判据**：value <= tolerance
- **说明**：检查 256 个分量
- **结果**：PASS

### `GRAD-04` · SwiGLU 梯度
- **命题**：解析梯度(autograd) 与 中心差分 ∂f/∂xᵢ≈[f(x+εeᵢ)−f(x−εeᵢ)]/(2ε) 的最大相对误差 ≤ 5e-5
- **判据**：value <= tolerance
- **说明**：SwiGLU(D=256, I=512)，检查 256 个分量
- **结果**：PASS

### `GRAD-05` · MLA(naive) 梯度
- **命题**：解析梯度(autograd) 与 中心差分 ∂f/∂xᵢ≈[f(x+εeᵢ)−f(x−εeᵢ)]/(2ε) 的最大相对误差 ≤ 5e-5
- **判据**：value <= tolerance
- **说明**：MLA naive 路径，检查 256 个分量
- **结果**：PASS

### `GRAD-06` · MLA(absorb) 梯度
- **命题**：解析梯度(autograd) 与 中心差分 ∂f/∂xᵢ≈[f(x+εeᵢ)−f(x−εeᵢ)]/(2ε) 的最大相对误差 ≤ 5e-5
- **判据**：value <= tolerance
- **说明**：MLA absorb 路径，检查 256 个分量
- **结果**：PASS

### `GRAD-07` · DeepSeekMoE 专家梯度
- **命题**：解析梯度(autograd) 与 中心差分 ∂f/∂xᵢ≈[f(x+εeᵢ)−f(x−εeᵢ)]/(2ε) 的最大相对误差 ≤ 5e-5
- **判据**：value <= tolerance
- **说明**：DeepSeekMoE(E=8,k=2) 连续部分：固定路由权重下的专家 FFN 梯度，检查 256 个分量。top-k 离散选择不可微，单独由 GRAD-09 检验其可微的打分部分
- **结果**：PASS

### `GRAD-09` · MoE 路由打分梯度
- **命题**：解析梯度(autograd) 与 中心差分 ∂f/∂xᵢ≈[f(x+εeᵢ)−f(x−εeᵢ)]/(2ε) 的最大相对误差 ≤ 5e-5
- **判据**：value <= tolerance
- **说明**：路由打分 sigmoid(w_gate·x) 的梯度（可微部分），不含离散 top-k
- **结果**：PASS

### `GRAD-08` · Newton-Schulz 梯度
- **命题**：解析梯度(autograd) 与 中心差分 ∂f/∂xᵢ≈[f(x+εeᵢ)−f(x−εeᵢ)]/(2ε) 的最大相对误差 ≤ 5e-5
- **判据**：value <= tolerance
- **说明**：Gram Newton-Schulz 5 步迭代的梯度（正交化算子可微性）。ε 由实测标定：迭代展开后是 5⁵=3125 次多项式，ε 过大则截断误差主导、过小则舍入误差主导，实测呈 U 型——ε=1e-3→1.5e-5，1e-4→1.6e-7（最优），1e-6→5.2e-7，1e-7→1.2e-5，1e-8→1.2e-4
- **结果**：PASS

### `EQUIV-01` · MLA naive ≡ absorb
- **命题**：MLA 的朴素展开实现与矩阵吸收实现输出相对误差 ≤ 1e-4
- **判据**：value <= tolerance
- **说明**：逐层相对误差 ['1.80e-07', '2.80e-07', '2.03e-07', '3.07e-07', '2.72e-07', '2.87e-07', '2.45e-07', '2.24e-07']；推导：q·kᵀ=(qᶜWᵁᴷ)cᵀ+qᴿkᴿᵀ
- **依据**：DeepSeek-V2 arXiv:2405.04434 §MLA 吸收形式
- **结果**：PASS

### `EQUIV-02` · Gram NS ≡ 原版 NS
- **命题**：Gram Newton-Schulz 与原版 Newton-Schulz 输出相对误差 ≤ 1e-4
- **判据**：value <= tolerance
- **说明**：令 Pₖ=aI+bGₖ+cGₖ²，则 Gₖ₊₁=Pₖ²Gₖ，Xₙ=(Π Pₖ)X₀ —— 代数恒等，非近似
- **依据**：Gram Newton-Schulz (2026)；Keller Jordan Muon 系数 (3.4445,−4.7750,2.0315)
- **结果**：PASS

### `EQUIV-03` · KV cache ≡ 全序列重算
- **命题**：增量解码（KV cache）与全序列重算的最后位置 logits 相对误差 ≤ 1e-4
- **判据**：value <= tolerance
- **说明**：增量解码 logits 与全序列重算的相对误差；贪心 next-token 一致=True
- **依据**：MLA KV cache 只存 c^KV 与 k^R
- **结果**：PASS

### `EQUIV-04` · MoE 稀疏 ≡ 稠密+mask
- **命题**：MoE 的稀疏分组实现与稠密全专家加 mask 的参考实现相对误差 ≤ 1e-4
- **判据**：value <= tolerance
- **说明**：稀疏实现（只对 top-2 分组计算 + index_add）≡ 稠密实现（全部 8 专家 × mask 加权）；两者计算同一数学对象，差异只来自浮点累加顺序
- **依据**：DeepSeekMoE arXiv:2405.04434
- **结果**：PASS

### `INVAR-01` · Newton-Schulz 方向均衡
- **命题**：正交化后所有奇异值与 1 的偏离 ≤ 0.5（近似正交，追求方向均衡而非精确实正交）
- **判据**：value <= tolerance
- **说明**：正交化后所有奇异值与 1 的最大偏离 0.318（实测区间约 [0.68, 1.13]）；各矩阵条件数改善：(64, 128): 奇异值比 5.3→1.66；(128, 256): 奇异值比 5.7→1.66；(256, 512): 奇异值比 5.7→1.66
- **依据**：Higham (1986) Newton-Schulz；Muon 系数 (3.4445,−4.7750,2.0315)；Jordan et al. 2024 称之为 approximate orthogonalization
- **结果**：PASS

### `INVAR-02` · RoPE 相对性
- **命题**：⟨RoPE(q,m), RoPE(k,n)⟩ = ⟨RoPE(q,m−n), k⟩ 相对误差 ≤ 1e-4
- **判据**：value <= tolerance
- **说明**：随机 50 组 (q,k,m,n) 验证 ⟨R_m q, R_n k⟩ = ⟨R_{m−n} q, k⟩；这是 RoPE 能表达**相对**位置的根本性质
- **依据**：Su et al. (2021) RoFormer / RoPE
- **结果**：PASS

### `INVAR-03` · RMSNorm 方向保持
- **命题**：RMSNorm 输出方向等于输入按 weight 缩放后的方向，误差 ≤ 1e-5
- **判据**：value <= tolerance
- **说明**：RMSNorm 只做重缩放、不做去均值与平移 ⇒ 输出方向 = 输入的逐元素缩放方向
- **依据**：Zhang & Sennrich (2019) RMSNorm
- **结果**：PASS

### `INVAR-04` · MoE 路由权重归一
- **命题**：MoE top-k 路由权重之和 = 1，误差 ≤ 1e-5
- **判据**：value <= tolerance
- **说明**：top-2 权重归一化后行和 = 1；权重用**未加偏置**的分数，保证偏置不污染梯度
- **依据**：DeepSeek-V3 无辅助损失偏置路由
- **结果**：PASS

### `INVAR-05` · Softmax 平移不变
- **命题**：softmax(z + c) = softmax(z)，误差 ≤ 1e-5
- **判据**：value <= tolerance
- **说明**：softmax(z+c)=softmax(z) 是 max-subtract 数值稳定技巧的理论依据
- **依据**：logsumexp 稳定性
- **结果**：PASS

### `INVAR-06` · 因果性（未来不泄漏）
- **命题**：修改位置 t 之后的输入，位置 ≤ t 的输出变化 ≤ 1e-5
- **判据**：value <= tolerance
- **说明**：篡改位置 20+ 后：位置 0–19 输出最大变化 4.77e-07（应为 0）；位置 20+ 变化 5.08e+00（应显著非零，证明检验有效）
- **依据**：自回归因果掩码
- **结果**：PASS

### `INVAR-07` · 注意力行和归一
- **命题**：softmax 后注意力权重每行和 = 1，误差 ≤ 1e-5
- **判据**：value <= tolerance
- **说明**：注意力权重每行和为 1（softmax 归一）+ 因果掩码下三角可见
- **结果**：PASS

### `INVAR-08` · MLA KV cache 与头数无关
- **命题**：改变 n_heads 时每 token 的 KV cache 维度保持不变
- **判据**：value == tolerance
- **说明**：头数→cache维度 [(4, 48), (8, 48), (16, 48), (32, 48)]；同配置 MHA 需 1024 维/token，MLA 只需 48 维/token，压缩比 21.3×
- **依据**：DeepSeek-V2 MLA：KV cache 与 n_heads 解耦
- **结果**：PASS

### `INVAR-09` · RMSNorm 尺度不变性
- **命题**：RMSNorm(αx) = RMSNorm(x)（α>0，0-齐次/尺度不变），误差 ≤ 1e-5
- **判据**：value <= tolerance
- **说明**：RMSNorm 关于缩放**不变**（0-齐次）：分子 αx 与分母 RMS(αx)=|α|·RMS(x) 同步缩放，α 被完全抵消 ⇒ 输出范数恒为 O(√d)，与输入尺度无关。残差流因此不会随深度失控——这是它在 2026 年全面取代 LayerNorm 的关键原因。（本条初版曾误写为 1-齐次，由 JS-06 交叉验证时发现并更正）
- **结果**：PASS

### `DETERM-01` · 位级确定性复现
- **命题**：同一 seed 下两次独立训练的权重 SHA-256 指纹完全相同（bit-exact）
- **判据**：value == tolerance
- **说明**：权重 SHA-256 指纹（前16位）：run1=5def95444b04603f run2=5def95444b04603f；loss 序列最大漂移=0.000e+00；两次在**独立进程上下文**中构建模型
- **依据**：PyTorch Reproducibility / CUBLAS_WORKSPACE_CONFIG=:4096:8
- **结果**：PASS

### `DETERM-02` · 种子敏感性
- **命题**：不同 seed 应产生不同权重指纹（证明 seed 真实生效）
- **判据**：value == tolerance
- **说明**：seed=1337 → 7382a6142e6aa69d；seed=2026 → 22781ed2f786b43f；两者必须不同，否则说明 seed 根本没生效（确定性开关把随机性也关掉了）
- **结果**：PASS

### `DETERM-03` · 确定性环境开关
- **命题**：deterministic_algorithms=True 且 cudnn.deterministic=True 且 benchmark=False
- **判据**：value == tolerance
- **说明**：环境快照：{'seed': 1, 'threads': 4, 'warn_only': False, 'cudnn_deterministic': True, 'cudnn_benchmark': False, 'deterministic_algorithms': True, 'cublas_workspace_config': ':4096:8', 'torch_version': '2.14.1+cpu', 'device': 'cpu'}
- **结果**：PASS

### `BOUND-01` · fp32 前向误差界
- **命题**：fp32 前向相对 fp64 的误差 ≤ 1e-4
- **判据**：value <= tolerance
- **说明**：fp32 相对 fp64 的前向误差；fp32 机器精度 u=1.19e-7，8 层网络理论最坏界 ~1e-3，实测应远低于此
- **依据**：Higham, Accuracy and Stability of Numerical Algorithms, Ch.2
- **结果**：PASS

### `BOUND-02` · RoPE 长序列角度精度
- **命题**：4096 长度的 RoPE 角度表 fp32 误差 ≤ 1e-5
- **判据**：value <= tolerance
- **说明**：L=4096 的 RoPE 角度表 fp32 vs fp64；角度表在长上下文下会被放大，故单独成证
- **结果**：PASS

### `BOUND-03` · MoE 负载均衡界
- **命题**：MoE 最大负载 / 平均负载 ≤ 3.0（偏置路由收敛后）
- **判据**：value <= tolerance
- **说明**：最大负载/平均负载 = 1.06，专家利用率 = 1.00；无辅助损失偏置路由的目标是把比值压到 1 附近（上界取 3.0 为健康阈值）
- **依据**：DeepSeek-V3 无辅助损失负载均衡
- **结果**：PASS

### `MONO-01` · 下降引理（全批量单调性）
- **命题**：全批量 + 足够小步长下，每步 loss 上升量 ≤ 1e-9
- **判据**：value <= tolerance
- **说明**：全批量 SGD(η=0.001) 30 步，loss 6.285588 → 4.736981；最大单步上升 0.000e+00
- **依据**：下降引理：η ≤ 1/L ⇒ f(x_{k+1}) ≤ f(x_k) − (η/2)‖∇f‖²
- **结果**：PASS

### `MONO-02` · Muon 更新各向同性
- **命题**：Muon 单步更新的奇异值最大/最小比 ≤ 3.0
- **判据**：value <= tolerance
- **说明**：Muon 更新矩阵的奇异值最大/最小 = 1.66；原梯度（未正交化）的该比值通常在 10~100，正交化把它压到 O(1)，这正是 Muon '各方向均匀发力' 的量化体现
- **依据**：Muon / Newton-Schulz 正交化更新
- **结果**：PASS
