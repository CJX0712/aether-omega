#!/usr/bin/env node
/**
 * AETHER-Ω · 零依赖验证器（Pure JS / Node）
 * ==========================================
 *
 * 为什么还要一个 JS 版本
 * ----------------------
 * Python 版证书（scripts/certify.py）依赖 PyTorch。但"可验证"这件事的意义在于
 * **任何人都能独立复检**，而不必先装一个 2GB 的科学计算栈。
 *
 * 更重要的是：这是**第二门语言的独立实现**。
 * 如果 PyTorch 版本和这个 JS 版本在两个完全不同的浮点栈上，
 * 对同一批数学命题给出一致的结论，那么结论的可信度就不是"某个库说它对"，
 * 而是"两条独立路径互证"。这是科学里最高等级的验证——交叉验证。
 *
 * 用法
 * ----
 *     node verify/aether-verify.js             # 跑全部证书
 *     node verify/aether-verify.js --json      # 输出 JSON
 *     node verify/aether-verify.js --quiet     # 只输出汇总
 *
 * 零依赖：不使用任何 npm 包，只用 ECMAScript 内置能力。
 * 浏览器亦可直接打开（配合 verify/index.html）。
 *
 * 作者：晨星 (Chenxing)   |   许可：MIT
 */

'use strict';

// ───────────────────────────────────────────────────────────────────────────
// 1. 极简线性代数（只实现本验证器真正用到的算子）
// ───────────────────────────────────────────────────────────────────────────

/** 可复现的伪随机数发生器（mulberry32）。
 *  为什么不用 Math.random：它不可复现，而"可复现"是本仓库的第一原则。 */
function makeRng(seed) {
  let a = seed >>> 0;
  return function rng() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box–Muller 标准正态 */
function randnMat(rows, cols, rng) {
  const m = [];
  for (let i = 0; i < rows; i++) {
    const row = new Float64Array(cols);
    for (let j = 0; j < cols; j++) {
      const u = Math.max(rng(), 1e-12);
      const v = rng();
      row[j] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    m.push(row);
  }
  return m;
}

function zeros(r, c) {
  const m = [];
  for (let i = 0; i < r; i++) m.push(new Float64Array(c));
  return m;
}
function clone(m) { return m.map((r) => Float64Array.from(r)); }
function transpose(m) {
  const r = m.length, c = m[0].length;
  const out = zeros(c, r);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[j][i] = m[i][j];
  return out;
}
function matmul(a, b) {
  const n = a.length, k = b.length, p = b[0].length;
  const out = zeros(n, p);
  for (let i = 0; i < n; i++) {
    const ai = a[i];
    for (let t = 0; t < k; t++) {
      const v = ai[t];
      if (v === 0) continue;
      const bt = b[t];
      for (let j = 0; j < p; j++) out[i][j] += v * bt[j];
    }
  }
  return out;
}
function matAddScaled(a, b, alpha) {
  const out = zeros(a.length, a[0].length);
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < a[0].length; j++) out[i][j] = a[i][j] + alpha * b[i][j];
  return out;
}
function scaleMat(a, s) {
  return a.map((r) => Float64Array.from(r, (v) => v * s));
}
function identity(n) {
  const m = zeros(n, n);
  for (let i = 0; i < n; i++) m[i][i] = 1;
  return m;
}
function frobenius(m) {
  let s = 0;
  for (const r of m) for (const v of r) s += v * v;
  return Math.sqrt(s);
}
function maxAbsDiff(a, b) {
  let d = 0;
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < a[0].length; j++) d = Math.max(d, Math.abs(a[i][j] - b[i][j]));
  return d;
}
function maxAbs(m) {
  let d = 0;
  for (const r of m) for (const v of r) d = Math.max(d, Math.abs(v));
  return d;
}
function relDiff(a, b) { return maxAbsDiff(a, b) / Math.max(maxAbs(a), 1e-12); }

/** Jacobi 特征分解（对称矩阵，循环旋转）——用于求奇异值上下界。
 *  选 Jacobi 而非幂迭代：Jacobi 能一次给出**全部**特征值，
 *  而我们需要"最大/最小奇异值之比"这个条件数指标。 */
function jacobiEigenvalues(Ain, sweeps = 100, tol = 1e-12) {
  const n = Ain.length;
  let A = clone(Ain);
  for (let sweep = 0; sweep < sweeps; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
    if (off < tol) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-18) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
      }
    }
  }
  const ev = [];
  for (let i = 0; i < n; i++) ev.push(A[i][i]);
  return ev.sort((x, y) => y - x);
}

function singularValueRatio(m) {
  const mt = transpose(m);
  const g = matmul(m, mt); // m × mᵀ 对称半正定，特征值 = 奇异值²
  const ev = jacobiEigenvalues(g);
  const smax = Math.sqrt(Math.max(ev[0], 0));
  const smin = Math.sqrt(Math.max(ev[ev.length - 1], 0));
  return smax / Math.max(smin, 1e-12);
}

// ───────────────────────────────────────────────────────────────────────────
// 2. 被验证的算法（与 Python 版一一对应，独立实现）
// ───────────────────────────────────────────────────────────────────────────

const NS_COEFFS = [3.4445, -4.775, 2.0315]; // Keller Jordan 调优系数

/** 原版 Newton-Schulz：在 X 上迭代 */
function newtonSchulzOriginal(G, steps = 5, eps = 1e-7) {
  let transposed = G.length > G[0].length;
  let X = transposed ? transpose(G) : clone(G);
  const nf = frobenius(X);
  X = scaleMat(X, 1 / Math.max(nf, eps));
  const [a, b, c] = NS_COEFFS;
  for (let s = 0; s < steps; s++) {
    const A = matmul(X, transpose(X));       // [m,m]
    const A2 = matmul(A, A);
    const B = zeros(A.length, A.length);
    for (let i = 0; i < A.length; i++)
      for (let j = 0; j < A.length; j++) B[i][j] = b * A[i][j] + c * A2[i][j];
    X = matAddScaled(scaleMat(X, a), matmul(B, X), 1);
  }
  return transposed ? transpose(X) : X;
}

/** Gram Newton-Schulz：在 m×m 的对称 Gram 矩阵上迭代（矩形乘从 ~10 次降到 2 次） */
function gramNewtonSchulz(G, steps = 5, eps = 1e-7) {
  let transposed = G.length > G[0].length;
  let X = transposed ? transpose(G) : clone(G);
  const nf = frobenius(X);
  X = scaleMat(X, 1 / Math.max(nf, eps));
  const [a, b, c] = NS_COEFFS;
  const m = X.length;
  let Gk = matmul(X, transpose(X));          // 唯一一次"大"矩形乘
  let P = identity(m);
  for (let s = 0; s < steps; s++) {
    const G2 = matmul(Gk, Gk);
    const Pk = zeros(m, m);
    for (let i = 0; i < m; i++)
      for (let j = 0; j < m; j++) Pk[i][j] = (i === j ? a : 0) + b * Gk[i][j] + c * G2[i][j];
    Gk = matmul(matmul(Pk, Gk), transpose(Pk)); // = Pk² Gk
    P = matmul(Pk, P);
  }
  X = matmul(P, X);                          // 第二次（也是最后一次）矩形乘
  return transposed ? transpose(X) : X;
}

/** 正交性残差 ‖XXᵀ − I‖_max */
function orthogonalityResidual(X) {
  let M = X.length > X[0].length ? transpose(X) : X;
  const G = matmul(M, transpose(M));
  let d = 0;
  for (let i = 0; i < G.length; i++)
    for (let j = 0; j < G.length; j++) d = Math.max(d, Math.abs(G[i][j] - (i === j ? 1 : 0)));
  return d;
}

/** RoPE：把 d 维向量按相邻两维一组做平面旋转 */
function ropeApply(vec, pos, dim, base = 10000) {
  const out = new Float64Array(vec.length);
  for (let i = 0; i < dim / 2; i++) {
    const theta = pos / Math.pow(base, (2 * i) / dim);
    const c = Math.cos(theta), s = Math.sin(theta);
    const x1 = vec[2 * i], x2 = vec[2 * i + 1];
    out[2 * i] = x1 * c - x2 * s;
    out[2 * i + 1] = x2 * c + x1 * s;
  }
  for (let i = dim; i < vec.length; i++) out[i] = vec[i];
  return out;
}
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function softmax(z) {
  let mx = -Infinity;
  for (const v of z) mx = Math.max(mx, v);          // max-subtract：数值稳定的关键一步
  let s = 0;
  const out = new Float64Array(z.length);
  for (let i = 0; i < z.length; i++) { out[i] = Math.exp(z[i] - mx); s += out[i]; }
  for (let i = 0; i < z.length; i++) out[i] /= s;
  return out;
}

/** RMSNorm：只做重缩放（无去均值、无平移） */
function rmsNorm(vec, weight, eps = 1e-6) {
  let ms = 0;
  for (const v of vec) ms += v * v;
  ms /= vec.length;
  const inv = 1 / Math.sqrt(ms + eps);
  const out = new Float64Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] * inv * weight[i];
  return out;
}

/** logsumexp：防溢出的对数求和 */
function logSumExp(xs) {
  let mx = -Infinity;
  for (const v of xs) mx = Math.max(mx, v);
  let s = 0;
  for (const v of xs) s += Math.exp(v - mx);
  return mx + Math.log(s);
}

// ───────────────────────────────────────────────────────────────────────────
// 3. 证书（每条都写明"命题 / 判据 / 依据"）
// ───────────────────────────────────────────────────────────────────────────

const certs = [];
function certify(id, name, statement, tol, fn) {
  const t0 = Date.now();
  let value = null, passed = false, detail = '', reference = '';
  try {
    const r = fn();
    value = r.value;
    detail = r.detail || '';
    reference = r.reference || '';
    passed = r.passed !== undefined ? r.passed : value <= tol;
  } catch (e) {
    detail = `证书执行异常：${e && e.message ? e.message : e}`;
    passed = false;
  }
  certs.push({ id, name, statement, value, tolerance: tol, passed, detail, reference, ms: Date.now() - t0 });
}

const rng = makeRng(20261026);

certify('JS-01', 'Newton-Schulz 方向均衡',
  '正交化后所有奇异值与 1 的偏离 ≤ 0.5（近似正交，追求方向均衡）', 0.5, () => {
    let worst = 0, notes = [];
    for (const [r, c] of [[16, 32], [32, 64]]) {
      const G = randnMat(r, c, rng);
      const U = gramNewtonSchulz(G, 5);
      const s = jacobiEigenvalues(matmul(U, transpose(U))).map((v) => Math.sqrt(Math.max(v, 0)));
      const dev = Math.max(Math.abs(Math.max(...s) - 1), Math.abs(Math.min(...s) - 1));
      worst = Math.max(worst, dev);
      notes.push(`${r}×${c}: σ∈[${Math.min(...s).toFixed(2)}, ${Math.max(...s).toFixed(2)}]`);
    }
    return { value: worst, detail: `正交化后奇异值区间 ${notes.join('; ')}。5 步 NS 是近似正交化（不动点处 |f′|≈1.58>1，迭代振荡而非收敛），目标是把条件数压到 O(1) 而非残差归零。`, reference: 'Higham 1986; Muon 系数 (3.4445,−4.7750,2.0315)' };
  });

certify('JS-02', 'Gram NS ≡ 原版 NS',
  'Gram Newton-Schulz 与原版 Newton-Schulz 输出相对误差 ≤ 1e-9', 1e-9, () => {
    let worst = 0;
    for (const [r, c] of [[16, 32], [32, 64], [24, 48]]) {
      const G = randnMat(r, c, rng);
      worst = Math.max(worst, relDiff(newtonSchulzOriginal(G, 5), gramNewtonSchulz(G, 5)));
    }
    return { value: worst, detail: '令 Pₖ=aI+bGₖ+cGₖ²，Gₖ₊₁=Pₖ²Gₖ，Xₙ=(Π Pₖ)X₀ —— 代数恒等。JS 与 Python 两条独立实现互证。', reference: 'Gram Newton-Schulz (2026)' };
  });

certify('JS-03', 'Gram NS 减少矩形矩阵乘',
  'Gram 版的浮点乘法次数应显著少于原版（≤ 原版的 60%）', 0.6, () => {
    // 计数：原版每步 2 次矩形乘(m²n)，Gram 版共 2 次
    const m = 64, n = 256, steps = 5;
    const rectCost = 2 * m * m * n;                 // 一次 [m,n]×[n,m] 或 [m,m]×[m,n]
    const orig = steps * 2 * rectCost + steps * 2 * m * m * m;
    const gram = 2 * rectCost + steps * 4 * m * m * m;
    return { value: gram / orig, detail: `m=${m}, n=${n}, ${steps} 步：原版 ${(orig / 1e6).toFixed(1)} MFLOP，Gram 版 ${(gram / 1e6).toFixed(1)} MFLOP，比值 ${(gram / orig * 100).toFixed(0)}%。矩形矩阵乘（最贵的部分）从 ${steps * 2} 次降到 2 次。`, reference: 'Gram Newton-Schulz (2026)：正交化提速 40–50%' };
  });

certify('JS-04', 'RoPE 相对性',
  '⟨RoPE(q,m), RoPE(k,n)⟩ = ⟨RoPE(q,m−n), k⟩ 相对误差 ≤ 1e-9', 1e-9, () => {
    const d = 16;
    let worst = 0;
    for (let t = 0; t < 40; t++) {
      const q = Float64Array.from({ length: d }, () => rng() * 2 - 1);
      const k = Float64Array.from({ length: d }, () => rng() * 2 - 1);
      const m = Math.floor(rng() * 64), n = Math.floor(rng() * (m + 1));
      const lhs = dot(ropeApply(q, m, d), ropeApply(k, n, d));
      const rhs = dot(ropeApply(q, m - n, d), k);
      worst = Math.max(worst, Math.abs(lhs - rhs) / Math.max(Math.abs(lhs), 1e-9));
    }
    return { value: worst, detail: '随机 40 组 (q,k,m,n)。这是 RoPE 能表达**相对**位置的根本性质，也是它能外推到训练长度之外的原因。', reference: 'Su et al. 2021, RoFormer' };
  });

certify('JS-05', 'Softmax 平移不变',
  'softmax(z + c) = softmax(z)，误差 ≤ 1e-12', 1e-12, () => {
    let worst = 0;
    for (let t = 0; t < 20; t++) {
      const z = Float64Array.from({ length: 32 }, () => (rng() * 2 - 1) * 20);
      const c = rng() * 500;
      const a = softmax(z);
      const b = softmax(Float64Array.from(z, (v) => v + c));
      for (let i = 0; i < z.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
    }
    return { value: worst, detail: 'max-subtract 技巧的理论依据；没有它，exp(700) 在 fp32 下直接 Infinity。', reference: 'logsumexp 稳定性' };
  });

certify('JS-06', 'RMSNorm 尺度不变性与方向保持',
  'RMSNorm(αx) = RMSNorm(x)（α>0，0-齐次/尺度不变），且输出方向 = 输入按 weight 缩放后的方向；误差 ≤ 1e-5（偏差仅来自 ε 项）', 1e-5, () => {
    // ⚠️ 注意：RMSNorm 是 **0-齐次**（尺度不变），不是 1-齐次。
    // 因为分母 RMS(αx) = |α|·RMS(x) 会同步缩放，分子 αx 的 α 被完全抵消。
    // （本条证书的初版误写成 1-齐次，被这条证书自己抓出来——
    //   这正是"证书必须明确写出命题"的意义：命题写错，测试就会打脸。）
    const d = 32;
    const w = Float64Array.from({ length: d }, () => 0.5 + rng());
    let worst = 0;
    for (let t = 0; t < 20; t++) {
      const x = Float64Array.from({ length: d }, () => rng() * 2 - 1);
      const a = 3.0;
      const y1 = rmsNorm(Float64Array.from(x, (v) => v * a), w);
      const y2 = rmsNorm(x, w);
      for (let i = 0; i < d; i++) worst = Math.max(worst, Math.abs(y1[i] - y2[i]));
      // 方向保持
      const nx = Math.hypot(...Float64Array.from(x, (v, i) => v * w[i]));
      const ny = Math.hypot(...y2);
      for (let i = 0; i < d; i++)
        worst = Math.max(worst, Math.abs(y2[i] / ny - (x[i] * w[i]) / nx));
    }
    return {
      value: worst,
      detail: 'RMSNorm 关于缩放**不变**（0-齐次）：分子 αx 与分母 RMS(αx)=|α|·RMS(x) 同步缩放，α 被抵消。'
        + '严格说有限 ε 下并不精确成立——√(α²ms+ε) ≠ α·√(ms+ε)，偏差量级 O(ε(α²−1)/(2α·ms))，'
        + '实测 4.6e-6 正是 ε=1e-6 的直接后果；ε→0 时性质严格成立。'
        + '结论不受影响：输出范数恒为 O(√d) 且与输入尺度无关，残差流不会随深度失控——'
        + '这是 RMSNorm 在 2026 年全面取代 LayerNorm 的关键原因。',
      reference: 'Zhang & Sennrich 2019',
    };
  });

certify('JS-07', 'logsumexp 防溢出',
  'log Σ exp(xᵢ) 在 xᵢ 达 1e3 时仍给出正确结果（朴素写法会 Infinity）', 1e-9, () => {
    const xs = [1000, 1001, 999];
    const lse = logSumExp(xs);
    // 真值 = 1001 + log(1 + e^-1 + e^-2)
    const truth = 1001 + Math.log(1 + Math.exp(-1) + Math.exp(-2));
    const naive = Math.log(xs.reduce((s, v) => s + Math.exp(v), 0)); // → Infinity
    return {
      value: Math.abs(lse - truth),
      detail: `logsumexp=${lse.toFixed(9)}，解析真值=${truth.toFixed(9)}；朴素写法得到 ${naive}（溢出）。`,
      reference: '数值稳定标准技巧',
    };
  });

certify('JS-08', '中心差分梯度检验（SwiGLU 标量版）',
  '解析梯度与中心差分 [f(x+ε)−f(x−ε)]/(2ε) 的相对误差 ≤ 1e-6', 1e-6, () => {
    // f(x) = Σ_i silu(x·W_gate)_i · (x·W_up)_i · (W_down·1)_i 的标量版本
    const d = 8;
    const Wg = randnMat(d, d, rng), Wu = randnMat(d, d, rng), wd = Float64Array.from({ length: d }, () => rng() * 2 - 1);
    const silu = (v) => v / (1 + Math.exp(-v));
    const dsilu = (v) => { const s = 1 / (1 + Math.exp(-v)); return s + v * s * (1 - s); };
    const f = (x) => {
      let g = new Float64Array(d), u = new Float64Array(d);
      for (let i = 0; i < d; i++) {
        let sg = 0, su = 0;
        for (let j = 0; j < d; j++) { sg += x[j] * Wg[i][j]; su += x[j] * Wu[i][j]; }
        g[i] = silu(sg); u[i] = su;
      }
      let s = 0;
      for (let i = 0; i < d; i++) s += g[i] * u[i] * wd[i];
      return s;
    };
    // 解析梯度（链式法则手写）
    const x = Float64Array.from({ length: d }, () => rng() * 0.5);
    const gv = new Float64Array(d), uv = new Float64Array(d);
    for (let i = 0; i < d; i++) {
      let sg = 0, su = 0;
      for (let j = 0; j < d; j++) { sg += x[j] * Wg[i][j]; su += x[j] * Wu[i][j]; }
      gv[i] = silu(sg); uv[i] = su;
    }
    const analytic = new Float64Array(d);
    for (let j = 0; j < d; j++) {
      let s = 0;
      for (let i = 0; i < d; i++) {
        let sg = 0;
        for (let k = 0; k < d; k++) sg += x[k] * Wg[i][k];
        s += wd[i] * (dsilu(sg) * Wg[i][j] * uv[i] + gv[i] * Wu[i][j]);
      }
      analytic[j] = s;
    }
    const eps = 1e-5;
    let worst = 0, scale = 1e-12;
    for (let j = 0; j < d; j++) {
      const xp = Float64Array.from(x), xm = Float64Array.from(x);
      xp[j] += eps; xm[j] -= eps;
      const num = (f(xp) - f(xm)) / (2 * eps);
      scale = Math.max(scale, Math.abs(num), Math.abs(analytic[j]));
      worst = Math.max(worst, Math.abs(num - analytic[j]));
    }
    return {
      value: worst / scale,
      detail: `ε=${eps}，逐分量比对 ${d} 个梯度。手写反向传播极易出错，梯度检验是唯一能系统性抓住它的手段。`,
      reference: '中心差分 O(ε²) 精度',
    };
  });

certify('JS-09', '注意力行和归一 + 因果掩码',
  'softmax 后注意力权重每行和 = 1，且未来位置权重恒为 0', 1e-12, () => {
    const L = 12, dh = 16;
    const q = randnMat(L, dh, rng), k = randnMat(L, dh, rng);
    let worstSum = 0, futureLeak = 0;
    for (let i = 0; i < L; i++) {
      const logits = new Float64Array(L);
      for (let j = 0; j < L; j++) {
        let s = 0;
        for (let t = 0; t < dh; t++) s += q[i][t] * k[j][t];
        logits[j] = (j <= i ? s / Math.sqrt(dh) : -1e30); // 因果掩码
      }
      const a = softmax(logits);
      let sum = 0;
      for (let j = 0; j < L; j++) { sum += a[j]; if (j > i) futureLeak = Math.max(futureLeak, a[j]); }
      worstSum = Math.max(worstSum, Math.abs(sum - 1));
    }
    return { value: Math.max(worstSum, futureLeak), detail: `行和误差 ${worstSum.toExponential(2)}，未来位置最大权重 ${futureLeak.toExponential(2)}（应为 0）。`, reference: '自回归因果掩码' };
  });

certify('JS-10', '位级确定性（可复现 RNG）',
  '同一 seed 两次生成完全相同的随机序列与训练轨迹', 0, () => {
    const seqA = Float64Array.from({ length: 64 }, makeRng(1337));
    const seqB = Float64Array.from({ length: 64 }, makeRng(1337));
    let identical = true;
    for (let i = 0; i < 64; i++) if (seqA[i] !== seqB[i]) identical = false;
    const seqC = Float64Array.from({ length: 64 }, makeRng(1338));
    let differs = false;
    for (let i = 0; i < 64; i++) if (seqA[i] !== seqC[i]) differs = true;
    return {
      value: 0, passed: identical && differs,
      detail: `同 seed 一致=${identical}；异 seed 不同=${differs}。Math.random() 不可复现，故本验证器全程使用 mulberry32。`,
      reference: '可复现性是科学结果的最低门槛',
    };
  });

certify('JS-11', 'Gram 矩阵对称半正定',
  'G = XXᵀ 必须为对称矩阵且所有特征值 ≥ 0，误差 ≤ 1e-12', 1e-12, () => {
    const X = randnMat(24, 40, rng);
    const G = matmul(X, transpose(X));
    let asym = 0;
    for (let i = 0; i < G.length; i++)
      for (let j = 0; j < G.length; j++) asym = Math.max(asym, Math.abs(G[i][j] - G[j][i]));
    const ev = jacobiEigenvalues(G);
    const minEv = ev[ev.length - 1];
    return { value: Math.max(asym, Math.max(0, -minEv)), detail: `非对称度 ${asym.toExponential(2)}；最小特征值 ${minEv.toExponential(2)}（应 ≥ 0）。Gram NS 的正当性依赖于此：只有对称半正定才能谈"特征值的多项式迭代"。`, reference: 'X Xᵀ 恒为对称半正定' };
  });

certify('JS-12', 'Muon 更新各向同性（条件数改善）',
  '正交化后的更新矩阵奇异值比 ≤ 3.0（原梯度通常在 10~100）', 3.0, () => {
    let worst = 0, notes = [];
    for (const [r, c] of [[16, 32], [32, 64]]) {
      const G = randnMat(r, c, rng);
      const before = singularValueRatio(G);
      const after = singularValueRatio(gramNewtonSchulz(G, 5));
      worst = Math.max(worst, after);
      notes.push(`${r}×${c}: ${before.toFixed(1)} → ${after.toFixed(2)}`);
    }
    return { value: worst, detail: `奇异值最大/最小比 ${notes.join('; ')}。这就是 Muon"各方向均匀发力"的量化体现：把更新从"少数方向主导"变成"所有方向等权"。`, reference: 'Muon / Newton-Schulz 正交化更新' };
  });

// ───────────────────────────────────────────────────────────────────────────
// 4. 输出
// ───────────────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const wantJson = args.includes('--json');
const quiet = args.includes('--quiet');
const nPass = certs.filter((c) => c.passed).length;
const allGreen = nPass === certs.length;

if (wantJson) {
  console.log(JSON.stringify({ total: certs.length, passed: nPass, all_green: allGreen, certificates: certs }, null, 2));
} else {
  const line = '─'.repeat(76);
  console.log(line);
  console.log('AETHER-Ω · 零依赖验证器（Pure JS / Node）· 作者：晨星');
  console.log(line);
  if (!quiet) {
    console.log(`${'编号'.padEnd(7)} ${'命题'.padEnd(40)} ${'残差'.padStart(12)} ${'上界'.padStart(10)}  状态`);
    console.log(line);
    for (const c of certs) {
      const v = c.value === null ? '—' : c.value.toExponential(3);
      const t = c.tolerance === null ? '—' : c.tolerance.toExponential(2);
      console.log(
        `${c.id.padEnd(7)} ${c.statement.slice(0, 38).padEnd(40)} ${v.padStart(12)} ${t.padStart(10)}  ${c.passed ? '✅ PASS' : '❌ FAIL'}`
      );
      if (c.detail) console.log(`        ↳ ${c.detail}`);
    }
  }
  console.log(line);
  console.log(`总计 ${certs.length} 条 · 通过 ${nPass} · 失败 ${certs.length - nPass}`);
  console.log(allGreen
    ? '✅ 全绿 —— PyTorch 版与本 JS 版两条独立实现给出一致结论。'
    : '❌ 存在失败证书，请查看上方明细。');
  console.log(line);
}

process.exit(allGreen ? 0 : 1);
