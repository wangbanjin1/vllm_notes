---
series: 1Cat-vLLM 源码精读中文讲义 · 知识点专题深潜
title: 专题讲义：注意力机制演进全解（MHA → MQA → GQA → MLA）
subtitle: 从 KV Cache 显存墙第一性原理、GQA 算子宽矩阵打包到 DeepSeek MLA 低秩矩阵吸收的代数统一视角
authors: 五道口纳什风格 & Antigravity
date: 2026-10-23
repo: https://github.com/1CatAI/1Cat-vLLM
hardware: 通用 GPU 内存层次 (HBM / SRAM / Tensor Core) & Tesla V100 (SM70) 算子协同
source_files: csrc/attention/sm70_v37/prefill.cu · csrc/attention/sm70_v37/tail.cu
topics: KV Cache 容量公式与算例 · MHA/MQA/GQA 子空间折衷 · DeepSeek MLA 低秩压缩与矩阵吸收 · 解耦 RoPE 数学证明
---

## 1 导论：为什么过去七年注意力机制一直在改写 KV 结构？

如果回顾从 2017 年原始 Transformer 到如今 LLaMA-3、Qwen-2.5、DeepSeek-V3/R1 的架构演进，你会发现一个非常耐人寻味的现象：前馈网络（FFN/MoE）一直在想方设法把参数量做大，而自注意力模块（Self-Attention）却一直在想方设法把中间状态做小。

推动这一演进的根本物理矛盾只有一条：**在自回归解码（Decode）阶段，注意力计算受困于严重的 Memory-Bound（访存带宽受限）与 KV Cache 容量墙。**

- **带宽瓶颈**：每生成 1 个新 Token，GPU 只需要对当前 Token 做一次极窄的向量投影，却必须把历史上所有已生成 Token 的 Key 和 Value 从 HBM 全局显存完整搬运到片上 SRAM。算术强度 $I$ 低至 $1\text{ FLOP/Byte}$ 量级，远低于 V100 的 $138.9\text{ FLOPs/Byte}$ 或 A100 的 $156\text{ FLOPs/Byte}$ 拐点。
- **容量瓶颈**：随着上下文窗口从 4K 迈向 128K 乃至 1M，KV Cache 的显存体积随序列长度 $S$ 与并发批次 $B$ 呈严格线性膨胀，迅速榨干整卡显存并锁死系统最大并发量。

> [!IMPORTANT] 注意力架构演进的唯一主线
> 从 **MHA**（多头独立）到 **MQA**（全头共享单 KV）、再到 **GQA**（分组共享 KV）与 **MLA**（低秩潜向量压缩 + 推理期矩阵吸收），所有结构创新的本质都是在回答同一个问题：**如何在不牺牲（甚至增强）多头子空间表达能力的前提下，把每个 Token 落盘到 HBM 显存里的 KV 字节数压缩到极致？**

---

## 2 基础概念厘清：多头子空间投影与 MHA 显存灾难

在深入变体之前，我们先建立多头注意力（Multi-Head Attention, MHA）的严格代数记号。设输入序列隐藏状态为 $\mathbf{X} \in \mathbb{R}^{S \times d_{\text{model}}}$，共有 $n_h$ 个注意力头，每个头的特征维度为 $d_h$。标准设计下满足：

$$d_{\text{model}} = n_h \cdot d_h, \qquad \mathbf{Q}_i = \mathbf{X}\mathbf{W}_i^Q, \quad \mathbf{K}_i = \mathbf{X}\mathbf{W}_i^K, \quad \mathbf{V}_i = \mathbf{X}\mathbf{W}_i^V \quad (i \in \{1, \dots, n_h\})$$

符号解释如下：
1. $d_{\text{model}}$：模型隐藏层总宽度（如 LLaMA-3-8B 中 $d_{\text{model}} = 4096$）。
2. $n_h$：Query 注意力头总数（如 $n_h = 32$）。
3. $d_h$：单个注意力头的子空间维度（如 $d_h = 4096 / 32 = 128$；部分大模型采用 $d_h = 256$）。
4. $\mathbf{W}_i^Q, \mathbf{W}_i^K, \mathbf{W}_i^V \in \mathbb{R}^{d_{\text{model}} \times d_h}$：第 $i$ 个头专属的投影权重矩阵。

### 2.1 为什么要切成多个头，而不是直接算一个 $d_{\text{model}}$ 维的单头注意力？

单头注意力每行只能算出一组标量 Softmax 概率分布，相当于强迫模型用同一把尺子去混合所有语义特征；而多头机制将高维表征投影到 $n_h$ 个相互独立的低维子空间（Subspaces）中，使不同头能够并行捕捉局部语法搭配、长程指代消解、代码缩进对齐等正交特征。

### 2.2 MHA 的 KV Cache 显存账本：为什么 70B 模型撑不起高并发？

在标准 MHA 中，Query、Key、Value 的头数完全相等（$n_q = n_{kv} = n_h$）。当采用 FP16/BF16（每元素 $2\text{ Bytes}$）存储 KV Cache 时，一个包含 $L$ 层 Transformer、并发批次为 $B$、序列长度为 $S$ 的系统，其 KV Cache 总显存开销满足：

$$\text{Mem}_{\text{MHA}} = 2_{\text{(K与V)}} \times 2_{\text{(FP16字节)}} \times B \cdot S \cdot L \cdot n_h \cdot d_h = 4 \cdot B \cdot S \cdot L \cdot n_h \cdot d_h \;\text{Bytes}$$

符号解释如下：
1. 第一个因子 $2$：每一层必须同时缓存 $\mathbf{K}$ 和 $\mathbf{V}$ 两个张量。
2. 第二个因子 $2$：FP16 / BF16 每个浮点数占 $2$ 字节。
3. $B, S, L$：分别为并发请求数（Batch Size）、上下文长度（Sequence Length）与模型层数（Layers）。
4. $n_h \cdot d_h$：单层单 Token 的 KV 特征总宽度。

> [!WARNING] 一个触目惊心的工业算例：70B 模型在 MHA 下的显存爆炸
> 假设一个 70B 规模的模型采用标准 MHA 配置（$L = 80$ 层，$n_h = 64$ 头，$d_h = 128$）：
> - **单 Token 显存消耗**：$4 \times 80 \times 64 \times 128 = 2,621,440\text{ Bytes} \approx \mathbf{2.5\text{ MB / Token}}$；
> - **32 并发 × 8K 上下文总开销**：$32 \times 8192 \times 2.5\text{ MB} = \mathbf{640\text{ GB 显存}}$！
> 仅凭模型权重（FP16 约 140 GB）加 KV Cache（640 GB），就需要整整 **10 张 80GB A100** 才能跑起区区 32 个并发请求。这就是为什么现代大模型彻底抛弃了标准 MHA。

### 2.3 本章小结

MHA 赋予了模型丰富的多子空间表达力，但「每个 Query 头都配一套独立 Key/Value 头」的 $1:1$ 绑定设计，使得 KV Cache 容量和 Decode 访存带宽随头数 $n_h$ 线性成倍膨胀，成为长文本推理的头号瓶颈。

---

## 3 从 MQA 到 GQA：头间共享的折衷艺术与算子宽矩阵打包

既然 KV 头数过多是万恶之源，最直接的削减方案就是**让多个 Query 头共享同一组 Key 和 Value 头**。

### 3.1 MQA（多查询注意力）与 GQA（分组查询注意力）的形式化对比

1. **MQA（Multi-Query Attention，Shazeer 2019）**：走极端路线，保留 $n_q = n_h$ 个 Query 头，但全层**只保留 $n_{kv} = 1$ 个 Key 头和 $1$ 个 Value 头**。所有 $n_h$ 个 Query 头全部与这唯一的一组 $(\mathbf{K}, \mathbf{V})$ 做注意力交互。虽然 KV Cache 瞬降为原来的 $1/n_h$（如 $1/32$），但全网被迫在单一 KV 子空间内检索，长文本复杂推理精度出现明显退化。
2. **GQA（Grouped-Query Attention，Ainslie et al. 2023）**：在 MHA 与 MQA 之间取最优工程折中。将 $n_q$ 个 Query 头均分为 $n_{kv}$ 个组（Group），每组内部包含 $G = n_q / n_{kv}$ 个 Query 头，**同组内的 $G$ 个 Query 头共享 1 套 $(\mathbf{K}_g, \mathbf{V}_g)$**：

$$\mathbf{O}_{g, r} = \text{Softmax}\!\left(\frac{\mathbf{Q}_{g, r} \mathbf{K}_g^\top}{\sqrt{d_h}}\right)\mathbf{V}_g, \qquad g \in \{1, \dots, n_{kv}\}, \quad r \in \{1, \dots, G\}$$

符号解释如下：
1. $n_{kv}$：Key/Value 的头数（即分组数，典型值为 $4$ 或 $8$）。
2. $G = n_q / n_{kv}$：每组共享同一份 KV 的 Query 头数（如 LLaMA-3-8B 中 $32 / 8 = 4$；`1Cat-vLLM` SM70 v37 针对的 GQA 模型中 $G = 6$）。
3. $\mathbf{Q}_{g, r} \in \mathbb{R}^{M \times d_h}$：第 $g$ 组内第 $r$ 个 Query 子头。
4. $\mathbf{K}_g, \mathbf{V}_g \in \mathbb{R}^{N \times d_h}$：第 $g$ 组共享的唯一 Key 与 Value 矩阵。

### 3.2 GQA 赋予算子层的神级红利：GQA-Packed Wide GEMM

在很多框架中，GQA 仅仅被当作“省显存的模型配置”，算子执行时依然把 $\mathbf{K}_g, \mathbf{V}_g$ 广播（Broadcast/Repeat）成 $n_q$ 份去跑普通 MHA Kernel。这种做法白白浪费了 GQA 最宝贵的代数性质！

观察上面的公式：既然同组内的 $\mathbf{Q}_{g,1}, \dots, \mathbf{Q}_{g,G}$ 乘的都是**完全相同的 $\mathbf{K}_g^\top$ 和 $\mathbf{V}_g$**，根据分块矩阵乘法法则，我们可以把这 $G$ 个 Query 头沿行轴（Row Axis）上下拼接成一个高度为 $G \cdot M$ 的宽矩阵：

$$\begin{bmatrix} \mathbf{Q}_{g,1} \\ \vdots \\ \mathbf{Q}_{g,G} \end{bmatrix}_{(G \cdot M) \times d_h} \mathbf{K}_g^\top = \begin{bmatrix} \mathbf{S}_{g,1} \\ \vdots \\ \mathbf{S}_{g,G} \end{bmatrix}_{(G \cdot M) \times N}$$

符号解释如下：
1. $[\mathbf{Q}_{g,1}; \dots; \mathbf{Q}_{g,G}] \in \mathbb{R}^{(G \cdot M) \times d_h}$：沿行维度打包后的组合 Query 矩阵（在 `1Cat-vLLM` 的 `prefill.cu` 中即 `kRows = kQuery * kHeadsQ`）。
2. $\mathbf{K}_g^\top \in \mathbb{R}^{d_h \times N}$：无需任何复制广播、仅从 HBM 讀取 1 次的共享 Key 转置矩阵。
3. $[\mathbf{S}_{g,1}; \dots; \mathbf{S}_{g,G}] \in \mathbb{R}^{(G \cdot M) \times N}$：单次宽 GEMM 直接产出的 $G$ 头拼接打分矩阵。

![GQA-Packed 宽矩阵打包算子与 MLA 低秩矩阵吸收机制对比|源码定位：csrc/attention/sm70_v37/prefill.cu:L455--L466 & DeepSeek-V2/V3 MLA 架构规范](assets/fig04_gqa_wide_gemm_and_mla.svg)

> [!KNOWLEDGE] 为什么 GQA 行打包在 Decode 与小 Batch 下尤为致命？
> 在 Decode 阶段（$M=1$）或短序列 Prefill 下，如果单头独立启动 GEMM，矩阵左操作数只有 $1$ 行（$1 \times d_h$），实质退化为矩阵-向量乘（GEMV），Tensor Core 的 $8 \times 8 \times 4$（Volta）或 $16 \times 8 \times 16$（Ampere）硬件矩阵单元有 $87.5\% \sim 93.75\%$ 的行通道处于空置状态！而通过 **GQA-Packed** 将 $G=6$ 或 $G=8$ 个 Query 头叠成 $(G \times d_h)$ 矩阵，直接把 GEMV 转化为真正的 GEMM，**在不增加任何 1 字节 HBM 访存的前提下，让 Tensor Core 的有效算术强度瞬间翻升 $G$ 倍！**

### 3.3 本章小结

GQA 之所以成为当今开源大模型的绝对工业标准，不仅因为它将 KV Cache 显存削减为 $1/G$ 且精度几乎无损，更因为它在算子层天然支持 **Row-Axis Query Packing**，将多次低效窄矩阵访存折叠为单次高吞吐宽矩阵乘。

---

## 4 DeepSeek MLA（多头潜在注意力）：低秩压缩与矩阵吸收的艺术

尽管 GQA 取得了巨大成功，DeepSeek 团队在设计 DeepSeek-V2 / V3（以及 Kimi K3 Gated MLA 等演进架构）时敏锐地提出了一个更深层的问题：**GQA 本质上依然是“物理砍掉 $n_{kv}$ 的头数”，能不能在计算时恢复全部 $n_h$ 个独立 KV 头的完整高秩表达空间，而在存入 HBM 显存时却比 GQA 压得更小？**

这就是 **MLA（Multi-Head Latent Attention）** 的核心动机。它的破局由环环相扣的三步组成：**低秩联合压缩 → 推理期矩阵结合律吸收 → 解耦旋转位置编码（Decoupled RoPE）**。

### 4.1 机制一：KV 低秩联合压缩（Low-Rank Joint Compression）

MLA 不再直接把层输入 $\mathbf{h}_t \in \mathbb{R}^d$ 投影为庞大的多头 $\mathbf{K}_t, \mathbf{V}_t$，而是先通过一个下投影矩阵 $\mathbf{W}^{DKV} \in \mathbb{R}^{d_c \times d}$ 将其压缩为一个极短的**低维潜向量（Latent Vector）** $\mathbf{c}_t^{KV}$，在需要计算时再通过上投影矩阵 $\mathbf{W}^{UK}, \mathbf{W}^{UV}$ 恢复出各个头的 Key 与 Value：

$$\mathbf{c}_t^{KV} = \mathbf{W}^{DKV} \mathbf{h}_t \in \mathbb{R}^{d_c}, \qquad \mathbf{k}_{t, i}^C = \mathbf{W}_i^{UK} \mathbf{c}_t^{KV} \in \mathbb{R}^{d_h}, \qquad \mathbf{v}_{t, i}^C = \mathbf{W}_i^{UV} \mathbf{c}_t^{KV} \in \mathbb{R}^{d_v}$$

符号解释如下：
1. $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$：第 $t$ 个 Token 的 KV 联合低秩潜向量（典型值 $d_c = 512$，远小于多头展开总宽 $n_h \cdot d_h = 128 \times 128 = 16384$）。
2. $\mathbf{W}_i^{UK} \in \mathbb{R}^{d_h \times d_c}$：第 $i$ 个注意力头专属的 Key 上投影解压矩阵。
3. $\mathbf{W}_i^{UV} \in \mathbb{R}^{d_v \times d_c}$：第 $i$ 个注意力头专属的 Value 上投影解压矩阵。
4. $\mathbf{k}_{t, i}^C, \mathbf{v}_{t, i}^C$：解压还原出的第 $i$ 头内容键（Content Key）与内容值（Content Value）。

**在推理期间，HBM 的 KV Cache 中彻底不再存储展开后的 $\mathbf{k}_{t,i}^C$ 和 $\mathbf{v}_{t,i}^C$，每个 Token 每层只存这一个 $d_c = 512$ 维的潜向量 $\mathbf{c}_t^{KV}$（外加 64 维解耦位置键）！**

### 4.2 机制二：推理期矩阵吸收（Weight Absorbing）——为什么不需要把 $\mathbf{c}_t^{KV}$ 解压出来？

看到上面的公式，系统工程师立刻会追问一个致命问题：**如果每次 Decode 计算注意力时，都要先把历史所有 Token 的 $\mathbf{c}_t^{KV}$ 乘以 $\mathbf{W}_i^{UK}$ 重新解压成 128 个头的 $\mathbf{k}_{t,i}^C$，那片上 SRAM 和计算量岂不是瞬间爆炸？**

MLA 最惊艳的一笔就在这里——利用线性代数的**矩阵乘法结合律**，把静态解压权重 $\mathbf{W}_i^{UK}$ 直接“吸收”到当前 Token 的 Query 向量上！

我们将 $\mathbf{k}_{t,i}^C = \mathbf{W}_i^{UK} \mathbf{c}_t^{KV}$ 代入第 $i$ 个头的注意力内积打分项：

$$\mathbf{q}_{t, i}^\top \mathbf{k}_{s, i}^C = \mathbf{q}_{t, i}^\top \left(\mathbf{W}_i^{UK} \mathbf{c}_s^{KV}\right) = \left(\mathbf{q}_{t, i}^\top \mathbf{W}_i^{UK}\right) \mathbf{c}_s^{KV} = \tilde{\mathbf{q}}_{t, i}^\top \mathbf{c}_s^{KV}$$

同理，对于注意力加权输出与最终输出投影 $\mathbf{W}_i^O \in \mathbb{R}^{d \times d_v}$ 的乘积，也可以将 $\mathbf{W}_i^{UV}$ 吸收到输出端：

$$\mathbf{u}_t = \sum_{i=1}^{n_h} \mathbf{W}_i^O \sum_{s \le t} \alpha_{t, s, i} \left(\mathbf{W}_i^{UV} \mathbf{c}_s^{KV}\right) = \sum_{i=1}^{n_h} \left(\mathbf{W}_i^O \mathbf{W}_i^{UV}\right) \sum_{s \le t} \alpha_{t, s, i} \mathbf{c}_s^{KV}$$

符号解释如下：
1. $\tilde{\mathbf{q}}_{t, i}^\top = \mathbf{q}_{t, i}^\top \mathbf{W}_i^{UK} \in \mathbb{R}^{1 \times d_c}$：吸收了 Key 上投影矩阵后的 $d_c$ 维新查询向量。
2. $\alpha_{t, s, i} \in [0, 1]$：第 $i$ 个头在位置 $t$ 对历史位置 $s$ 的 Softmax 注意力概率权重。
3. $\sum_{s \le t} \alpha_{t, s, i} \mathbf{c}_s^{KV} \in \mathbb{R}^{d_c}$：直接在 $d_c = 512$ 维潜空间内对共享缓存 $\mathbf{c}_s^{KV}$ 完成加权求和，随后再通过吸收矩阵 $\mathbf{W}_i^O \mathbf{W}_i^{UV}$ 映射回模型隐藏维。

> [!IMPORTANT] 矩阵吸收（Weight Absorbing）的系统级意义
> 经过结合律重排后，**历史序列长度 $S$ 所在的大循环里，彻底消失了 $\mathbf{W}_i^{UK}$ 和 $\mathbf{W}_i^{UV}$ 的解压运算！** 在 Decode 阶段，所有 $n_h$ 个吸收后的查询头 $\tilde{\mathbf{q}}_{t,1}, \dots, \tilde{\mathbf{q}}_{t,n_h}$ 全部直接与显存中**唯一一份** $\mathbf{c}_s^{KV} \in \mathbb{R}^{512}$ 做矩阵乘——这意味着 MLA 在推理期自动转化为了一个特征维为 $d_c = 512$ 的 **MQA（全头共享单潜向量）** 算子，可享受极致的宽矩阵打包加速！

### 4.3 机制三：为什么标准 RoPE 会破坏矩阵吸收？（解耦 RoPE 的必然性）

为什么不能直接在 $\mathbf{k}_{s,i}^C$ 上施加标准的旋转位置编码（RoPE）？我们写出带位置旋转矩阵 $\mathcal{R}_t, \mathcal{R}_s \in \mathbb{R}^{d_h \times d_h}$ 的内积表达式就能一眼看穿：

$$\left(\mathcal{R}_t \mathbf{q}_{t,i}\right)^\top \left(\mathcal{R}_s \mathbf{W}_i^{UK} \mathbf{c}_s^{KV}\right) = \mathbf{q}_{t,i}^\top \underbrace{\left(\mathcal{R}_t^\top \mathcal{R}_s\right)}_{\mathcal{R}_{s - t}} \mathbf{W}_i^{UK} \mathbf{c}_s^{KV} = \mathbf{q}_{t,i}^\top \mathcal{R}_{s - t} \mathbf{W}_i^{UK} \mathbf{c}_s^{KV}$$

符号解释如下：
1. $\mathcal{R}_{s - t} = \mathcal{R}_t^\top \mathcal{R}_s$：正交旋转矩阵夹在 $\mathbf{q}_{t,i}^\top$ 与 $\mathbf{W}_i^{UK}$ 之间，且随每一个历史位置 $s$ 动态变化。

> [!WARNING] 阻断矩阵吸收的罪魁祸首：依赖历史位置 $s$ 的旋转矩阵 $\mathcal{R}_{s-t}$
> 因为中间横插了一个随历史 Token 位置 $s$ 变化的矩阵 $\mathcal{R}_{s-t}$，且矩阵乘法不满足交换律（$\mathcal{R}_{s-t} \mathbf{W}_i^{UK} \ne \mathbf{W}_i^{UK} \mathcal{R}_{s-t}$），$\mathbf{q}_{t,i}^\top$ 与静态权重 $\mathbf{W}_i^{UK}$ 被生生隔开，**无法在扫描历史 KV 之前预先合并**！如果强行对所有历史位置 $s$ 实时重算 $\mathcal{R}_{s-t} \mathbf{W}_i^{UK}$，推理复杂度将直接退化。

为了保住矩阵吸收，MLA 提出了极其优雅的 **解耦 RoPE（Decoupled RoPE）**（见图 4 `(b)`）：
1. **内容通路（NoPE，占比 $512$ 维）**：$\mathbf{c}_s^{KV}$ 完全不加位置编码，畅通无阻地执行矩阵吸收 $\tilde{\mathbf{q}}_{t,i}^\top \mathbf{c}_s^{KV}$；
2. **位置通路（旁路单头 MQA，占比 $d_R = 64$ 维）**：单独切出一个所有头共享的极窄位置键 $\mathbf{k}_s^R = \mathcal{R}_s (\mathbf{W}^{KR} \mathbf{h}_s) \in \mathbb{R}^{64}$，与各头的旋转位置查询 $\mathbf{q}_{t,i}^R = \mathcal{R}_t (\mathbf{W}_i^{QR} \mathbf{h}_t) \in \mathbb{R}^{64}$ 计算位置相关性，最后将两路打分直接相加：

$$\text{Score}_{t, s, i} = \frac{\tilde{\mathbf{q}}_{t, i}^\top \mathbf{c}_s^{KV} + (\mathbf{q}_{t, i}^R)^\top \mathbf{k}_s^R}{\sqrt{d_h + d_R}}$$

这样一来，每个 Token 落盘到 KV Cache 的总宽度仅为 $d_c + d_R = 512 + 64 = 576$（对比展开后的 $K, V$ 动辄 $30,720$ 维，压缩率超过 $53\times$）！

### 4.4 本章小结

MLA 并不是简单地加了一个 AutoEncoder 瓶颈层，而是通过「低秩潜向量缓存 + 结合律矩阵吸收 + 解耦旁路 RoPE」三位一体的代数设计，让模型在**训练与表征上等价于多头 MHA**，而在**推理访存与算子执行上等价于超窄 MQA**。

---

## 5 四大注意力机制横向对比与总结延伸

### 5.1 全景对比账本

| 维度与指标 | MHA (标准多头) | MQA (多查询) | GQA (分组查询) | MLA (多头潜在注意力) |
| :--- | :--- | :--- | :--- | :--- |
| **Q : KV 拓扑关系** | $n_h : n_h$ ($1:1$ 独立) | $n_h : 1$ (全头共享单 KV) | $n_h : n_{kv}$ ($G:1$ 分组共享) | 表征为 $n_h$ 头，缓存为共享潜向量 |
| **单 Token 每层缓存元素数** | $2 n_h d_h$（基线 **100%**） | $2 d_h$（约 **3.1%**） | $2 n_{kv} d_h$（约 **12.5% ~ 16.7%**） | $d_c + d_R = 576$（约 **1.8% ~ 9%**） |
| **KV Cache 物理存储对象** | 展开的多头 $\mathbf{K}, \mathbf{V}$ | 单头 $\mathbf{K}, \mathbf{V}$ | $n_{kv}$ 组 $\mathbf{K}_g, \mathbf{V}_g$ | 低秩潜向量 $\mathbf{c}_t^{KV}$ + 共享 $\mathbf{k}_t^R$ |
| **推理期矩阵吸收** | 不支持 | 不支持 | 不支持 | **支持（$\mathbf{W}_i^{UK}, \mathbf{W}_i^{UV}$ 吸收至两端）** |
| **算子层宽矩阵加速方式** | 仅靠序列维 $M$ 分块 | 全头 $n_h \cdot M$ 打包 | **同组 $G \cdot M$ 行打包 (`prefill.cu`)** | Decode 期全头 $\tilde{\mathbf{Q}}$ 拼成宽矩阵乘 $\mathbf{c}^{KV}$ |
| **代表性工业模型** | GPT-3, LLaMA-1 | PaLM, Falcon | LLaMA-2/3, Qwen-2.5, Mistral | DeepSeek-V2/V3/R1, Kimi K3 |

### 5.2 压缩总结与核心启发

> [!IMPORTANT] 带走本专题的三条核心判断
> 1. **看注意力架构先算 KV Cache 字节数与算术强度**：任何脱离 HBM 带宽和 Roofline 拐点谈注意力改结构的讨论都是空中楼阁。
> 2. **GQA 的精髓在「算法-算子协同」**：模型层面的分组共享直接催生了算子层面的 `kRows = G * Q` 宽矩阵打包，在 Volta SM70 等硬件上实现了带宽减负与 Tensor Core 满载的双赢。
> 3. **MLA 的精髓在「交换计算顺序」**：利用矩阵结合律把随历史长度 $O(S)$ 重复执行的解压投影转移到 $O(1)$ 的当前 Token 端，并用解耦 RoPE 绕开非交换旋转矩阵的阻断。

### 5.3 拓展阅读

1. **GQA 原论文**：Joshua Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (EMNLP 2023)
2. **DeepSeek-V2 / V3 技术报告（MLA 原始文献）**：DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* (2024)
3. **1Cat-vLLM GQA-Packed 算子实现**：`csrc/attention/sm70_v37/prefill.cu` 与 `tail.cu`
