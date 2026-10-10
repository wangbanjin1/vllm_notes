# 知识点汇总：注意力机制深度全解（从 MHA、MQA、GQA 到 MLA 与 KV Cache 演进）

> **栏目**：知识点汇总 · 注意力机制全景精读  
> **核心主题**：多头注意力概念厘清、KV Cache 显存墙瓶颈、MHA / MQA / GQA / MLA 架构演进与数学推导、DeepSeek 矩阵吸收工程实践

---

## 一、为什么注意力机制一直在变？（核心矛盾：KV Cache 显存墙）

在大模型推理的自回归生成阶段（Token-by-Token Decode 阶段），硬件的计算模式存在一个根本性的物理瓶颈：**Memory-Bound（访存受限）**。

### 1. 显存墙与 Roofline 模型
在生成每一个新 Token 时，GPU 只需要对当前这 1 个 Token 计算向量投影，计算量（FLOPs）极小；但为了计算自注意力，GPU **必须把历史上所有已经生成过的 Token 的 Key 和 Value 矩阵从全局显存（HBM / DRAM）完整搬运到片上 SRAM**。

* **算力与访存的失衡**：以 NVIDIA Tesla V100 为例，其半精度 Tensor Core 算力高达 **125 TFLOP/s**，而全局显存带宽仅有 **900 GB/s**。
* **算术强度（Operational Intensity）极低**：在 Decode 阶段，单步生成的计算访存比往往小于 1 FLOP/Byte，使得 GPU 的超大规模计算单元（Tensor Core）绝大部分时间都在**饥饿等待显存数据搬运**。

### 2. 长序列下的 KV Cache 爆炸
随着上下文窗口从 4K 扩展到 32K、128K 乃至 1M，KV Cache 占用的显存空间随序列长度 $L$ 呈**严格线性增长**。在并发 Batch Size 增加时，KV Cache 会极其迅速地吞噬掉所有 GPU 显存，导致并发承载能力断崖式下跌。

> [!IMPORTANT]
> **注意力机制的演进主线**：
> 从 2017 年的原始 Transformer 到如今最新的 DeepSeek-V3，注意力机制结构演化的核心驱动力只有一个——**在尽可能保全模型表达能力的前提下，极度压缩 KV Cache 的显存容量与每次自回归解码的访存带宽！**

---

## 二、基础概念厘清：什么是“头”与“头维度”？

很多初学者容易混淆多头注意力（MHA）中的各个维度概念，我们在此建立严格的数学定义。

```text
输入张量 X (维度: hidden_size / d_model)
                   │
    ┌──────────────┼──────────────┐
    ▼ 投影为 Q     ▼ 投影为 K     ▼ 投影为 V
 [num_heads,   [num_heads,   [num_heads,
   head_dim]     head_dim]     head_dim]
```

### 1. 维度对应关系
设大模型的隐藏层维度为 $d_{\text{model}}$（亦称 `hidden_size`），注意力头总数为 $n_{\text{heads}}$，则每一个“头”（Head）的特征维度 $d_{\text{head}}$ 满足：

$$d_{\text{head}} = \frac{d_{\text{model}}}{n_{\text{heads}}}$$

以典型主流模型为例：
* **LLaMA-2-7B**：$d_{\text{model}} = 4096$，$n_{\text{heads}} = 32$，则单个头的维度为：
  $$d_{\text{head}} = \frac{4096}{32} = 128$$
* **LLaMA-3-8B**：同样为 $d_{\text{model}} = 4096$，$n_{\text{heads}} = 32$，$d_{\text{head}} = 128$。
* **部分大尺寸或多模态模型**：如 Qwen-2.5 某些配置或特殊多头设计，可能采用 $d_{\text{head}} = 64$ 或 $d_{\text{head}} = 256$。

### 2. 为什么需要划分为“多头”？
单头注意力只能把所有上下文信息压缩到一个统一的注意力权重分布上；而多头注意力将高维向量投影到多个独立的**低维特征子空间（Subspaces）**中：
* Head 0 可能专注于捕捉**局部语法短语**结构；
* Head 1 可能专注于追踪**长程代词指代**关系；
* Head 2 可能专注于识别**标点符号与句子边界**。

多头机制赋予了模型同时关注来自不同子空间中不同位置信息的能力。

---

## 三、四大注意力机制演进全景

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 1. MHA (Multi-Head Attention): 1 对 1 独立头绑定                      │
 │    Q0 ──> K0, V0    Q1 ──> K1, V1    ...    Q(N-1) ──> K(N-1), V(N-1) │
 └────────────────────────────────────────────────────────────────────────┘
                                     │
                    极度追求显存削减：所有 Q 共享 1 个 KV
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 2. MQA (Multi-Query Attention): N 对 1 极限共享                       │
 │    Q0, Q1, Q2, ..., Q(N-1) ───────────────> K, V (全局仅 1 组 KV)      │
 └────────────────────────────────────────────────────────────────────────┘
                                     │
                    工程折中妥协：平衡表达精度与显存
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 3. GQA (Grouped-Query Attention): 分组折中共享 (工业主流，如 8:1)     │
 │    Group 0: Q0 ~ Q7 ──> K0, V0                                         │
 │    Group 1: Q8 ~ Q15 ──> K1, V1                                        │
 └────────────────────────────────────────────────────────────────────────┘
                                     │
                    突破头数删减：低秩联合投影 + 矩阵吸收 (DeepSeek 创新)
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 4. MLA (Multi-Head Latent Attention): 低秩隐空间联合压缩               │
 │    KV Cache 只存一个低维隐向量 c_t (如 512维)                          │
 │    计算时利用结合律直接将解码矩阵吸收进 Q，兼具 MHA 表达力与极限压缩   │
 └────────────────────────────────────────────────────────────────────────┘
```

---

### 1. MHA (Multi-Head Attention，标准多头注意力)

#### 结构定义
在经典 Transformer 中，$Q, K, V$ 拥有完全相同数量的头数：
$$n_q = n_k = n_v = N$$

每个头独立拥有各自的投影权重矩阵 $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{head}}}$：
$$\text{head}_i = \text{Softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d_{\text{head}}}}\right) V_i$$

#### KV Cache 显存开销公式
对于一个具有 $L_{\text{layer}}$ 层 Transformer、总头数为 $n_{\text{heads}}$、头维度为 $d_{\text{head}}$ 的模型，单个 Token 采用 FP16（每个元素 2 字节）存储时，每一层需要保存 $K$ 和 $V$ 两个张量：

$$\text{单 Token 显存占用} = 2 \times n_{\text{heads}} \times d_{\text{head}} \times 2 \text{ Bytes} = 4 \times n_{\text{heads}} \times d_{\text{head}} \text{ Bytes/层}$$

对于序列长度为 $S$、并发 Batch Size 为 $B$ 的请求，总 KV Cache 显存占用为：
$$\text{Total KV Cache}_{\text{MHA}} = 4 \cdot B \cdot S \cdot L_{\text{layer}} \cdot n_{\text{heads}} \cdot d_{\text{head}} \text{ Bytes}$$

> **算例**：以 70B 模型（$L_{\text{layer}}=80, n_{\text{heads}}=64, d_{\text{head}}=128$）为例：
> 单个 Token 在单层占用 $4 \times 64 \times 128 = 32 \text{ KB}$；80 层总计 **2.56 MB / Token**！  
> 若并发处理 32 个长度为 8K 的请求，仅 KV Cache 就需要：
> $$32 \times 8192 \times 2.56 \text{ MB} \approx \mathbf{671 \text{ GB 显存}}！$$
> 这需要足足 9 张 80GB A100 显卡仅仅用来存 KV Cache，完全无法支撑高并发。

---

### 2. MQA (Multi-Query Attention，多查询注意力)

#### 结构定义
为了彻底解决显存瓶颈，Noam Shazeer 于 2019 年提出了 MQA：
* Query 保持原有的 $N$ 个头不变：$n_q = N$；
* **Key 和 Value 整个层只有 1 个头**：$n_k = n_v = 1$。

所有 $N$ 个 Query 头在计算注意力打分时，共享同一份 $K$ 和 $V$：
$$\text{head}_i = \text{Softmax}\left(\frac{Q_i K^\top}{\sqrt{d_{\text{head}}}}\right) V$$

#### 收益与致命缺陷
* **收益**：KV Cache 显存体积直接缩减为 MHA 的 $\frac{1}{N}$！如果原本有 32 个头，KV Cache 显存**暴降 96.9%**，显存读取带宽压力几乎归零。
* **缺陷**：由于全网只有一组 Key 和 Value，所有的 Query 头被迫在完全相同的上下文语义池中打分，模型失去了在不同子空间捕捉不同特征的能力，导致大模型在复杂推理、长文本关联等任务上**严重欠拟合，精度出现显著下跌**。

---

### 3. GQA (Grouped-Query Attention，分组查询注意力)

#### 结构定义
GQA 是在 MHA 与 MQA 之间取得最佳折中的工业界统一标准（2023 年提出）：
* 将 $n_q = N$ 个 Query 头均分成 $G$ 个分组（Group）；
* 每一组包含 $\frac{N}{G}$ 个 Query 头，这一组 Query 头**共享 1 个 Key 头和 1 个 Value 头**；
* Key 和 Value 的头数变为 $n_k = n_v = G$。

典型配置是 $8 : 1$（例如 32 个 Q 头配 4 个 KV 头）或 $6 : 1$。

```text
以 8:1 共享比例为例:
[Group 0] Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7  ───> 共享 K0, V0
[Group 1] Q8, Q9, Q10, Q11, Q12, Q13, Q14, Q15 ──> 共享 K1, V1
...
```

#### 工程收益与工业地位
1. **显存降低 87.5%**（以 8:1 为例）：KV Cache 体积仅为 MHA 的 $1/8$，极大缓解了显存墙；
2. **精度几乎无损**：实验表明，经过充分训练后，GQA 在绝大部分评测基准上的表现与 MHA 几乎完全无差异；
3. **成为当今开源大模型绝对主流**：LLaMA-2/3、Qwen-2/2.5、Mistral、Gemma 均全面拥抱 GQA。

> [!NOTE]
> **与推理引擎优化联动**：
> 正是由于 GQA 拥有多头共享同一份 KV 的特性，在 1Cat-vLLM 等推理算子中，才诞生了 **GQA-packed Wide GEMM**：将属于同一个 KV 分组的多个 Q 头在行维度打包拼接，只做单次宽矩阵乘，从而消除同一份 KV 在显存中的多次重复加载！

---

### 4. MLA (Multi-Head Latent Attention，多头潜在注意力)

MLA 是 DeepSeek 在 **DeepSeek-V2** 中提出并沿用至 **DeepSeek-V3 / R1** 的革命性注意力架构。

#### 核心思考：GQA 的局限
GQA 的压缩本质是**“硬性删减 KV 头的数量”**，虽然比 MQA 好，但仍然不可避免地压缩了 Key/Value 的表征自由度。  
DeepSeek 提出了颠覆性的逆向思考：
> **我们能不能在计算时依然拥有等价于完整 MHA（例如 128 个头）的庞大表征空间，而在将数据写入显存（KV Cache）时进行极致压缩？**

#### 核心机制一：低秩联合压缩（Low-Rank Joint Compression）
MLA 不直接把每一层的隐藏状态 $h_t$ 投影成多头 $K$ 和 $V$，而是先用一个降维矩阵 $W^{DKV}$ 将它们联合压缩成一个**低维隐向量（Latent Vector）** $\mathbf{c}_t^{KV}$：

$$\mathbf{c}_t^{KV} = W^{DKV} h_t \quad (\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c})$$

其中压缩维度 $d_c$（例如 512 维）远小于所有多头展开后的总维度（$n_h \times d_h = 128 \times 128 = 16384$ 维）。

在显存中，**KV Cache 彻底抛弃了展开后的多头 Key 和 Value，仅仅保存这个极短的低维向量 $\mathbf{c}_t^{KV}$！**

```text
传统 MHA KV Cache 存储内容:
[ Head 0 ] [ Head 1 ] ... [ Head 127 ] ──> 巨幅显存开销

MLA KV Cache 存储内容:
[              低维隐向量 c_t^{KV} (仅 512 维)              ] ──> 显存压缩数倍！
```

#### 核心机制二：矩阵吸收（Weight Absorbing）与推理无损还原
如果在每次推理时，都要先把 $\mathbf{c}_t^{KV}$ 乘以升维矩阵 $W^{UK}$ 展开成完整的多头 Key，那么依然会在片上产出巨大的张量，得不偿失。  
MLA 运用了**矩阵结合律**实现了神级的计算融合：

根据定义，还原出的第 $i$ 个 Key 为：
$$K_i = W_i^{UK} \mathbf{c}_t^{KV}$$

Query 与 Key 的点积打分为：
$$Q_i K_i^\top = Q_i \left( W_i^{UK} \mathbf{c}_t^{KV} \right)^\top = Q_i \left( \mathbf{c}_t^{KV} \right)^\top (W_i^{UK})^\top = \left( Q_i W_i^{UK} \right) \left( \mathbf{c}_t^{KV} \right)^\top$$

> [!TIP]
> **关键数学结论（矩阵吸收）**：
> 升维矩阵 $W_i^{UK}$ 是**静态模型权重**！在自回归推理期间，我们**根本不需要在显存中展开 Key**！  
> 只需要在算完当前 Token 的 $Q_i$ 之后，直接用 $Q_i$ 乘以上层权重矩阵 $W_i^{UK}$，得到一个吸收后的新查询向量：
> $$\tilde{Q}_i = Q_i W_i^{UK}$$
> 然后直接拿 $\tilde{Q}_i$ 与显存中缓存的低维隐向量 $\mathbf{c}_t^{KV}$ 做点积即可！对 Value 的计算也具有完全相同的吸收对称性。

#### 核心机制三：解耦旋转位置编码（Decoupled RoPE）
如果将旋转位置编码（RoPE）直接乘在 $K$ 上：
$$K_i^{\text{RoPE}} = \mathcal{R}_t \left( W_i^{UK} \mathbf{c}_t^{KV} \right)$$
由于位置旋转矩阵 $\mathcal{R}_t$ 与时间步 $t$ 强绑定且不满足简单的矩阵交换律，**它会彻底破坏上述矩阵吸收的结合律**！

为了解决这个难题，MLA 提出了**解耦 RoPE 策略**：
1. **语义内容向量（无位置编码）**：走低秩联合压缩 $\mathbf{c}_t^{KV}$，完全享受矩阵吸收与极致显存压缩；
2. **位置编码向量（解耦外挂）**：为每个头单独分配一个极小维度的专属 RoPE 向量（例如 $d_R = 64$ 维）：
   $$K_t^R = \mathcal{R}_t \left( W^{KR} h_t \right)$$
3. **最终打分**：内容打分与位置打分直接相加：
   $$\text{Score}_{i, t} = \frac{\tilde{Q}_i (\mathbf{c}_t^{KV})^\top + Q_i^R (K_t^R)^\top}{\sqrt{d_{\text{head}} + d_R}}$$

---

## 四、四大注意力机制横向对比与选型全景

| 机制属性 | MHA (标准多头) | MQA (多查询) | GQA (分组查询) | MLA (多头潜在) |
| :--- | :---: | :---: | :---: | :---: |
| **Q : KV 比例** | $1 : 1$ 绑定 | $N : 1$ 极限共享 | $G : 1$（如 $8 : 1$）分组 | 全头表达保留，低秩隐空间共享 |
| **单 Token KV 缓存大小** | $2 \cdot n_h \cdot d_h$（基线 **100%**） | $2 \cdot d_h$（~**3%~5%**） | $2 \cdot \frac{n_h}{G} \cdot d_h$（~**12.5%**） | $d_c + d_R$（~**9% 以下**） |
| **KV Cache 存储内容** | 完整多头 $K, V$ 矩阵 | 仅 1 组单头 $K, V$ | $G$ 组独立的 $K, V$ | 仅存压缩隐向量 $\mathbf{c}^{KV}$ + 解耦 $K^R$ |
| **表达能力 / 建模精度** | 完整无损（理论上限） | 明显受损（易欠拟合） | 几乎无损（基准测试打平） | 媲美乃至超越标准 MHA |
| **矩阵吸收可行性** | 否（直接展开） | 否 | 否 | **是（推理期权重完美吸收）** |
| **典型代表模型** | 经典 Transformer, GPT-3 | PaLM, Falcon | LLaMA-2/3, Qwen-2.5 | DeepSeek-V2, V3, R1 |
| **工业定位与评价** | 训练友好但推理代价过高 | 极端节省显存但损伤能力 | 当下主流大模型通用工业标准 | 超大参数长文本推理的代际革新 |

---

## 五、在推理系统（vLLM / PagedAttention）中的协同考量

在大模型 Serving 系统（如 vLLM、SGLang、1Cat-vLLM）中，不同注意力机制对底层物理显存管理和算子调度提出了截然不同的要求：

1. **PagedAttention 的 Block 组织**：
   * 在 **MHA / GQA** 中，物理块（Physical Page Block，如每块 16 个 Token）存储的是展开后的 `[block_size, num_kv_heads, head_dim]` 张量；
   * 在 **MLA** 中，物理块存储的张量形状直接转变为 `[block_size, latent_dim + rope_dim]`，单块内存体积大幅缩小，使得单卡可以容纳数十倍的并发上下文 Block。
2. **算子 GEMM 优化方向**：
   * **GQA 场景**：核心优化在于打满小矩阵计算密度（即 1Cat-vLLM 实现的 **GQA-packed Wide GEMM**，将 6~8 个 Q 头在行维度拼接成大矩阵，只扫一遍 KV Cache）；
   * **MLA 场景**：核心优化在于**投影视角的算子重排**，在 Prefill 阶段进行低秩投影，在 Decode 阶段将 $W^{UK}$ 与动态 Query 预先融合为宽 GEMM 运算。

---
*本文档为 1Cat-vLLM 学习笔记的独立模块，归属于「知识点汇总」栏目，由构建脚本自动渲染编译。*
