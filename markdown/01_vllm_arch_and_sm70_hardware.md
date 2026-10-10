# 01 - vLLM 架构全景与 Volta (SM70) 硬件底座

> **模块标签**：`1Cat-vLLM 源码精读` · `第 01 讲` · `CUDA 体系结构` · `SM70 Volta`

---

## 导读与学习目标

本讲作为全系列的基础篇，旨在建立从大模型高层系统调度（vLLM）到底层 GPU 物理硬件（Tesla V100 / SM70）的完整链路认知：

1. **计算与访存边界**：厘清大模型推理两阶段的硬件瓶颈差异（Prefill 算力受限 vs Decode 访存受限）。
2. **系统解构**：掌握 vLLM 的 PagedAttention 显存分页机制，以及张量元数据如何进入 C++/CUDA 算子层。
3. **硬件架构限制**：剖析 Volta 架构的底层特征（缺少现代指令集、寄存器与共享内存容量紧张）。
4. **数学基础**：掌握 FlashAttention 原理与 Online Softmax 的分块递推公式。

---

## 一、大模型推理的两大阶段与性能瓶颈

在编写任何 GPU 算子前，需要区分大模型运行的两个阶段：

### 1. Prefill（首字预填充 / 提示词阶段）
* **计算特征**：输入整个 Prompt（例如 2048 个 token），所有 Token 并行计算 Q, K, V。
* **瓶颈类型**：**Compute-Bound（算力受限）**。
  - GEMM 矩阵尺寸大（如 $M=2048, K=4096, N=4096$）；
  - 能够打满 GPU 的 Tensor Core 计算单元。

### 2. Decode（自回归解码 / 逐字生成阶段）
* **计算特征**：每次根据上一个 token 生成下一个 token（即 $M=1$）。为了计算当前单个 Token 与历史所有上下文的注意力关联，必须从显存中完整读取历史所有已生成 Token 的 Key 和 Value（**KV Cache**）。
* **瓶颈类型**：**Memory-Bound（访存带宽受限）**。
  - 浮点计算量较小，但显存数据搬运量大。

> [!NOTE] 算力访存比（Arithmetic Intensity）分析
>
> $$\text{Arithmetic Intensity} = \frac{\text{计算浮点数 (FLOPs)}}{\text{显存搬运字节数 (Bytes)}}$$
>
> 在单请求 Decode 阶段，算力访存比极低（通常仅 $1 \sim 2 \text{ FLOPs/Byte}$）。
> **Tesla V100** 的 FP16 Tensor Core 理论算力为 **125 TFLOP/s**，HBM2 显存带宽为 **900 GB/s**。
> 计算单元饱和所需的临界算力访存比为：
>
> $$\frac{125 \times 10^{12} \text{ FLOP/s}}{900 \times 10^9 \text{ B/s}} \approx 138.8 \text{ FLOPs/Byte}$$
>
> **结论**：单请求 Decode 阶段大部分时间 GPU 计算单元处于等待数据状态，显存带宽利用率直接决定了系统的实际吞吐。

---

## 二、vLLM 系统骨架与数据流转

vLLM 核心解决了显存碎片化问题，并引入了 Iteration 级别的调度流水线：

```text
1. 调度层 (Scheduler & Continuous Batching)
   └─ 每个 Iteration 动态调度新请求加入与完成请求释放。
       ↓
2. 显存管理层 (BlockManager & PagedAttention)
   └─ 类似 OS 分页机制，显存划分为物理 Block (例如 16 tokens/block)，消除显存空洞。
       ↓
3. 执行器与元数据装配 (ModelRunner)
   └─ 打包 GPU 输入：input_tokens, block_tables, context_lens, slot_mapping。
       ↓
4. C++/CUDA 桥接层 (csrc/torch_bindings.cpp)
   └─ 通过 Torch Custom Op 将张量指针直接送入 CUDA Kernel。
       ↓
5. 架构特化算子层 (Flash-V100 / SM70-TurboMind)
   └─ 执行 Volta 定制的 Wide QK/PV 打包、W4A16 软解算子。
```

### Paged 机制对底层 CUDA 寻址的影响

连续显存矩阵乘法的地址计算是线性的：
```text
Address = Base + token_idx * Stride
```

而在 PagedAttention 中，读写 KV Cache 变成了**二级跳表寻址**（整数除法算块号，取模算余数）。

#### 具体计算示例
假设每个物理块固定存放 16 个 Token（`BLOCK_SIZE = 16`）：
* 当前正在处理第 35 个 Token（下标从 0 起，`token_idx = 35`）；
* 调度器分配给该请求的物理块映射表为：`block_table = [7, 12, 3, 9]`。

寻址定位过程如下：
1. **计算逻辑块号（整除）**：`35 // 16 = 2`，说明第 35 个 Token 属于该序列的第 2 号逻辑块；
2. **查表获取物理块号（查表）**：`block_table[2] = 3`，说明数据实际落在显存池的第 3 号物理块；
3. **计算块内偏移（取模）**：`35 % 16 = 3`，说明在该物理块内部位于第 3 个槽位；
4. **最终显存物理地址**：
   ```text
   物理地址 = 显存池基址 + (3 * 物理块字节大小) + (3 * 单个Token的KV字节大小)
   ```

#### 真实 CUDA 算子中的代码对照
对应于项目源码 `csrc/attention/sm70_v37/tail.cu`：
```cuda
// 1. 整数除法得到逻辑页号，直接查表获取物理页号
const int physical_page = block_table[token_idx / page_size];

// 2. 取模得到物理页内的具体行/槽位
const int page_offset   = token_idx % page_size;

// 3. 计算实际显存读取指针
const half* k_ptr = kv_cache + physical_page * page_stride + page_offset * head_dim;
```

> [!WARNING] 访存合并要求
> 如果每个 CUDA 线程各自独立计算非对齐的地址，会导致严重的非合并访存（Uncoalesced Access）。因此在 Kernel 设计中，需要通过线程协同，将一个物理块内的数据通过 **128-bit 向量化读写（`float4` / `uint4`）** 批量搬运至 Shared Memory。

---

## 三、Volta 架构（SM70 / Tesla V100）硬件特征

| 架构特性 | Tesla V100 (Volta / SM70) | A100 (Ampere / SM80) | H100 (Hopper / SM90) |
| :--- | :--- | :--- | :--- |
| **Tensor Core 指令** | `mma.sync.m8n8k4` (小粒度 HMMA) | `mma.sync.m16n8k16` (吞吐翻倍) | WGMMA / TMA 硬件单元支持 |
| **低比特硬件支持** | **无原生 FP8 / 无原生 INT4** (仅 FP16/FP32) | 原生 INT8 / INT4 | 原生硬件 FP8 (E4M3 / E5M2) |
| **显存到共享内存** | **必须经过通用寄存器中转** (消耗寄存器) | `cp.async` 硬件异步直拷 | TMA (Tensor Memory Accelerator) |
| **Shared Memory 容量** | L1 与 Shared Memory 共享 128KB | 最高 164 KB，硬件异步屏障 | 最高 228 KB，集群间共享 (DSM) |

### V100 算子开发的主要约束

1. **寄存器溢出（Register Spilling）**：
   缺乏 `cp.async` 指令，显存到共享内存必须遵循 `Global Memory -> Register -> Shared Memory`。线程占用寄存器过多时，活跃 Warp 比例下降，难以掩盖访存延迟。
2. **HMMA 指令发射开销**：
   Volta Tensor Core 单次只计算 $8 \times 8 \times 4$ 矩阵。计算一个较大的 Block Tile 需要发射大量 HMMA 汇编指令，容易触及指令发射瓶颈。
3. **缺少低比特计算指令**：
   在 V100 上运行量化模型（如 NVFP4 / W4A16 / FP8）时，无法直接调用专用硬件指令。必须在 CUDA 软件层面通过**位运算在寄存器内解包为 FP16**，再交由 HMMA 执行。

---

## 四、Attention 计算演进：从朴素实现到 Online Softmax

标准 Attention 公式：
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### 1. 朴素实现的显存开销
若依次执行 $S = QK^T \to P = \text{Softmax}(S) \to O = PV$：
- 中间矩阵 $S \in \mathbb{R}^{N \times N}$ 和 $P \in \mathbb{R}^{N \times N}$ 必须完整写回显存再读出；
- 显存占用与显存带宽往返都是 $O(N^2)$。在长上下文下容易导致显存溢出。

### 2. FlashAttention 原理：Online Softmax

标准 Attention 公式：
$$O = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V = P V$$

为了防止指数爆炸（数值溢出），工程中必须使用 **Safe Softmax**。若序列长度为 $N$，传统 Safe Softmax 计算向量 $x \in \mathbb{R}^N$ 需顺序完成三遍遍历：
1. **第 1 遍**：求全局最大值 $m = \max_{1 \le i \le N} x_i$；
2. **第 2 遍**：求减去最大值后的指数和分母 $l = \sum_{i=1}^N e^{x_i - m}$；
3. **第 3 遍**：计算每个元素的概率 $P_i = \frac{e^{x_i - m}}{l}$ 并与 $V$ 做加权求和。

**痛点**：传统方法必须等**全部 $N$ 个元素都看完**后才能得到全局 $m$，这意味着不能边加载局部 $K, V$ 边计算最终结果，不得不将 $N \times N$ 的中间矩阵 $S$ 和 $P$ 频繁往返写入全局显存。

---

#### 核心推导：两块数据的 Softmax 增量拼接

设整个序列被切分为两个数据块（Block 1 与 Block 2）：
$$x = \left[ x^{(1)}, x^{(2)} \right], \quad x^{(1)} \in \mathbb{R}^{B_1}, \; x^{(2)} \in \mathbb{R}^{B_2}$$

##### 步骤 1：局部统计量定义
假设我们刚刚只加载了 Block 1，算出了局部最大值与局部指数和：
$$m^{(1)} = \max_{i \in B_1} x_i^{(1)}, \quad l^{(1)} = \sum_{i \in B_1} e^{x_i^{(1)} - m^{(1)}}$$

此时仅用 Block 1 计算的局部输出向量 $O^{(1)}$ 为：
$$O^{(1)} = \sum_{i \in B_1} P_i^{(1)} V_i^{(1)} = \frac{\sum_{i \in B_1} e^{x_i^{(1)} - m^{(1)}} V_i^{(1)}}{l^{(1)}}$$
为了便于后续分子合并，定义局部未归一化加权和分子为 $\tilde{O}^{(1)}$：
$$\tilde{O}^{(1)} = \sum_{i \in B_1} e^{x_i^{(1)} - m^{(1)}} V_i^{(1)} = l^{(1)} \cdot O^{(1)}$$

同理，当新加载 Block 2 时，它的局部统计量为：
$$m^{(2)} = \max_{j \in B_2} x_j^{(2)}, \quad l^{(2)} = \sum_{j \in B_2} e^{x_j^{(2)} - m^{(2)}}, \quad \tilde{O}^{(2)} = \sum_{j \in B_2} e^{x_j^{(2)} - m^{(2)}} V_j^{(2)} = l^{(2)} \cdot O^{(2)}$$

##### 步骤 2：全局最大值更新
两块合并后的全局最大值显然为两者较大者：
$$m^{\text{new}} = \max\left(m^{(1)}, m^{(2)}\right)$$

##### 步骤 3：归一化分母（指数和）的增量修正
全局正确的归一化分母定义为以 $m^{\text{new}}$ 为基准的全部元素指数和：
$$l^{\text{new}} = \sum_{k \in B_1 \cup B_2} e^{x_k - m^{\text{new}}} = \sum_{i \in B_1} e^{x_i^{(1)} - m^{\text{new}}} + \sum_{j \in B_2} e^{x_j^{(2)} - m^{\text{new}}}$$

利用指数恒等式 $e^{a - c} = e^{a - b} \cdot e^{b - c}$，将指数项拆分：
$$\sum_{i \in B_1} e^{x_i^{(1)} - m^{\text{new}}} = \sum_{i \in B_1} \left( e^{x_i^{(1)} - m^{(1)}} \cdot e^{m^{(1)} - m^{\text{new}}} \right) = e^{m^{(1)} - m^{\text{new}}} \cdot \underbrace{\sum_{i \in B_1} e^{x_i^{(1)} - m^{(1)}}}_{l^{(1)}}$$

同理对 Block 2 处理：
$$\sum_{j \in B_2} e^{x_j^{(2)} - m^{\text{new}}} = e^{m^{(2)} - m^{\text{new}}} \cdot \underbrace{\sum_{j \in B_2} e^{x_j^{(2)} - m^{(2)}}}_{l^{(2)}}$$

代回即可得到**分母递推公式**：
$$l^{\text{new}} = e^{m^{(1)} - m^{\text{new}}} \cdot l^{(1)} + e^{m^{(2)} - m^{\text{new}}} \cdot l^{(2)}$$

> **数值稳定性注意**：由于 $m^{\text{new}} \ge m^{(1)}$ 且 $m^{\text{new}} \ge m^{(2)}$，缩放指数差值 $m^{(1)} - m^{\text{new}} \le 0$ 以及 $m^{(2)} - m^{\text{new}} \le 0$，因此系数 $e^{\Delta m} \in (0, 1]$，计算时**绝对不会发生上溢**。

##### 步骤 4：注意力输出向量 $O$ 的在线递推
最终全局加权输出定义为：
$$O^{\text{new}} = \frac{\sum_{k \in B_1 \cup B_2} e^{x_k - m^{\text{new}}} V_k}{l^{\text{new}}}$$

展开分子（未归一化总分子 $\tilde{O}^{\text{new}}$）：
$$\tilde{O}^{\text{new}} = \sum_{i \in B_1} e^{x_i^{(1)} - m^{\text{new}}} V_i^{(1)} + \sum_{j \in B_2} e^{x_j^{(2)} - m^{\text{new}}} V_j^{(2)}$$

两块提取缩放系数的过程在数学上是**完全对称**的：
* **Block 1 分子缩放**：
  $$\sum_{i \in B_1} e^{x_i^{(1)} - m^{\text{new}}} V_i^{(1)} = e^{m^{(1)} - m^{\text{new}}} \sum_{i \in B_1} e^{x_i^{(1)} - m^{(1)}} V_i^{(1)} = e^{m^{(1)} - m^{\text{new}}} \cdot l^{(1)} \cdot O^{(1)}$$
* **Block 2 分子缩放**：
  $$\sum_{j \in B_2} e^{x_j^{(2)} - m^{\text{new}}} V_j^{(2)} = e^{m^{(2)} - m^{\text{new}}} \sum_{j \in B_2} e^{x_j^{(2)} - m^{(2)}} V_j^{(2)} = e^{m^{(2)} - m^{\text{new}}} \cdot l^{(2)} \cdot O^{(2)}$$

将分子代入归一化分母 $l^{\text{new}}$，得出**完全对称的输出增量递推公式**：
$$O^{\text{new}} = \frac{e^{m^{(1)} - m^{\text{new}}} \cdot l^{(1)} \cdot O^{(1)} \;+\; e^{m^{(2)} - m^{\text{new}}} \cdot l^{(2)} \cdot O^{(2)}}{l^{\text{new}}}$$

亦可直观写为“**旧结果衰减更新 + 新结果加权补入**”的凸组合形式：
$$O^{\text{new}} = \left( \frac{l^{(1)} \cdot e^{m^{(1)} - m^{\text{new}}}}{l^{\text{new}}} \right) \cdot O^{(1)} \;+\; \left( \frac{l^{(2)} \cdot e^{m^{(2)} - m^{\text{new}}}}{l^{\text{new}}} \right) \cdot O^{(2)}$$

---

#### 深入剖析：为什么工程上可以省去除法？（数学约分与硬件考量）

在纯数学递推中，Block 2 贡献的项写作：
$$\text{分子贡献项} = e^{m^{(2)} - m^{\text{new}}} \cdot \Big( l^{(2)} \cdot O^{(2)} \Big)$$

很多读者会感到困惑：**既然写了 $O^{(2)}$，为什么在工程代码中既看不到除以 $l^{(2)}$，也看不到乘以 $l^{(2)}$？**

##### 1. 数学上的直接约分抵消
回顾两者的严格定义：
* **未归一化的局部矩阵乘积**：定义 $\tilde{P}^{(2)} = \exp\left(Q (K^{(2)})^\top - m^{(2)}\right)$，其与 $V^{(2)}$ 的矩阵乘积即为分子的原始加权和：
  $$\text{局部未归一化分子} = \tilde{P}^{(2)} V^{(2)} = \sum_{j \in B_2} e^{x_j^{(2)} - m^{(2)}} V_j^{(2)}$$
* **局部归一化输出**：定义为分子除以分母 $l^{(2)}$：
  $$O^{(2)} = \frac{\tilde{P}^{(2)} V^{(2)}}{l^{(2)}}$$

若机械地将 $O^{(2)}$ 带入递推公式，展开后会发现：
$$l^{(2)} \cdot O^{(2)} = l^{(2)} \cdot \left( \frac{\tilde{P}^{(2)} V^{(2)}}{l^{(2)}} \right) = \tilde{P}^{(2)} V^{(2)}$$

**关键结论**：分母上的 $l^{(2)}$ 与外层的 $l^{(2)}$ 在数学上**直接完全抵消**！因此新块实际需要累加的分子项，纯粹就是：
$$e^{m^{(2)} - m^{\text{new}}} \cdot \left( \tilde{P}^{(2)} V^{(2)} \right)$$

##### 2. 硬件层面的关键收益（GPU Tensor Core vs 浮点除法）
在 GPU 芯片底层硬件中，这一约分抵消带来了决定性的性能优势：

* **Tensor Core 的极致吞吐**：NVIDIA GPU（如 Tesla V100）的 Tensor Core 专门针对半精度矩阵乘累加（MMA）做了硬件级加速，单指令即可吞吐大规模矩阵乘，能极速算完 $\tilde{P}^{(2)} V^{(2)}$；
* **浮点除法（DIV）极其昂贵**：Tensor Core **原生不支持除法**！除法运算必须交给普通的通用算术单元（ALU/SFU）逐个标量串行计算，指令延迟长达数十个周期。如果每处理一个 Tile 都强行先做一次全量除法求出 $O^{(2)}$，会导致严重的计算单元流水线停顿（Stall）；
* **流水线对比**：
  * **低效做法（机械算 $O^{(2)}$）**：Tensor Core 算完 $\tilde{P}^{(2)} V^{(2)} \to$ 打断流水线切换到通用 ALU 逐元素除以 $l^{(2)} \to$ 再乘回 $l^{(2)} \to$ 累加进寄存器；
  * **优化做法（利用数学抵消）**：Tensor Core 算完 $\tilde{P}^{(2)} V^{(2)} \to$ 直接由 Tensor Core / FMA 乘以缩放因子累加进片上寄存器，**全程无多余除法指令**！

---

#### 递推结论与硬件状态演化

在 CUDA Kernel 运行期间，每个线程/Warp 无需任何全局中间显存缓冲区，仅在**片上寄存器**中维持三个常数级状态变量：
* 标量 $m \in \mathbb{R}$（当前累积最大值，初值设为 $-\infty$）；
* 标量 $l \in \mathbb{R}$（当前累积指数和分母，初值设为 $0$）；
* 向量 $O \in \mathbb{R}^d$（当前累积加权结果向量，初值设为 $\vec{0}$）。

```text
片上通用寄存器 (维持当前状态):
   ┌─────────┐   ┌─────────┐   ┌─────────────────┐
   │  m = -∞ │   │  l = 0  │   │     O = 0       │
   └─────────┘   └─────────┘   └─────────────────┘
        │             │                 │
        ▼ 循环依次载入下一个 Physical Block 的 K, V
   ┌─────────────────────────────────────────────┐
   │ 1. GEMM-1:  S_tile = Q * K_tile^T           │
   │ 2. Max/Sum: 计算该 Tile 的局部 m_tile, l_tile│
   │ 3. Rescale: 按上述公式就地更新 m, l, O 寄存器 │
   │ 4. GEMM-2:  O += P_tile * V_tile 累加       │
   └─────────────────────────────────────────────┘
        │
   循环结束 ──> 寄存器中的 O 即为严格等于标准 Softmax 的最终数学结果！
```

##### 核心工程收益
1. **显存访问复杂度降低**：中间 $N \times N$ 的打分矩阵 $S$ 和概率矩阵 $P$ 彻底无需写入全局显存，显存访存复杂度从 $O(N^2)$ 降低到 $O(N)$；
2. **算子融合（Fused Kernel）**：GEMM-1 ($QK^T$)、Softmax 和 GEMM-2 ($PV$) 在单次 Kernel 执行内流水线完成。

---

## 五、1Cat-vLLM 在 V100 上的优化实现（结合真实源码解析）

在早期实现中，1Cat-vLLM 在 Tesla V100（Volta / SM70）上运行自注意力算子的实测吞吐仅约 **17.92 TFLOP/s**，仅达到 V100 理论半精度峰值算力（125 TFLOP/s）的约 14.3%，出现了严重的算力塌陷。在引入架构优化后，实测吞吐提升至 **≈60.8 TFLOP/s**（并在最新的 `sm70_79t` 架构中突破 **75+ TFLOP/s**，算力提升超 3.4~4.2 倍）。

下面结合 `1CatAI/1Cat-vLLM` 仓库的实际 CUDA 源码（位于 [csrc/attention/sm70_v37/tail.cu](file:///D:/1Cat-vLLM/csrc/attention/sm70_v37/tail.cu) 与 [csrc/attention/sm70_v37/prefill.cu](file:///D:/1Cat-vLLM/csrc/attention/sm70_v37/prefill.cu)），对痛点根因与核心技术展开深度剖析。

---

### 1. 痛点根因：SM70 片上资源瓶颈、寄存器溢出与活跃度坍塌

现代开源大模型（如 LLaMA-2/3、Qwen 2.5、Mistral）的标准头维度通常为：
$$d_{\text{head}} = 128 \quad \text{或} \quad d_{\text{head}} = 256$$

原生 FlashAttention-2 主要是面向 Ampere (SM80 / A100) 及更现代架构设计的：
* **A100 架构优势**：单 SM 配备 164 KB 可配置共享内存（Shared Memory），原生支持硬件级异步数据搬运指令 `cp.async`（直接由全局显存直通共享内存，不消耗通用寄存器），且具备更大规模的寄存器堆与更高发射带宽。
* **Volta (SM70 / V100) 架构硬约束**：

#### (1) 共享内存（Shared Memory）容量受限
* V100 每个 SM 的 L1 Cache 与 Shared Memory 是**统一物理 SRAM**（共 128 KB），硬件只允许配置为 64 KB 或 96 KB 的 Shared Memory。
* 若不加切分地分配 $B_r = 64, B_c = 64, d = 128$ 的 $Q, K, V$ Tile 及中间双缓冲，仅张量本身就逼近甚至超出 64 KB，导致单 SM 无法容纳多个并发 Block。

#### (2) 寄存器文件（Register File）与溢出（Register Spilling）机制
* **硬件规格**：V100 每个 SM 拥有 65,536 个 32-bit 通用寄存器，由 4 个独立的 Processing Block（SMSP / Sub-core）均分，每个 Sub-core 拥有 16,384 个寄存器和 1 个 Warp 调度器。
* **单线程上限**：CUDA 硬件架构限制单个线程最多只能分配 **255 个 32-bit 寄存器**。
* **寄存器膨胀根因**：在标准 FlashAttention 中，线程需要同时驻留：
  1. $QK^\top$ 矩阵乘累加的 FP32 累加器片段（Accumulator Fragments）；
  2. $PV$ 矩阵乘累加的输出 $O$ 累加器片段（$B_r \times d$，若 $d=256$，累加器矩阵高达 $64 \times 256$ 个 float，即 64 KB 寄存器数据）；
  3. 全局访存到共享内存的数据中转寄存器（因为 Volta **没有** `cp.async`，所有数据搬运必须经由 `LDG.E -> 寄存器 -> STS` 这一路径，常驻寄存器占用极高）；
  4. Online Softmax 的行最大值 $m$、行和 $l$、循环计数与指针偏移。
* **溢出惩罚（Local Memory Spilling）**：当单个线程所需寄存器超过限额（或受 `__launch_bounds__` 限制）时，NVIDIA NVCC 编译器会强制将溢出的变量存放到**本地内存（Local Memory）**。
  > [!CAUTION]
  > 在 GPU 物理硬件中，所谓的 **Local Memory 在物理上根本不是片上内存，而是位于高延迟、高功耗的板载显存（HBM2 DRAM）**，仅靠少量的 L1/L2 缓存缓冲。
  > 一旦发生 Register Spill，核心内密集的数学运算就会被大量的本地内存读写指令（`LDL` / `STL`）打断，指令延迟从寄存器的 **1~4 个时钟周期** 暴增到显存访存的 **200~400 个周期**，流水线直接被挂起！

#### (3) 活跃度（Occupancy）暴跌与延迟掩盖失效
GPU 依赖海量并发 Warp 之间的快速轮转切换（Warp Interleaving）来掩盖长达数百周期的访存延迟。
* V100 单 SM 最大并发容量为 **2048 个线程（64 个 Warp）**。
* 假设一个线程块（CTA）包含 256 个线程，若因头维度过大，每个线程占用 128 个寄存器：
  $$\text{单 Block 寄存器消耗} = 256 \times 128 = 32,768 \text{ 个寄存器}$$
  单 SM（总共 65,536 个寄存器）最多只能同时常驻：
  $$\frac{65,536}{32,768} = 2 \text{ 个 Block} = 512 \text{ 个线程} = 16 \text{ 个 Warp}$$
* 此时理论活跃度（Theoretical Occupancy）暴跌至：
  $$\text{Occupancy} = \frac{16}{64} = 25\%$$
* 活跃度跌至 25% 意味着当这 16 个 Warp 都在等待显存加载或本地内存 Spill 访存时，SM 的计算单元完全处于**饥饿空转状态（Stall）**，导致实测算力暴跌至 17.92 TFLOP/s。

---

### 2. 优化一：Split-D（头维度空间切分）源码与架构落地

#### 核心设计思路
既然 $d=128$ 或 $d=256$ 会导致单线程维护的 $O$ 累加器片段过大并引爆寄存器溢出，1Cat-vLLM 提出将特征维度 $d$ 沿空间轴进行切分（Tile 划分）。以大模型常用的 $d=256$ 为例，切分成 4 个小块（每块 $d_{\text{chunk}} = 64$）。

#### 真实源码剖析：`Sm70D256SplitDTraits`
在 [csrc/attention/sm70_v37/tail.cu](file:///D:/1Cat-vLLM/csrc/attention/sm70_v37/tail.cu) 的第 28~78 行中，1Cat-vLLM 定义了严格针对 Volta 硬件特性的 Traits 结构体：

```cpp
// 摘自 1Cat-vLLM/csrc/attention/sm70_v37/tail.cu
struct Sm70D256SplitDTraits {
  using Element = cutlass::half_t;
  // Volta 硬件 Tensor Core 基础原子: m8n8k4 FP32 累加
  using MmaAtom = MMA_Atom<SM70_8x8x4_F32F16F16F32_TN>;
  using PvMmaAtom = MMA_Atom<SM70_8x8x4_F32F16F16F32_TT>;

  static constexpr int kHeadDim = 256;
  static constexpr int kBlockM = 64;   // Query 行块大小
  static constexpr int kBlockN = 32;   // Key/Value 列块大小
  static constexpr int kDChunk = 64;   // 【Split-D 核心】头维度切块粒度为 64
  static constexpr int kDChunks = kHeadDim / kDChunk; // 256 / 64 = 4 个子块
  static constexpr int kOwnedDChunks = kDChunks / 2;  // 每对 Warp 仅负责 2 个 Chunk (即 D/2)
  static constexpr int kNThreads = 256; // 8 个 Warp
  static constexpr int kMmaThreads = 32;
  static constexpr int kWarpsPerGroup = 2; // 每组 2 个 Warp 协作
  static constexpr int kMmaGroups = kNThreads / (kWarpsPerGroup * kMmaThreads); // 4 组
  static constexpr int kGroupRows = kBlockM / kMmaGroups; // 64 / 4 = 16
  static constexpr int kQkWarpRows = kGroupRows / kWarpsPerGroup; // 16 / 2 = 8
  static constexpr int kQkRowsPerThread = kQkWarpRows / 4;        // 单线程仅持有 2 行 QK!
  static constexpr int kOutputRowsPerThread = kGroupRows / 4;     // 单线程仅持有 4 行输出!
```

#### 严格锁死的 Shared Memory 预算
在 [tail.cu](file:///D:/1Cat-vLLM/csrc/attention/sm70_v37/tail.cu) 第 123~127 行中，代码通过静态断言强制将 Shared Memory 大小锁死在 45.5 KB：
```cpp
  static constexpr int kTensorSmemBytes =
      (kQElements + 2 * kKVElements + kPElements) * sizeof(Element);
  static constexpr int kExchangeBytes = 2 * kExchangeRows * sizeof(float);
  static constexpr int kSmemBytes = kTensorSmemBytes + kExchangeBytes;
  // 严格断言总 Smem 为 45,568 字节 (~44.5 KB)
  static_assert(kSmemBytes == 45568);
```
> **硬件收益**：44.5 KB 极其精准地适配了 V100 的 64 KB / 96 KB Shared Memory 硬件配置，剩余的 19.5 KB 空间全部留给硬件 L1 Cache，保证了缓存命中率和并发调度灵活性。

#### 寄存器使用精确控制与零溢出（Zero Spilling）
在 [tail.cu](file:///D:/1Cat-vLLM/csrc/attention/sm70_v37/tail.cu) 第 437~448 行的 Kernel 主体中：
```cpp
  using OFragment = decltype(partition_fragment_C(
      pv_tiled_mma, Shape<Int<Traits::kGroupRows>, Int<kDChunk>>{}));
  constexpr int kOElements = decltype(size(OFragment{}))::value;
  using OLayout = typename OFragment::layout_type;
  
  // 单线程中驻留的 O 累加器: 仅维护 kOwnedDChunks (2个) 乘 kOElements
  float o_storage[Traits::kOwnedDChunks][kOElements];
```
* 因为单线程持有的 `kOutputRowsPerThread` 仅为 4，`kDChunk` 为 64，单线程中累加器占用的寄存器数量被严格控制在极小范围内（数十个 float 寄存器）；
* 整个 Kernel 加上启动边界声明：
  ```cpp
  __global__ __launch_bounds__(Sm70D256SplitDTraits::kNThreads, 1)
  void sm70_d256_splitd_dense_kernel(...)
  ```
  单线程总寄存器开销保持在 $\le 64 \sim 80$ 个，彻底消除了向 Local Memory 的溢出，同时使得单 SM 可以容纳更多并发 Block，Occupancy 大幅攀升。

---

### 3. 优化二：GQA-packed Wide QK/PV GEMM 源码与架构落地

#### 核心设计思路
在 LLaMA-3、Qwen 2.5 等主流大模型中，均采用 GQA（Grouped-Query Attention）结构。以 6 个 Query Heads 共享 1 个 KV Head（比例 $6 : 1$）为例：
* **朴素实现的缺陷**：
  1. 若将 6 个 Query Head 分别作为独立的 Grid/Block 启动，显存中的同一份 KV Cache 数据会被重复读取 6 次，极度挤占 HBM2 访存带宽；
  2. Volta 的 Tensor Core 指令 `mma.sync.m8n8k4` 单次仅处理 $8 \times 8 \times 4$ 微矩阵。若针对单个 Query Head 运算，行维度过窄（$M$ 维度过小），Tensor Core 的计算密度无法打满，而 GPU 前端的指令发射单元（Instruction Issue Unit）却被海量的小矩阵发射指令撑爆。
* **1Cat-vLLM 的方案**：在行维度（$M$ 轴）将 6 个 Query Heads **直接打包（Pack）**，拼接为一个大行数的宽矩阵，执行**单次超宽 GEMM 运算**！

#### 真实源码剖析：`sm70_d256_gqa_v37_fwd`
在 [csrc/attention/sm70_v37/prefill.cu](file:///D:/1Cat-vLLM/csrc/attention/sm70_v37/prefill.cu) 的第 455~480 行与 628~652 行中，给出了极其精妙的打包调度：

```cpp
// 摘自 1Cat-vLLM/csrc/attention/sm70_v37/prefill.cu
at::Tensor sm70_d256_gqa_v37_fwd(const at::Tensor& q, const at::Tensor& k,
                                 const at::Tensor& v, at::Tensor& out,
                                 double softmax_scale, bool causal) {
  constexpr int kHeadDim = 256;
  constexpr int kHeadsQ = 6;     // 6 个 Query 头
  constexpr int kHeadsKV = 1;    // 共享 1 个 KV 头
  
  const int kQuery = static_cast<int>(q.size(1));
  // 【核心打包】将 6 个 Query 头在行维度上打包拼接，行数变为 kQuery * 6
  const int kRows = kQuery * kHeadsQ; 
```

随后，在计算非因果前缀（Prefix）与后续块时，代码直接将打包后的行数 `kRows` 送入 CUTLASS GEMM 算子：
```cpp
  // QK 宽 GEMM 运算: 问题规模为 (M = kRows, N = width, K = kHeadDim)
  // 即 (6 * Q, block_kv, 256)
  typename QKGemm::Arguments arguments(
      {kRows, width, kHeadDim}, 1, 
      {query, QKLayoutA(kHeadDim)},                         // 打包后的 6 头 Query
      {key + size_t(begin) * kHeadDim, QKLayoutB(kHeadDim)}, // 单组 Key
      {score_ptr, typename QKGemm::LayoutC(width)},
      ...);
  operation.qk(stream); // 单次启动打满 Tensor Core!

  // PV 宽 GEMM 运算: 同样采用打包后的行数 kRows 执行单次宽乘累加
  operation.pv = std::make_unique<PVLauncher>(
      score_ptr, value + size_t(begin) * kHeadDim, prefix_accumulator_ptr,
      kRows, width, old_scale_ptr, block_scale_ptr, block == 0);
  operation.pv->launch(stream);
```

#### 软硬件协同收益
1. **显存读取带宽降低至 1/6（片上复用率提升 600%）**：同一份 Key 和 Value 数据加载进片上后，被 6 个 Query Head 在单次 GEMM 中共同乘累加消费，彻底消除跨 Kernel 的 KV 重复加载；
2. **打满 Volta Tensor Core 计算密度**：单次 GEMM 的行数由 $Q$ 扩展为 $6 \cdot Q$（例如 $Q=64$ 时扩展至 384 行，甚至数千行），使得 CUTLASS M128/N256/K32 Tensor Core 流水线能够完全被填满，消除了小矩阵指令发射饥饿，实测吞吐一举从 17.92 TFLOP/s 跃升至 60.8+ TFLOP/s。

---

### 4. 优化前后全景对比矩阵

| 评估指标 | 原生实现 / 基线版本 | 1Cat-vLLM 架构优化实现 | 底层硬件与编译机制 |
| :--- | :--- | :--- | :--- |
| **实测计算吞吐** | **17.92 TFLOP/s** (~14.3% 峰值) | **≈ 60.8 ~ 75+ TFLOP/s** | 算力翻升 3.4 ~ 4.2 倍，逼近 Volta 硬件真实天花板 |
| **特征维切分策略** | 维持完整 $d=128/256$ 维 | **Split-D**: $d_{\text{chunk}} = 64$（`kDChunk=64`） | 消除过大 Tile 导致的片上资源耗尽 |
| **寄存器溢出 (Spilling)** | 严重溢出（Spill 至 Local Memory） | **零溢出（Zero Spilling）** | 单线程累加器规模严格控制，无需 `LDL`/`STL` |
| **共享内存（Smem）占用** | 动态膨胀，超出 64 KB 限额 | 静态锁死在 **45,568 字节（~44.5 KB）** | 留足 L1 Cache 空间，适配 SM70 物理限制 |
| **活跃度 (Occupancy)** | 跌至 25% 左右，延迟掩盖失效 | 提升至较高水准，Warp 轮转充分 | 充分掩盖全局显存访问时延 |
| **GQA KV Cache 访存** | 6 个 Q 分别读显存（重复 6 次） | **GQA-packed**: 行打包 6 头单次读取 | 片上复用提升 600%，节约 83.3% KV 显存带宽 |
| **Tensor Core 指令利用** | 小矩阵发射饥饿，前端单元卡顿 | $M=6 \cdot Q$ 超宽 GEMM 饱满发射 | 充分释放 Volta Tensor Core 硬件峰值吞吐 |

---

## 研讨记录与思考题

- [ ] **思考题 1**：在 GQA 中，多个 Q Head 共享同一个 KV Head。如果每个 Q 分别启动一个 Kernel，会产生哪些重复访存？打包成宽 GEMM 为什么能提高算力利用率？
- [ ] **思考题 2**：在计算 $QK^T$ 时，$K$ 原本按行存储，转置后按列读取。Shared Memory 的 32 个 Bank 会产生何种冲突？工业界通常采用什么排布方式（Padding 或 Swizzle）解决？

---
*源文件位于 `learn_notes/markdown/01_vllm_arch_and_sm70_hardware.md`*
