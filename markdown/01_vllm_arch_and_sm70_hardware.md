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

同样提取缩放系数：
$$\sum_{i \in B_1} e^{x_i^{(1)} - m^{\text{new}}} V_i^{(1)} = e^{m^{(1)} - m^{\text{new}}} \sum_{i \in B_1} e^{x_i^{(1)} - m^{(1)}} V_i^{(1)} = e^{m^{(1)} - m^{\text{new}}} \cdot \tilde{O}^{(1)} = e^{m^{(1)} - m^{\text{new}}} \cdot l^{(1)} \cdot O^{(1)}$$

$$\sum_{j \in B_2} e^{x_j^{(2)} - m^{\text{new}}} V_j^{(2)} = e^{m^{(2)} - m^{\text{new}}} \sum_{j \in B_2} e^{x_j^{(2)} - m^{(2)}} V_j^{(2)} = e^{m^{(2)} - m^{\text{new}}} \cdot \left( P^{(2)} V^{(2)} \right) \cdot l^{(2)}$$

将分子代入归一化分母 $l^{\text{new}}$，得出**最终输出增量递推公式**：
$$O^{\text{new}} = \frac{e^{m^{(1)} - m^{\text{new}}} \cdot l^{(1)} \cdot O^{(1)} + e^{m^{(2)} - m^{\text{new}}} \cdot \left( P^{(2)} V^{(2)} \right) \cdot l^{(2)}}{l^{\text{new}}}$$

若记局部块的加权得分 $P^{(2)} = \frac{e^{x^{(2)} - m^{(2)}}}{l^{(2)}}$，则 $P^{(2)} V^{(2)} \cdot l^{(2)} = \sum_{j \in B_2} e^{x_j^{(2)} - m^{(2)}} V_j^{(2)}$，公式亦可紧凑写为：
$$O^{\text{new}} = \frac{e^{m^{(1)} - m^{\text{new}}} \cdot l^{(1)} \cdot O^{(1)} + e^{m^{(2)} - m^{\text{new}}} \cdot \sum_{j \in B_2} e^{x_j^{(2)} - m^{(2)}} V_j^{(2)}}{l^{\text{new}}}$$

或者更直观地写成“**旧结果衰减更新 + 新结果加权补入**”的凸组合形式：
$$O^{\text{new}} = \left( \frac{l^{(1)} \cdot e^{m^{(1)} - m^{\text{new}}}}{l^{\text{new}}} \right) \cdot O^{(1)} + \left( \frac{l^{(2)} \cdot e^{m^{(2)} - m^{\text{new}}}}{l^{\text{new}}} \right) \cdot O^{(2)}$$

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

## 五、1Cat-vLLM 在 V100 上的优化方向（PR #286）

针对早期实现在 V100 上吞吐偏低的问题（约 17.92 TFLOP/s），1Cat-vLLM 在 **PR #286** 中引入了两项核心改进：
- **Split-D**：将较大的 Head Dimension 切分成适合 Volta 共享内存与寄存器容量的小 Tile；
- **GQA-packed Wide QK/PV GEMM**：将 6 个共享同一组 KV 的 Query Heads 打包成一个更宽的 GEMM，降低指令发射开销并提高 Tensor Core 计算密度，将实测吞吐提升至 **≈60.8 TFLOP/s**。

---

## 研讨记录与思考题

- [ ] **思考题 1**：在 GQA 中，多个 Q Head 共享同一个 KV Head。如果每个 Q 分别启动一个 Kernel，会产生哪些重复访存？
- [ ] **思考题 2**：在计算 $QK^T$ 时，$K$ 原本按行存储，转置后按列读取。Shared Memory 的 32 个 Bank 会产生何种冲突？工业界通常采用什么排布方式（Padding 或 Swizzle）解决？

---
*源文件位于 `learn_notes/markdown/01_vllm_arch_and_sm70_hardware.md`*
