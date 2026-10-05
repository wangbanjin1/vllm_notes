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
连续矩阵乘法的地址计算是线性的：
$$\text{Address} = \text{Base} + i \times \text{Stride}$$

而在 PagedAttention 中，读写 KV Cache 变为**二级跳表寻址**：
1. 计算逻辑块号：$\text{logical\_block} = t / \text{BlockSize}$
2. 查表获取物理块号：$\text{physical\_block} = \text{block\_tables}[\text{req\_id}, \text{logical\_block}]$
3. 计算块内偏移：$\text{offset} = t \pmod{\text{BlockSize}}$
4. 计算实际显存地址：$\text{physical\_addr} = \text{Base} + \text{physical\_block} \times \text{BlockBytes} + \text{offset} \times \text{TokenBytes}$

> [!WARNING] 访存合并要求
> 如果线程直接执行上述散乱寻址，会导致严重的非合并访存（Uncoalesced Access）。因此在 Kernel 设计中，需要通过线程协同，将一个物理块内的数据通过 **128-bit 向量化读写（`float4` / `uint4`）** 搬运至 Shared Memory。

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
FlashAttention 将 $Q, K, V$ 分块载入片上 **Shared Memory**，避免向全局显存写出中间矩阵。
核心计算基于 **Online Softmax**：在仅加载局部 $K$ 向量时，增量维护全局 Softmax 归一化。

> [!IMPORTANT] Online Softmax 递推公式
> 设前一段的局部最大值为 $m^{(1)}$，指数和为 $l^{(1)}$，部分输出为 $O^{(1)}$；
> 新输入块的局部最大值为 $m^{(2)}$，局部指数和为 $l^{(2)}$：
>
> 1. **更新全局最大值**：
>    $$m^{\text{new}} = \max\left(m^{(1)}, m^{(2)}\right)$$
>
> 2. **更新归一化分母（指数和）**：
>    $$l^{\text{new}} = e^{m^{(1)} - m^{\text{new}}} \cdot l^{(1)} + e^{m^{(2)} - m^{\text{new}}} \cdot l^{(2)}$$
>
> 3. **输出向量增量修正**：
>    $$O^{\text{new}} = \frac{e^{m^{(1)} - m^{\text{new}}} \cdot l^{(1)} \cdot O^{(1)} + e^{m^{(2)} - m^{\text{new}}} \cdot \left(P^{(2)} V^{(2)} \right)}{l^{\text{new}}}$$

通过在片上寄存器中滚动累加并动态缩放，显存访问复杂度降低至 $O(N)$。

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
