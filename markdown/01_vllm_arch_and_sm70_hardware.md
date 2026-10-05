# 01 - vLLM 架构全景与 Volta (SM70) 硬件底座

> **模块标签**：`1Cat-vLLM 源码精读` · `第 01 讲` · `CUDA 体系结构` · `SM70 Volta`

---

## 导读与学习目标

本讲作为全系列的第一篇基石，核心目标是**打破黑盒**，建立从大模型高层系统调度（vLLM）到底层 GPU 物理硬件（Tesla V100 / SM70）的全链路全景图：

1. **认知重塑**：理解为什么大模型推理在不同阶段的硬件瓶颈完全相反（Prefill 算力受限 vs Decode 访存受限）。
2. **系统解构**：理清 vLLM 的 PagedAttention 显存分页机制，以及张量元数据如何穿透到 C++/CUDA 算子层。
3. **硬件硬伤剖析**：深入 Volta 架构的底层限制（为什么缺乏现代指令，为什么寄存器和 Shared Memory 极易爆仓）。
4. **数学武器**：掌握 FlashAttention 核心原理与 Online Softmax 的分块递推公式。

---

## 一、为什么我们需要 vLLM？—— 大模型推理的两大阶段

在编写任何 GPU 算子前，必须清晰区分大模型运行的两个阶段：

### 1. Prefill（首字预填充 / 提示词阶段）
* **计算特征**：输入整个 Prompt（例如 2048 个 token），所有 Token 一起并行计算 Q, K, V。
* **瓶颈类型**：**Compute-Bound（算力受限）**。
  - GEMM 矩阵尺寸大（如 $M=2048, K=4096, N=4096$）；
  - 能够充分占满 GPU 的 Tensor Core 阵列。

### 2. Decode（自回归解码 / 逐字生成阶段）
* **计算特征**：每次根据前一个 token 生成下一个 token（即 $M=1$）。为了计算当前单个 Token 与历史所有上下文的注意力关联，必须从显存中完整读取历史所有已生成 Token 的 Key 和 Value（**KV Cache**）。
* **瓶颈类型**：**Memory-Bound（访存带宽受限）**。
  - 浮点计算量极小，但显存搬运量极大。

> [!NOTE] 💡 算力访存比（Arithmetic Intensity）与硬件失衡
>
> $$\text{Arithmetic Intensity} = \frac{\text{计算浮点数 (FLOPs)}}{\text{显存搬运字节数 (Bytes)}}$$
>
> 在单请求 Decode 阶段，算力访存比极低（通常仅 $1 \sim 2 \text{ FLOPs/Byte}$）。
> 而 **Tesla V100** 的 FP16 Tensor Core 理论算力为 **125 TFLOP/s**，HBM2 显存带宽为 **900 GB/s**。
> 计算峰值所要求的饱和算力访存比为：
>
> $$\frac{125 \times 10^{12} \text{ FLOP/s}}{900 \times 10^9 \text{ B/s}} \approx 138.8 \text{ FLOPs/Byte}$$
>
> **结论**：单请求 Decode 阶段 95% 以上的时间 GPU 计算单元都在“空转等待显存读取”，显存带宽利用率直接决定了推理吞吐！

---

## 二、vLLM 核心骨架与数据流转

vLLM 之所以能取得数倍于原生 PyTorch 的吞吐，核心在于解决了显存碎片问题，并构建了 Iteration 级别的调度流水线：

```
1. 调度层 (Scheduler & Continuous Batching)
   └─ 不等待整个批次结束，每个 Iteration 动态调度新请求加入与完成请求释放。
       ↓
2. 显存管理层 (BlockManager & PagedAttention)
   └─ 模仿 OS 分页，显存划分为物理 Block (例如 16 tokens/block)，消除 60% 显存空洞。
       ↓
3. 执行器与元数据装配 (ModelRunner)
   └─ 打包 GPU 输入：input_tokens, block_tables, context_lens, slot_mapping。
       ↓
4. C++/CUDA 桥接层 (csrc/torch_bindings.cpp)
   └─ 通过 Torch Custom Op 零拷贝将张量物理指针送入 CUDA Kernel。
       ↓
5. 架构特化算子层 (Flash-V100 / SM70-TurboMind)
   └─ 执行 Volta 深度定制的 Wide QK/PV 打包、W4A16 软解算子。
```

### Paged 机制对底层 CUDA 寻址的冲击
普通的连续矩阵乘法，地址计算是平坦线性的：
$$\text{Address} = \text{Base} + i \times \text{Stride}$$

而在 PagedAttention 中，读写 KV Cache 变成了**二级跳表寻址**：
1. 逻辑块号：$\text{logical\_block} = t / \text{BlockSize}$
2. 物理块号：$\text{physical\_block} = \text{block\_tables}[\text{req\_id}, \text{logical\_block}]$
3. 块内偏移：$\text{offset} = t \pmod{\text{BlockSize}}$
4. 物理显存地址：$\text{physical\_addr} = \text{Base} + \text{physical\_block} \times \text{BlockBytes} + \text{offset} \times \text{TokenBytes}$

> [!WARNING] 核心挑战
> CUDA 线程若直接按此逻辑随意读取，极易产生非合并访存（Uncoalesced Access）。因此在 Kernel 设计中，必须将同一个物理块内的连续数据以 **128-bit 向量化读写（`float4` / `uint4`）** 的方式协同搬运到 Shared Memory。

---

## 三、Volta 架构（SM70 / Tesla V100）硬件硬核剖析

| 架构特性 | Tesla V100 (Volta / SM70) | A100 (Ampere / SM80) | H100 (Hopper / SM90) |
| :--- | :--- | :--- | :--- |
| **Tensor Core 指令** | `mma.sync.m8n8k4` (极小粒度 HMMA) | `mma.sync.m16n8k16` (算力吞吐倍增) | WGMMA / TMA 硬件单元支持 |
| **低比特硬件支持** | **无原生 FP8 / 无原生 INT4** (仅 FP16/FP32) | 原生 INT8 / INT4 | 原生硬件 FP8 (E4M3 / E5M2) |
| **全局显存 ➔ 共享内存** | **必须经由通用寄存器中转** (极耗寄存器) | `cp.async` 硬件异步直拷 | TMA (Tensor Memory Accelerator) |
| **Shared Memory 架构** | L1 与 Shared Memory 共享 128KB | 最高 164 KB，硬件异步屏障 | 最高 228 KB，集群间共享 (DSM) |

### V100 算子开发的三大“死穴”

1. **寄存器溢出（Register Spilling）**：
   因为缺乏 `cp.async` 指令，显存到共享内存必须走 `Global Memory -> Register -> Shared Memory`。当线程占用寄存器过多（如超过 64~96 个），GPU 的 Occupancy（活跃 Warp 占比）会断崖式下降，无法掩盖流水线延迟。
2. **HMMA 指令发射瓶颈**：
   Volta 的硬件 Tensor Core 每次只能算一个 $8 \times 8 \times 4$ 的极小矩阵。为了计算一个 $64 \times 64 \times 16$ 的 Block Tile，需要发射海量的 HMMA 汇编指令，容易让指令发射单元（Instruction Issue Unit）跑满阻塞。
3. **缺乏低比特硬件单元**：
   现代量化大模型（如 NVFP4 / W4A16 / FP8）无法直接调用硬件 Tensor Core。必须在软件层面通过**位移、掩码运算在寄存器内解包为 FP16**，再输入 HMMA 执行。

---

## 四、Attention 核心演进：从朴素实现到 Online Softmax

标准 Attention 公式：
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### 1. 朴素实现的 $O(N^2)$ 显存灾难
若依次执行 $S = QK^T \to P = \text{Softmax}(S) \to O = PV$：
- 必须将中间矩阵 $S \in \mathbb{R}^{N \times N}$ 和 $P \in \mathbb{R}^{N \times N}$ 完整写回 HBM 显存再读出；
- 显存占用与显存带宽往返都是 $O(N^2)$。当 $N=64\text{K}$ 时，单矩阵就占数十 GB，直接 OOM。

### 2. FlashAttention 的关键：Online Softmax（在线增量更新）
FlashAttention 将 $Q, K, V$ 分块载入片上 **Shared Memory**，全程不向 HBM 写出中间大矩阵。
核心数学技巧在于：**在只看到部分 $K$ 向量时，如何增量维护全局 Softmax 归一化？**

> [!IMPORTANT] 📐 Online Softmax 递推公式
> 设前一段的局部最大值为 $m^{(1)}$，指数和为 $l^{(1)}$，部分输出为 $O^{(1)}$；
> 新进入的一个块局部最大值为 $m^{(2)}$，局部指数和为 $l^{(2)}$：
>
> 1. **更新全局最大值**：
>    $$m^{\text{new}} = \max\left(m^{(1)}, m^{(2)}\right)$$
>
> 2. **更新归一化分母（指数和）**：
>    $$l^{\text{new}} = e^{m^{(1)} - m^{\text{new}}} \cdot l^{(1)} + e^{m^{(2)} - m^{\text{new}}} \cdot l^{(2)}$$
>
> 3. **输出向量增量修正**：
>    $$O^{\text{new}} = \frac{e^{m^{(1)} - m^{\text{new}}} \cdot l^{(1)} \cdot O^{(1)} + e^{m^{(2)} - m^{\text{new}}} \cdot \left(P^{(2)} V^{(2)}\right)}{l^{\text{new}}}$$

通过这种在片上寄存器里“滚动缩放”的方式，算力访存比大幅提高，显存读写复杂度降至 $O(N)$。

---

## 五、1Cat-vLLM 在 V100 上的破局：PR #286 的核心线索

官方旧版 FlashAttention 在 V100 上由于没有针对 Volta 硬件微架构深度调优，吞吐长期停留在 **17.92 TFLOP/s**（不足硬件峰值的 15%）。

1Cat-vLLM 在 **PR #286** 中完成了关键蜕变：
- **Split-D**：将较大的 Head Dimension 切分成适合 Volta 共享内存与寄存器预算的小 Tile；
- **GQA-packed Wide QK/PV GEMM**：将 6 个共享同一组 KV 的 Query Heads 打包成一个更宽的 GEMM，大幅降低指令发射开销并提高 Tensor Core 计算密度，一举将吞吐提升至 **≈60.8 TFLOP/s**！

---

## ✍️ 我的个人笔记 / 思考记录插槽

> *提示：你可以在此处直接添加你阅读过程中的心得、疑问或试验记录，重新构建后将自动呈现在护眼 HTML 中。*

- [ ] **思考题 1**：在 GQA 中，多个 Q Head 共享同一个 KV Head。如果每个 Q 分别起一个 Kernel，会造成哪些重复访存？
- [ ] **思考题 2**：在计算 $QK^T$ 时，$K$ 在内存中原本是以行存储的，转置后按列读取。Shared Memory 的 32 个 Bank 会发生什么？如何通过 Padding 或 Swizzle 消除？

---
*本文档为 1Cat-vLLM 系列学习笔记 · 源文件位于 `learn_notes/markdown/01_vllm_arch_and_sm70_hardware.md`*
