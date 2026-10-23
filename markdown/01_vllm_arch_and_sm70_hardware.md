---
series: 1Cat-vLLM 源码精读中文讲义 · 核心系统与硬件架构实战
title: Lecture 01: vLLM 架构全景与 Volta (SM70) 硬件底座
subtitle: 从 Roofline 访存边界、PagedAttention 二级寻址到 SM70 Split-D 算子突围的系统化理解
authors: 五道口纳什风格 & Antigravity
date: 2026-10-23
repo: https://github.com/1CatAI/1Cat-vLLM (Make Volta Fast Again)
hardware: NVIDIA Tesla V100 (Volta / SM70, 125 TFLOP/s FP16 Tensor Core, 900 GB/s HBM2)
source_files: csrc/attention/sm70_v37/tail.cu · csrc/attention/sm70_v37/prefill.cu · csrc/torch_bindings.cpp
topics: Prefill vs. Decode 瓶颈反转 · PagedAttention 页表寻址 · Online Softmax 代数约分 · SM70 Split-D 与 GQA-Packed 宽矩阵
---

## 1 导论：为什么在 Volta 上跑快 vLLM 是一道系统工程题

这讲的核心目标有两个。第一，把“大模型推理到底卡在算力还是卡在显存搬运”这件事从物理第一性原理讲透，建立从高层调度器（vLLM Continuous Batching & PagedAttention）到底层 CUDA 线程束（Warp）寻址的完整因果链。第二，回答一个非常具体的工程问题：**为什么面向 A100 (SM80) 设计的现代注意力算子直接搬到 Tesla V100 (SM70) 上会暴跌到 17.92 TFLOP/s（仅峰值算力的 14.3%），而 `1Cat-vLLM` 又是如何通过 Split-D 与 GQA-Packed 宽矩阵重新冲上 60.8~75+ TFLOP/s 的？**

> [!IMPORTANT] 本讲的总问题
> 大模型推理算子优化从来不是孤立地“背诵 CUDA 语法”，而是同时解开三道耦合约束：算法层面如何用 **Online Softmax** 消除 $O(N^2)$ 全局显存往返；系统层面如何让 **PagedAttention** 的离散页表寻址仍然保持 128-bit 合并访存；硬件层面如何在 **Volta SM70** 严苛的 64 KB 共享内存与寄存器上限下彻底消灭 Local Memory 溢出（Register Spilling）。

### 1.1 大模型推理的两大阶段与性能瓶颈反转

在编写任何一行 CUDA Kernel 之前，必须先看清大模型自回归推理的两个截然不同的物理阶段：

1. **Prefill（首字预填充 / 提示词阶段）**：用户输入完整 Prompt（例如 $M = 2048$ 个 Token），所有 Token 并行计算 $Q, K, V$ 投影与自注意力。此时 GEMM 矩阵行维度庞大（如 $M=2048, K=4096, N=4096$），权重与中间块在片上被上千个 Token 高度复用，属于典型的 **Compute-Bound（算力受限）** 阶段，核心矛盾是如何喂饱 GPU 的 Tensor Core。
2. **Decode（自回归解码 / 逐字生成阶段）**：每次仅根据刚生成的 1 个 Token 预测下一个 Token（即 $M = 1$）。但为了计算这 1 个 Token 与历史所有上下文的注意力关联，GPU 必须把历史上已生成的所有 Token 的 Key 和 Value（即 **KV Cache**）从板载显存（HBM2）完整搬运到片上 SRAM。此时浮点运算量极小，而显存搬运量随序列长度线性膨胀，属于极端的 **Memory-Bound（访存带宽受限）** 阶段。

为了定量判断一个算子究竟受限于算力还是带宽，我们需要引入 Roofline 模型。算子实际可达到的硬件吞吐上限满足以下关系：

$$\text{Performance} = \min\left(P_{\max},\; I \cdot B_{\text{HBM}}\right), \qquad I = \frac{\text{FLOPs}}{\text{Bytes}_{\text{moved}}}$$

符号解释如下：
1. $P_{\max}$：硬件的理论峰值浮点计算吞吐（对 Tesla V100 FP16 Tensor Core 而言，$P_{\max} = 125\text{ TFLOP/s}$）。
2. $B_{\text{HBM}}$：硬件的全局显存可用带宽（对 Tesla V100 HBM2 而言，$B_{\text{HBM}} = 900\text{ GB/s}$）。
3. $I$：算术强度（Arithmetic Intensity），即从全局显存每搬运 1 字节（Byte）数据所能支撑的浮点运算次数（FLOPs）。
4. $I^* = P_{\max} / B_{\text{HBM}}$：硬件的计算饱和临界点（Ridge Point）。

将 Tesla V100 的物理参数代入临界点公式，可以得到计算单元满载所需的最低算术强度：

$$I^*_{\text{V100}} = \frac{125 \times 10^{12}\text{ FLOP/s}}{900 \times 10^9\text{ Byte/s}} \approx 138.9\text{ FLOPs/Byte}$$

> [!KNOWLEDGE] Roofline 拐点背后的直觉
> $I^*_{\text{V100}} \approx 138.9\text{ FLOPs/Byte}$ 意味着：你从 HBM2 显存里每读出 1 个 FP16 元素（2 字节），必须在片上寄存器或 Shared Memory 里对它完成至少 **278 次浮点乘加运算**，V100 的 Tensor Core 才不会因为等数据而闲置。而在单请求 Decode 阶段（$M=1$），算术强度 $I$ 往往只有 $1 \sim 2\text{ FLOPs/Byte}$，不到饱和阈值的 $1.5\%$——此时再强的算力单元也只是在空转等待 HBM2 总线传输。

![Roofline 算术强度与 PagedAttention 二级页表寻址全景图|源码定位：csrc/attention/sm70_v37/tail.cu:L340--L351 (paged_kv_thread_offset)](assets/fig01_roofline_and_paging.svg)

### 1.2 本章小结

Prefill 与 Decode 的物理瓶颈完全相反：Prefill 拼的是如何通过宽矩阵分块把 Tensor Core 算力吃满；Decode 拼的是如何压缩 KV Cache 体积并榨干每一字节的 HBM2 显存带宽。理解了这一对矛盾，后续 vLLM 的分页管理与算子融合才有了明确的靶心。

---

## 2 vLLM 系统骨架与 PagedAttention 二级寻址机制

为什么传统推理框架在长上下文高并发下会迅速 OOM（显存溢出）？因为在自回归生成前，系统无法预知每个请求最终会输出多少个 Token。如果按最大长度（如 $8192$）为每个请求静态预分配一整块连续显存，绝大部分预留槽位都会沦为无法被其他请求借用的内部碎片（Internal Fragmentation）。

vLLM 的核心突破是借用了操作系统虚拟内存分页（Paging）的思想，将连续的逻辑 KV Cache 拆散为固定大小的物理块（Physical Blocks，例如每页 `page_size = 16` 个 Token），由 `BlockManager` 按需动态分配。整个系统的数据流转分为五层：

1. **调度层（Scheduler & Continuous Batching）**：在每个 Iteration 边界动态剔除已结束请求、插入新到达请求，消除请求级等待空泡。
2. **显存管理层（BlockManager & PagedAttention）**：维护每个序列的逻辑块号到物理页号的映射表 `block_table`，彻底消除显存外部碎片。
3. **执行器与元数据装配（ModelRunner）**：将当前 Batch 的 `input_tokens`、`block_tables`、`context_lens`、`slot_mapping` 打包为连续 GPU 张量。
4. **C++/CUDA 桥接层（`csrc/torch_bindings.cpp`）**：通过 PyTorch Custom Op 零拷贝提取底层裸指针并分发至架构特化算子。
5. **架构特化算子层（`csrc/attention/sm70_v37/`）**：在 CUDA Kernel 内部一边查页表完成二级跳表寻址，一边执行分块矩阵乘与 Online Softmax。

### 2.1 从线性寻址到二级跳表：Paged 机制如何改变 CUDA 访存

在传统连续显存中，访问第 `logical_row` 个 Token 的 KV 向量只需一次乘加线性偏移。而在 PagedAttention 中，逻辑上相邻的 Token 在物理显存池中不再连续，算子必须通过 `block_table` 完成二级地址翻译。

给定逻辑 Token 行号 $t$（即代码中的 `logical_row`）、物理页容量 $B_{\text{page}}$（即 `page_size`）以及特征维切块号 $c$（`d_chunk`），单个线程读取起始元素的物理地址偏移满足以下映射公式：

$$\text{Offset}(t, c, \text{tid}) = \text{ BlockTable}\!\left[\left\lfloor \frac{t}{B_{\text{page}}} \right\rfloor\right] \cdot S_{\text{page}} + \left(t \bmod B_{\text{page}}\right) \cdot S_{\text{row}} + c \cdot d_{\text{chunk}} + \left(\text{tid} \bmod T_{\text{row}}\right) \cdot E_{\text{load}}$$

符号解释如下：
1. $t$：当前读取的逻辑 Token 序列位置（`logical_row`）。
2. $B_{\text{page}}$：单个物理页容纳的 Token 槽位数（`page_size`，典型值为 $16$）。
3. $\text{BlockTable}[\lfloor t / B_{\text{page}} \rfloor]$：通过整除算出逻辑页号后，查表得到的物理页索引 `physical_page`。
4. $S_{\text{page}}, S_{\text{row}}$：分别为单个物理页的跨度（`page_stride`）与页内单行 Token 的跨度（`row_stride`）。
5. $c \cdot d_{\text{chunk}}$：Split-D 切分下当前第 $c$ 个特征维子块的列起始偏移（每块宽 $64$）。
6. $(\text{tid} \bmod T_{\text{row}}) \cdot E_{\text{load}}$：同行协作线程组内，当前线程负责的向量化加载列偏移。

我们以一个具体算例来验证（见图 1 下半部分）：假设 `page_size = 16`，当前要读取第 $t = 35$ 号 Token，且该请求的映射表为 `block_table = [7, 12, 3, 9]`：
- **第 1 步（算逻辑页号）**：$35 \mathbin{//} 16 = 2$，命中第 $2$ 号逻辑页；
- **第 2 步（查表得物理页号）**：`block_table[2] = 3`，说明数据实际存放在显存池的第 $3$ 号物理块；
- **第 3 步（算页内行槽位）**：$35 \bmod 16 = 3$，对应第 $3$ 号物理页内部的第 $3$ 行。

### 2.2 真实源码印证：`tail.cu` 中的向量化分页寻址

下面这段代码来自 `1Cat-vLLM` 的核心算子文件 [tail.cu](file:///D:/1Cat-vLLM/csrc/attention/sm70_v37/tail.cu#L339-L351)，展示了 Kernel 如何在分页显存池上精确计算每个线程的 128-bit 向量化读取基址：

```cuda | caption=PagedAttention 二级跳表与线程内向量化列偏移计算 | source=csrc/attention/sm70_v37/tail.cu:L339--L351
template <int kThreadsPerRow, int kRowsPerThread, int kElemsPerLoad>
__device__ __forceinline__ int64_t
paged_kv_thread_offset(int tid, int n_block, int d_chunk, int page_size,
                       const int* __restrict__ block_table, int64_t page_stride,
                       int64_t row_stride) {
  const int row_in_tile = (tid / kThreadsPerRow) * kRowsPerThread;
  const int logical_row = n_block * Sm70D256SplitDTraits::kBlockN + row_in_tile;
  const int physical_page = block_table[logical_row / page_size];
  return static_cast<int64_t>(physical_page) * page_stride +
         static_cast<int64_t>(logical_row % page_size) * row_stride +
         d_chunk * Sm70D256SplitDTraits::kDChunk +
         (tid % kThreadsPerRow) * kElemsPerLoad;
}
```

这段代码最精妙的地方不在于除法和取模本身，而在于最后一项 `(tid % kThreadsPerRow) * kElemsPerLoad`：当 `kDChunk = 64`、`kElemsPerLoad = 8`（即单次指令加载 8 个 FP16 = 128 bit）时，`kThreadsPerRow = 8` 个连续编号的线程拥有完全相同的 `logical_row` 和 `physical_page`，它们在同一时钟周期携手横扫同一行连续的 64 个半精度元素，从而在离散分页的物理池上完美维持了底层 DRAM Burst 合并访问！

> [!WARNING] 一个非常常见的坑：分页不等于纵容单线程随机乱跳
> 很多初学者以为引入 PagedAttention 后，每个 CUDA 线程去独立负责不同 Token 的读取也无所谓。这是极其致命的误区！如果同一个 Warp 内的 32 个线程各自跳往不同的 `physical_page`，硬件内存控制器将被迫把一次访问拆成 32 笔零碎的 32-Byte 事务，有效带宽瞬间跌穿 $1/10$。**「页间允许离散跳表，页内同行必须由相邻线程以 128-bit (`SM70_LDG_GLOBAL_CG_128b`) 锁步合并搬运」** 是 Paged 算子设计的铁律。

### 2.3 本章小结

PagedAttention 用一张极小的整数页表 `block_table` 终结了 KV Cache 的显存碎片；而在算子实现侧，只有把页表查找绑定在行块（Row Tile）粒度、并让组内连续线程对齐在特征维连续地址上，才能在不损失 HBM2 合并带宽的前提下享受分页带来的高并发红利。

---

## 3 Attention 计算演进：从 $O(N^2)$ 显存墙到 Online Softmax 约分消解

即便用 PagedAttention 管住了 KV Cache，朴素注意力计算本身仍然存在一道致命的显存墙。标准缩放点积注意力定义为：

$$\mathbf{O} = \text{Softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V} = \mathbf{P}\mathbf{V}$$

符号解释如下：
1. $\mathbf{Q} \in \mathbb{R}^{N \times d}$：Query 查询矩阵，$N$ 为序列长度，$d$ 为头维度。
2. $\mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$：Key 键矩阵与 Value 值矩阵。
3. $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top / \sqrt{d_k} \in \mathbb{R}^{N \times N}$：未归一化的注意力打分矩阵。
4. $\mathbf{P} = \text{Softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N}$：按行归一化后的注意力概率权重矩阵。
5. $\mathbf{O} \in \mathbb{R}^{N \times d}$：最终注意力输出矩阵。

如果按朴素方式分三步依次启动 Kernel（先算 $\mathbf{S}=\mathbf{Q}\mathbf{K}^\top$ 写回 HBM，再读出 $\mathbf{S}$ 算 $\mathbf{P}=\text{Softmax}(\mathbf{S})$ 写回 HBM，最后读出 $\mathbf{P}$ 乘 $\mathbf{V}$），中间矩阵 $\mathbf{S}$ 和 $\mathbf{P}$ 的显存占用与 HBM 读写流量全部是 $O(N^2)$ 级别。

### 3.1 为什么 Safe Softmax 看上去“天生排斥分块”？

为了防止指数运算 $\exp(x_i)$ 发生 FP16/FP32 数值上溢（Overflow），工程中必须先减去整行的全局最大值 $m$（即 Safe Softmax）。对于长度为 $N$ 的单行打分向量 $x \in \mathbb{R}^N$，传统计算必须顺序扫描全行三遍：
1. **第 1 遍（求全行最大值）**：$m = \max_{1 \le i \le N} x_i$；
2. **第 2 遍（求平移后指数和）**：$l = \sum_{i=1}^N e^{x_i - m}$；
3. **第 3 遍（求归一化加权输出）**：$O = \sum_{i=1}^N \frac{e^{x_i - m}}{l} V_i$。

痛点浮出水面：在流式分块（Tiling）加载 $\mathbf{K}, \mathbf{V}$ 时，我们在处理第 1 个块 $B_1$ 时根本不知道后续块 $B_2$ 里会不会出现更大的打分！如果非要等全部 $N$ 列都算完才能动手算 $\mathbf{P}\mathbf{V}$，算子融合就无从谈起。

### 3.2 核心数学推导：两块数据的 Online Softmax 增量递推

FlashAttention 与 Online Softmax 的破局点在于：**既然旧块的最大值 $m^{(1)}$ 只是暂时的，那么当新块 $B_2$ 到来并刷新出更大的全局最大值 $m^{\text{new}}$ 时，我们只需要乘上一个指数修正因子 $e^{m^{(1)} - m^{\text{new}}}$，就能把旧块已经算好的统计量在片上寄存器里原地“对齐”到新基准上！**

设序列被切分为两个子块 $x = [x^{(1)}, x^{(2)}]$，其中 $x^{(1)} \in \mathbb{R}^{B_1}$，$x^{(2)} \in \mathbb{R}^{B_2}$。在单独处理每个子块时，各自以局部最大值定义局部统计量与未归一化加权分子 $\tilde{\mathbf{O}}$：

$$m^{(k)} = \max_{i \in B_k} x_i^{(k)}, \qquad l^{(k)} = \sum_{i \in B_k} e^{x_i^{(k)} - m^{(k)}}, \qquad \tilde{\mathbf{O}}^{(k)} = \sum_{i \in B_k} e^{x_i^{(k)} - m^{(k)}} \mathbf{V}_i^{(k)} = l^{(k)} \mathbf{O}^{(k)} \quad (k \in \{1, 2\})$$

符号解释如下：
1. $m^{(k)}$：子块 $B_k$ 内部的局部行最大值。
2. $l^{(k)}$：子块 $B_k$ 以自身 $m^{(k)}$ 为平移基准的局部指数和分母。
3. $\tilde{\mathbf{O}}^{(k)}$：子块 $B_k$ 未除以分母 $l^{(k)}$ 的局部加权和分子（注意 $\tilde{\mathbf{O}}^{(k)} = l^{(k)} \mathbf{O}^{(k)}$）。

当需要将两个子块合并为全局结果时，利用指数恒等式 $e^{a - c} = e^{a - b} \cdot e^{b - c}$，全局最大值 $m^{\text{new}}$、全局分母 $l^{\text{new}}$ 以及全局输出 $\mathbf{O}^{\text{new}}$ 满足以下严格等价的一步递推关系：

$$m^{\text{new}} = \max\!\left(m^{(1)}, m^{(2)}\right), \qquad l^{\text{new}} = e^{m^{(1)} - m^{\text{new}}} l^{(1)} + e^{m^{(2)} - m^{\text{new}}} l^{(2)}$$

$$\mathbf{O}^{\text{new}} = \frac{e^{m^{(1)} - m^{\text{new}}} \tilde{\mathbf{O}}^{(1)} + e^{m^{(2)} - m^{\text{new}}} \tilde{\mathbf{O}}^{(2)}}{l^{\text{new}}} = \frac{e^{m^{(1)} - m^{\text{new}}} l^{(1)} \mathbf{O}^{(1)} + e^{m^{(2)} - m^{\text{new}}} l^{(2)} \mathbf{O}^{(2)}}{l^{\text{new}}}$$

符号解释如下：
1. $m^{\text{new}}$：合并两个子块后的新全局最大值。
2. $e^{m^{(k)} - m^{\text{new}}}$：把旧基准 $m^{(k)}$ 下的统计量重缩放到新基准 $m^{\text{new}}$ 下的衰减因子。
3. $l^{\text{new}}, \mathbf{O}^{\text{new}}$：合并后严格等于全量 Safe Softmax 的新分母与新输出向量。

> [!IMPORTANT] Online Softmax 数值稳定性的第一性原理
> 由于 $m^{\text{new}} = \max(m^{(1)}, m^{(2)})$，必然有 $m^{(1)} - m^{\text{new}} \le 0$ 且 $m^{(2)} - m^{\text{new}} \le 0$。这意味着两个重缩放因子永远落在区间 $(0, 1]$ 内！**整个在线递推过程中，指数函数内部永远是非正数，从数学根源上杜绝了任何中间态数值上溢的可能。**

### 3.3 深入剖析：为什么工程代码中看不到「先除 $l^{(2)}$ 再乘 $l^{(2)}$」？

很多读者在对照论文公式与 CUDA 源码时会产生一个强烈困惑：论文递推式分子里写着 $l^{(2)} \cdot \mathbf{O}^{(2)}$，为什么翻遍 `FlashAttention` 和 `1Cat-vLLM` 的源码，在内层循环里既看不到除以 $l^{(2)}$，也看不到乘以 $l^{(2)}$？

把局部归一化输出的定义 $\mathbf{O}^{(2)} = \tilde{\mathbf{P}}^{(2)}\mathbf{V}^{(2)} / l^{(2)}$（其中 $\tilde{\mathbf{P}}^{(2)} = \exp(\mathbf{S}^{(2)} - m^{\text{new}})$）直接代入合并分子，即可看清代数约分的本质：

$$e^{m^{(2)} - m^{\text{new}}} \cdot l^{(2)} \cdot \mathbf{O}^{(2)} = e^{m^{(2)} - m^{\text{new}}} \cdot \cancel{l^{(2)}} \cdot \left(\frac{\exp(\mathbf{S}^{(2)} - m^{(2)})\mathbf{V}^{(2)}}{\cancel{l^{(2)}}}\right) = \exp\!\left(\mathbf{S}^{(2)} - m^{\text{new}}\right)\mathbf{V}^{(2)}$$

符号解释如下：
1. $\cancel{l^{(2)}}$：局部归一化分母与外层还原权重在代数上完全抵消。
2. $\exp(\mathbf{S}^{(2)} - m^{\text{new}})\mathbf{V}^{(2)}$：直接用刷新后的当前最大值 $m^{\text{new}}$ 平移当前块打分 $\mathbf{S}^{(2)}$，做完指数后直接送入 Tensor Core 与 $\mathbf{V}^{(2)}$ 相乘累加到未归一化分子寄存器 $\tilde{\mathbf{O}}$ 中。

![FlashAttention Online Softmax 分块递推与浮点除法约分消解机制|源码定位：csrc/attention/sm70_v37/tail.cu:L255--L337 (splitd_n32_online_softmax)](assets/fig02_online_softmax_flow.svg)

> [!KNOWLEDGE] 代数约分带来的硬件级收益：保护 Tensor Core 流水线免受 SFU 除法停顿
> - **Tensor Core 原生不支持除法**：在 NVIDIA GPU 中，矩阵乘累加由高吞吐 Tensor Core 执行，而浮点除法（`DIV`）或倒数（`RCP`）必须交由特殊函数单元（SFU / ALU）逐个标量串行处理，延迟高达数十个周期。
> - **消除全部中间块除法**：通过在寄存器中直接维护**未归一化分子 $\tilde{\mathbf{O}}$**（即代码中的 `o_storage`），内层循环只需对旧累加器做一次标量乘法 `acc_o *= row_scale`，再由 Tensor Core 直接把 $\tilde{\mathbf{P}}^{(j)}\mathbf{V}^{(j)}$ 累加进去；**直到整条 KV 序列的所有分块全部遍历完毕、准备写回 HBM 前的最后一刻，才执行唯一一次除以最终分母 `row_sum` 的操作！**

### 3.4 本章小结

Online Softmax 的精髓不是“近似”，而是**用代数等价的重缩放（Rescaling）把三遍全局扫描折叠为一遍流式递推**，并利用分子分母的代数约分把昂贵的浮点除法推迟到 Kernel 尾声，从而让 $\mathbf{S}$ 和 $\mathbf{P}$ 彻底留在片上。

---

## 4 1Cat-vLLM 在 Volta (SM70) 上的算子突围：结合真实源码精读

理解了通用 FlashAttention 原理后，我们进入全讲最硬核的实战部分：**为什么把标准 FlashAttention-2 搬到 Tesla V100 (Volta / SM70) 上会遭遇严重算力坍塌（仅 17.92 TFLOP/s），而 `1Cat-vLLM` 又是如何通过芯片级协同设计将其拉升至 60.8~75+ TFLOP/s 的？**

### 4.1 痛点根因：SM70 的三大物理硬约束与 Occupancy 坍塌

现代主流开源大模型（如 LLaMA-3、Qwen-2.5）广泛采用 $d_{\text{head}} = 128$ 甚至 $d_{\text{head}} = 256$ 的大头维设计。我们先对比 Tesla V100 (SM70) 与 A100 (SM80) / H100 (SM90) 的硬件代差：

| 硬件架构维度 | Tesla V100 (Volta / SM70) | A100 (Ampere / SM80) | H100 (Hopper / SM90) |
| :--- | :--- | :--- | :--- |
| **Tensor Core 基础指令原子** | `mma.sync.m8n8k4`（小粒度 Quadpair） | `mma.sync.m16n8k16`（吞吐翻倍） | `wgmma.mma_async`（Warp Group 级） |
| **全局显存 → 共享内存通路** | **无异步拷贝，必须借用通用寄存器中转** | `cp.async` 硬件异步直达 Smem | TMA 专用张量内存引擎 |
| **单 SM 共享内存上限** | L1 与 Smem 共享 128 KB（Smem 最高 64/96 KB） | 最高 164 KB | 最高 228 KB + DSM 跨块直通 |
| **单 SM 寄存器堆容量** | 65,536 × 32-bit（单线程硬上限 255） | 65,536 × 32-bit（配合 `cp.async` 省寄存器） | 65,536 × 32-bit（支持寄存器重分配） |

当头维度达到 $d_{\text{head}} = 256$ 时，标准 FlashAttention 在 V100 上立刻撞上三堵死墙：
1. **缺少 `cp.async` 引发的数据中转寄存器常驻**：在 V100 上，任何从 HBM2 搬往 Shared Memory 的数据都必须走 `LDG.E.128 -> 通用寄存器 (RMEM) -> STS.128` 路径，预取缓冲张量本身就要吃掉大量寄存器。
2. **输出累加器 $\tilde{\mathbf{O}}$ 撑爆单线程寄存器上限**：若单 Warp 不加切分地维护全部 $d=256$ 列的 FP32 输出累加器，加上 $QK^\top$ 累加器与中转寄存器，单线程需求瞬间突破 128~255 个寄存器。
3. **Local Memory Spilling 与活跃度（Occupancy）雪崩**：

$$\text{Blocks per SM} = \left\lfloor \frac{65536}{T_{\text{block}} \times R_{\text{thread}}} \right\rfloor = \left\lfloor \frac{65536}{256 \times 128} \right\rfloor = 2\text{ Blocks} \implies \text{Occupancy} = \frac{16\text{ Warps}}{64\text{ Warps}} = 25\%$$

符号解释如下：
1. $65536$：V100 单个 SM 拥有的 32-bit 物理寄存器总数。
2. $T_{\text{block}} = 256$：单个线程块（CTA）包含的线程数（8 个 Warp）。
3. $R_{\text{thread}} = 128$：高维累加器下单个线程占用的寄存器数。
4. $\text{Occupancy} = 25\%$：单 SM 活跃 Warp 数（16）占硬件上限（64）的比例。

> [!WARNING] 致命陷阱：所谓 Local Memory 在物理上就是高延迟的 HBM2 显存！
> 千万不要被“Local Memory（本地内存）”这个名字骗了！在 NVIDIA GPU 物理芯片上，**根本不存在独立的片上 Local Memory 硬件，它纯粹是板载 HBM2 全局显存的一段私有映射区**。一旦单线程寄存器超限触发 NVCC 编译器的 **Register Spilling**，原本 1~4 个周期的寄存器读写就会退化为 200~400 个周期的 `LDL` / `STL` 显存指令；叠加 25% 的极低活跃度无法掩盖延迟，V100 的 Tensor Core 绝大部分时间都在饥饿停顿，吞吐直接坍塌到 17.92 TFLOP/s。

---

### 4.2 核心优化一：Split-D（头维度正交切分）与双 Warp 协同流水线

既然把 $d = 256$ 整个塞进单个 Warp 的寄存器堆会引发溢出，那么最直接的第一性原理破局就是：**把特征维 $d=256$ 切成 4 个 $d_{\text{chunk}} = 64$ 的子块（Split-D），并让每对协作 Warp（`kWarpsPerGroup = 2`）在 QK 阶段和 PV 阶段采取正交的分工策略！**

![1Cat-vLLM SM70 Split-D 双 Warp 正交分工与 45,568B Smem 内存布局|源码定位：csrc/attention/sm70_v37/tail.cu:L28--L128, L255--L337, L495--L620](assets/fig03_sm70_splitd_pipeline.svg)

我们先阅读 [tail.cu](file:///D:/1Cat-vLLM/csrc/attention/sm70_v37/tail.cu#L28-L56) 中定义全核几何骨架的 `Sm70D256SplitDTraits` 结构体：

```cuda | caption=Sm70D256SplitDTraits 核心分块、双 Warp 协作与 MMA 映射定义 | source=csrc/attention/sm70_v37/tail.cu:L28--L56
struct Sm70D256SplitDTraits {
  using Element = cutlass::half_t;
  using MmaAtom = MMA_Atom<SM70_8x8x4_F32F16F16F32_TN>;
  using PvMmaAtom = MMA_Atom<SM70_8x8x4_F32F16F16F32_TT>;

  static constexpr int kHeadDim = 256;
  static constexpr int kBlockM = 64;
  static constexpr int kBlockN = 32;
  static constexpr int kDChunk = 64;
  static constexpr int kDChunks = kHeadDim / kDChunk;
  static constexpr int kOwnedDChunks = kDChunks / 2;
  static constexpr int kNThreads = 256;
  static constexpr int kMmaThreads = 32;
  static constexpr int kWarpsPerGroup = 2;
  static constexpr int kMmaGroups = kNThreads / (kWarpsPerGroup * kMmaThreads);
  static constexpr int kGroupRows = kBlockM / kMmaGroups;
  static constexpr int kQkWarpRows = kGroupRows / kWarpsPerGroup;
  static constexpr int kQkRowsPerThread = kQkWarpRows / 4;
  static constexpr int kOutputRowsPerThread = kGroupRows / 4;

  // Each warp in a pair owns eight distinct Q rows for QK, then the pair
  // shares the resulting P tile and each warp owns D/2 for PV. This keeps
  // the standard FA2 N32 online-softmax order without duplicating QK work.
  using QkTiledMma = TiledMMA<MmaAtom, Layout<Shape<_1, _4, _1>>,
                              Tile<Int<kQkWarpRows>, Int<kBlockN / 4>, _4>>;
  using PvTiledMma = TiledMMA<PvMmaAtom, Layout<Shape<_1, _4, _1>>,
                              Tile<Int<kGroupRows>, Int<kDChunk / 4>, _4>>;
};
```

请特别留意第 21~23 行的英文源码注释，它道破了 `1Cat-vLLM` 整个 Split-D 算子最核心的**「双 Warp 正交分工机制」**（见图 3 的 `(a)` 与 `(b)`）：
- **全局线程块划分**：整个 CTA 有 `kNThreads = 256` 个线程（8 个 Warp），被划分为 `kMmaGroups = 4` 个独立协作组，每组包含 `kWarpsPerGroup = 2` 个 Warp（记为 Warp 0 和 Warp 1），共同负责 `kGroupRows = 16` 行 Query。
- **阶段一（QK 阶段——按行切分，零重复计算）**：如果让 Warp 0 和 Warp 1 各算一半 $d$，最后还得在 Shared Memory 里规约整个 $16 \times 32$ 的打分矩阵 $\mathbf{S}$。为了彻底避免规约开销，**在 QK 阶段，Warp 0 独占前 8 行（`kQkWarpRows = 8`），Warp 1 独占后 8 行**，两个 Warp 各自循环遍历全部 4 个 `d_chunk = 64` 累加出完整的 8 行打分 `acc_s`，并各自完成这 8 行的 Online Softmax！此时单线程仅需维护 `kQkRowsPerThread = 2` 行的 `row_max` 与 `row_sum`。
- **阶段二（跨 Warp 同步桥——`sP` 与 `row_scale_exchange`）**：进入 PV 阶段时，为了把输出累加器 `o_storage` 的寄存器砍半，不能再按行切分了，必须改为**按特征维 $D$ 对半切分**：**Warp 0 负责全部 16 行的前一半特征维（`d_chunk` $0, 1$，共 128 维），Warp 1 负责全部 16 行的后一半特征维（`d_chunk` $2, 3$，共 128 维）**。
  但这里有一个关键问题：Warp 0 手里只有前 8 行的 Online Softmax 缩放因子 $\alpha = 2^{(m_{\text{old}} - m_{\text{new}})\cdot s}$，它怎么去缩放自己持有的后 8 行输出累加器 `o_storage`？

答案就在 [tail.cu](file:///D:/1Cat-vLLM/csrc/attention/sm70_v37/tail.cu#L295-L325) 的 `splitd_n32_online_softmax` 函数中：

```cuda | caption=双 Warp 通过 row_scale_exchange 交换行衰减因子并重缩放各自 D/2 的输出累加器 | source=csrc/attention/sm70_v37/tail.cu:L295--L325
  if ((lane & 0x0e) == 0) {
#pragma unroll
    for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
      const int row =
          FLASH_NAMESPACE::sm70_row_slot<Traits::kQkWarpRows>(slot, lane);
      row_scale_exchange[mma_group * Traits::kGroupRows +
                         n_warp * Traits::kQkWarpRows + row] = work[slot];
    }
  }
  __syncthreads();

  if (!first_tile) {
#pragma unroll
    for (int d = 0; d < Traits::kOwnedDChunks; ++d) {
      auto acc_o = make_tensor(make_rmem_ptr(&o_storage[d][0]), o_layout);
      auto acc_o_rc = make_tensor(
          acc_o.data(),
          FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
#pragma unroll
      for (int row = 0; row < Traits::kOutputRowsPerThread; ++row) {
        const int logical_row =
            FLASH_NAMESPACE::sm70_row_slot<Traits::kGroupRows>(row, lane);
        const float row_scale =
            row_scale_exchange[mma_group * Traits::kGroupRows + logical_row];
#pragma unroll
        for (int col = 0; col < size<1>(acc_o_rc); ++col) {
          acc_o_rc(row, col) *= row_scale;
        }
      }
    }
  }
```

这段代码清晰地展示了：**每个 Warp 把自己算出的 8 行衰减系数 `work[slot]` 写入仅占 512 字节的共享内存缓冲区 `row_scale_exchange`，经过一次 `__syncthreads()` 后，两个 Warp 就能读取全部 16 行的 `row_scale` 来更新自己持有的 `kOwnedDChunks = 2` 个输出分块！**

---

### 4.3 极致压榨 45,568 字节 Shared Memory：生命周期别名与 Quadpair 消冲突

在解决了寄存器溢出后，还要面对 V100 单 SM 仅 64 KB / 96 KB 共享内存的硬约束。我们来看 [tail.cu](file:///D:/1Cat-vLLM/csrc/attention/sm70_v37/tail.cu#L64-L128) 中对 Shared Memory 布局与容量的静态锁死：

```cuda | caption=Volta HMMA.884 无冲突 SmemLayoutK 与 45,568 字节静态共享内存断言 | source=csrc/attention/sm70_v37/tail.cu:L64--L128
  // Volta HMMA.884 assigns one warp to four 8-thread quadpairs and services
  // the 64-bit operand loads as two half-warps. Pitch 68 advances each row
  // by one bank pair; the extra 16-half phase every 16 rows folds row bit 4
  // into bank-pair bit 2 so every half-warp covers all 16 bank pairs once.
  using SmemLayoutK = Layout<
      Shape<Shape<Int<16>, Int<2>>, Int<kDChunk>>,
      Stride<Stride<Int<kDChunk + 4>, Int<16 * (kDChunk + 4) + 16>>, _1>>;

  static constexpr int kQElements = size(SmemLayoutQ{});
  static constexpr int kKVElements = size(SmemLayoutKV{});
  // The second K stage is live only before P is materialized, so it may
  // alias the beginning of the later P region. Keep the K stages disjoint
  // and on the same 128-byte bank phase without moving the V/P regions.
  static constexpr int kKStageElements = 2240;
  static constexpr int kPElements = size(SmemLayoutP{});
  static constexpr int kExchangeRows = kMmaGroups * kGroupRows;
  static constexpr int kTensorSmemBytes =
      (kQElements + 2 * kKVElements + kPElements) * sizeof(Element);
  static constexpr int kExchangeBytes = 2 * kExchangeRows * sizeof(float);
  static constexpr int kSmemBytes = kTensorSmemBytes + kExchangeBytes;
  static_assert(kSmemBytes == 45568);
```

这里藏着两个极其精妙的硬件级技巧（见图 3 `(c)`）：
1. **针对 Volta `HMMA.884` Quadpair 的 `Pitch = 68` 消 Bank 冲突设计**：Volta 的 `mma.sync.m8n8k4` 指令将 1 个 Warp 拆成 4 个 8 线程组（Quadpair），并以两个 Half-warp 发起 64-bit 共享内存读取。若直接按行宽 `kDChunk = 64` 存储，不同行会落在相同的 Bank Pair 上引发严重冲突。代码将行跨度设为 `kDChunk + 4 = 68`（每行错开 1 个 Bank Pair），并在每 16 行处再额外错开 `+16` 个 `half`，使得单次 Half-warp 访存恰好无碰撞地均匀铺满全部 16 个 Bank Pair！
2. **基于存活期（Liveness）的共享内存别名复用（Aliasing）**：为了给 `Pitch = 68` 的双缓冲 `sK` 留出 Padding 空间（每 Stage 需 $2240$ 个 `half`，两个 Stage 共 $4480$ 个 `half`），按理说 `2 * kKVElements = 4096` 个 `half` 是不够放的。但作者敏锐地注意到：**第 2 个 K Stage 仅在 QK 循环内、即概率矩阵 `sP` 被写出之前存活！** 因此直接让第 2 个 K Stage 越界借用紧随其后的 `sP` 起始区域（`kKStageElements + cosize(SmemLayoutK{}) <= 2 * kKVElements + size(SmemLayoutP{})`），分文不花额外内存，将总 Smem 精确锁死在 **45,568 字节（44.5 KB）**，为 V100 留出宝贵的 ~19.5 KB 硬件 L1 Cache！

> [!IMPORTANT] Split-D 架构优化的核心工程收益
> 通过「QK 按行 8/8 分工 + `row_scale_exchange` + PV 按列 128/128 分工 + 44.5 KB 静态共享内存锁死」，单个线程的寄存器占用被稳稳压制在 64~80 个以内，实现了**零 Local Memory 溢出（Zero Spilling）**，并将 V100 单 SM 的并发线程与缓存命中率推向最佳甜点区。

---

### 4.4 核心优化二：GQA-Packed Wide GEMM（行维度 6 头打包）

解决了单个算子内部的片上资源分配后，我们还面临主流大模型 GQA（Grouped-Query Attention）带来的第二个瓶颈：当 6 个或 8 个 Query 头共享 1 个 KV 头时，如果每个 Query 头独立运算，不仅同一份 KV Cache 要被从 HBM2 重复读取 6 次，而且单个头的行维度 $M$ 太窄，喂不饱 Volta 的 Tensor Core 流水线。

![GQA-Packed 宽矩阵打包算子与 MLA 低秩矩阵吸收机制对比|源码定位：csrc/attention/sm70_v37/prefill.cu:L455--L480, L628--L652 (sm70_d256_gqa_v37_fwd)](assets/fig04_gqa_wide_gemm_and_mla.svg)

`1Cat-vLLM` 在 [prefill.cu](file:///D:/1Cat-vLLM/csrc/attention/sm70_v37/prefill.cu#L455-L480) 与 L628--L652 中给出了極めて干脆的解法——**将共享同一个 KV 头的 $G = 6$ 个 Query 头在行维度（$M$ 轴）直接拼接为高度 `kRows = 6 * kQuery` 的超宽矩阵，单次启动 CUTLASS GEMM**：

```cuda | caption=GQA 6 头 Query 沿行维度打包执行单次宽矩阵 QK 与 PV 乘累加 | source=csrc/attention/sm70_v37/prefill.cu:L455--L466, L628--L649
at::Tensor sm70_d256_gqa_v37_fwd(const at::Tensor& q, const at::Tensor& k,
                                 const at::Tensor& v, at::Tensor& out,
                                 double softmax_scale, bool causal) {
  constexpr int kHeadDim = 256;
  constexpr int kHeadsQ = 6;     // 6 个 Query 头
  constexpr int kHeadsKV = 1;    // 共享 1 个 KV 头

  const int kQuery = static_cast<int>(q.size(1));
  // 将 6 个 Query 头在行维度直接折叠打包: M = 6 * kQuery
  const int kRows = kQuery * kHeadsQ;
  ...
  // 单次宽矩阵 QK GEMM: 形状为 (6 * Q, width) = (6 * Q, 256) x (256, width)
  typename QKGemm::Arguments arguments(
      {kRows, width, kHeadDim}, 1,
      {query, QKLayoutA(kHeadDim)},                         // 打包后的 6 头 Query
      {key + size_t(begin) * kHeadDim, QKLayoutB(kHeadDim)}, // 单组共享 Key
      {score_ptr, typename QKGemm::LayoutC(width)}, ...);
  operation.qk(stream);

  // 单次宽矩阵 PV GEMM: 同样以 kRows = 6 * Q 执行宽乘累加
  operation.pv = std::make_unique<PVLauncher>(
      score_ptr, value + size_t(begin) * kHeadDim, prefix_accumulator_ptr,
      kRows, width, old_scale_ptr, block_scale_ptr, block == 0);
  operation.pv->launch(stream);
}
```

这段代码的执行路径直击 Roofline 公式的两个核心变量（见图 4 `(a)`）：
1. **分母 `Bytes moved` 缩减至 $1/6$**：共享的 `key` 和 `value` 块从 HBM2 读入片上一次，立即被打包在一起的 6 个 Query 头共同消费，KV 访存复用率暴涨 $600\%$。
2. **分子有效计算密度打满**：矩阵行高从 $Q$ 扩展为 $6Q$，彻底消除了小矩阵在 Volta `mma.sync.m8n8k4` 下的指令发射饥饿与 Wave Quantization 尾块浪费。

### 4.5 优化前后全景对比与本章小结

| 评估维度 | 原生基线实现 (Naive FA on SM70) | 1Cat-vLLM 架构特化实现 (`sm70_v37`) | 底层硬件与编译收益机制 |
| :--- | :--- | :--- | :--- |
| **实测计算吞吐** | **17.92 TFLOP/s** (~14.3% 峰值) | **60.8 ~ 75+ TFLOP/s** | 吞吐跃升 **3.4× ~ 4.2×**，逼近 V100 真实物理上限 |
| **特征维切分策略** | 单 Warp 硬扛完整 $d=256$ | **Split-D 正交分工** (`kDChunk=64`) | QK 按行切 8/8，PV 按列切 128/128，零重复计算 |
| **寄存器溢出 (Spilling)** | 严重溢出至 HBM2 Local Memory | **0 Bytes Spill (零溢出)** | 彻底消除 200+ 周期的 `LDL` / `STL` 停顿指令 |
| **Shared Memory 预算** | 动态膨胀逼近或超过 64 KB | 静态锁死 **45,568 Bytes (44.5 KB)** | 借用 `sP` 生命周期别名复用，保住 19.5 KB L1 Cache |
| **Bank Conflict 控制** | 64 列直排引发 Half-warp 冲突 | `Pitch=68` + 16 行相位折叠 | 契合 `HMMA.884` Quadpair，覆盖全部 16 个 Bank Pair |
| **GQA KV 访存与 GEMM** | 6 个 Q 头各自读 KV、窄矩阵发射 | **GQA-Packed** (`kRows = 6 * Q`) | KV 带宽立省 $83.3\%$，填满 CUTLASS Tensor Core 流水线 |

#### 本章小结

`1Cat-vLLM` 在 V100 上的逆袭证明了一个硬道理：**没有落后的架构，只有错配的数据流**。当硬件没有 `cp.async` 且共享内存受限时，强行照搬 A100 的大 Tile 必然死于寄存器溢出；唯有顺应 `HMMA.884` 的 Quadpair 访存规律、用 Split-D 正交分工把寄存器与 Smem 控制在硬件甜点区内，才能让老架构重新焕发惊人算力。

---

## 5 总结与延伸

### 5.1 核心脉络回看

回顾整篇讲义，我们将大模型推理系统与底层 CUDA 算子串成了一条完整的四级因果链：
1. **先用 Roofline 定性阶段瓶颈**：区分 Prefill 的算力受限（$M \gg 1$）与 Decode 的显存带宽受限（$M = 1, I \ll 138.9$）。
2. **再用 PagedAttention 解决系统显存碎片**：通过 `block_table` 实现逻辑块到物理页的二级寻址，并在页内保持 8 线程 128-bit 向量化合并访问。
3. **接着用 Online Softmax 打通算子融合**：通过指数平移恒等式实现单遍流式递推，并利用 $l^{(2)}$ 的代数约分消除全部中间块的 SFU 浮点除法停顿。
4. **最后用 Split-D 与 GQA-Packed 贴合 SM70 硬件**：通过双 Warp 正交分工、512B `row_scale_exchange`、45,568B 别名复用以及 $6\text{Q}$ 行打包，将 V100 实测吞吐从 17.92 TFLOP/s 推升至 60.8~75+ TFLOP/s。

> [!IMPORTANT] 全讲的最高层结论
> 优秀的 GPU 系统与算子工程，本质上都是在做同一件事：**用更聪明的数据流组织（分页映射、在线递推、正交切块、多头打包），去迎合硬件存储层次的物理脾气，把昂贵的外层 HBM2 访问与流水线停顿压缩到极致。**

### 5.2 开放问题与下一步探索

1. **从 `sm70_v37` (60.8 TFLOP/s) 到 `sm70_79t` (75+ TFLOP/s) 的进一步演进**：在 Prefill 极长序列与因果掩码（Causal Mask）交界处的对角线 Tile 上，如何进一步剥离全满块（Dense Prefix）与因果尾块（Causal Tail）以减少分支判断与掩码开销？
2. **低比特量化（W4A16 / KV Cache INT8/FP8）在无原生指令集硬件上的软解**：Volta 缺少原生 INT4/FP8 Tensor Core 指令，如何在寄存器内通过位运算（LOP3 / PRMT）高效软解包为 FP16 并与 HMMA 双缓冲流水线完全重叠？

### 5.3 拓展阅读

1. **1Cat-vLLM 源码仓库**：[https://github.com/1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)（重点阅读 `csrc/attention/sm70_v37/tail.cu` 与 `prefill.cu`）
2. **FlashAttention & FlashAttention-2 原论文**：Tri Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (NeurIPS 2022) & *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023)
3. **vLLM PagedAttention 原论文**：Woosuk Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023)
