# 01 - vLLM 架构全景与 Volta (SM70) 硬件底座

> **模块标签**：`1Cat-vLLM 源码精读` · `第 01 讲` · `CUDA 体系结构` · `SM70 Volta`

---

## 导读与核心目标

写高性能 GPU 算子，光看 Python 层的调度代码或者光背硬件参数都没用，必须把整条链路打通：从上层的显存分页（vLLM 怎么把上下文切成离散 Page），到底层硬件的物理限制（老架构 V100 怎么抠显存带宽和寄存器），再到数学算法的演进（FlashAttention 为什么能把显存压到 $O(N)$）。

本讲重点搞清楚这 4 件事：
1. **瓶颈定性**：彻底弄明白 Prefill 为什么卡算力、Decode 为什么卡带宽。
2. **分页寻址**：搞清楚 vLLM 的 PagedAttention 怎么从两级跳表算出真正的显存物理地址。
3. **老卡枷锁**：摸清 Volta (SM70) 的硬件短板（没异步拷贝、寄存器和共享内存极其吃紧）。
4. **数学底座**：吃透 FlashAttention 与 Online Softmax 的分块递推推导，以及工程实现里的约分细节。

---

## 一、大模型推理的两大阶段：到底卡在哪里？

写算子或调优性能前，第一件事是定性：当前算子到底卡在计算（Compute-Bound），还是卡在显存带宽（Memory-Bound）。大模型推理分为两个截然不同的阶段：

### 1. Prefill（首字预填充 / Prompt 阶段）
* **计算特点**：把用户输入的完整 Prompt（比如 2048 个 token）一口气喂给模型，所有 Token 并行算 Q, K, V。
* **瓶颈类型**：**Compute-Bound（算力受限）**。
  - 矩阵乘规模很大（典型如 $M=2048, K=4096, N=4096$）；
  - 能充分喂饱 GPU 的 Tensor Core 计算单元，这时候主要看峰值 TFLOP/s。

### 2. Decode（自回归解码 / 逐字生成阶段）
* **计算特点**：逐字吐词，每次输入只有一个 token（即 $M=1$）。但为了计算这个新 token 与前面所有历史上下文的注意力，必须把历史所有 token 的 Key 和 Value（**KV Cache**）从显存里一字不差地全部搬出来。
* **瓶颈类型**：**Memory-Bound（显存带宽受限）**。
  - 浮点计算量其实很小，但显存数据搬运量极大。

> [!NOTE] 算力访存比（Arithmetic Intensity）算笔明白账
>
> $$\text{Arithmetic Intensity} = \frac{\text{计算浮点数 (FLOPs)}}{\text{显存搬运字节数 (Bytes)}}$$
>
> 单请求在 Decode 阶段，算力访存比极低（通常只有 $1 \sim 2 \text{ FLOPs/Byte}$）。
> 我们以 **Tesla V100** 为例：
> - FP16 Tensor Core 理论算力：**125 TFLOP/s**
> - HBM2 显存带宽：**900 GB/s**
> 
> 要让计算单元刚好饱和，系统需要的临界算力访存比为：
>
> $$\frac{125 \times 10^{12} \text{ FLOP/s}}{900 \times 10^9 \text{ B/s}} \approx 138.8 \text{ FLOPs/Byte}$$
>
> **现实很残酷**：138.8 对比 1~2，意味着单请求 Decode 时，GPU 的计算单元 98% 以上的时间都在干等数据从显存搬过来。这时候谁能把显存带宽利用率榨干，谁的系统吞吐就高。

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

> [!WARNING] 访存合并的底层硬要求
> 很多初学者容易在这里写出“每个线程算一个 Token，各自去读非对齐地址”的代码，这会直接引发灾难性的非合并访存（Uncoalesced Access），显存带宽瞬间跌到不足 10%。
> **正确解法**：算子要解耦“搬运分工”与“计算分工”。外层循环锁定当前这一个物理块，块内 32 个线程全部化身“搬砖工”，排好队用 **128-bit 向量化读写（`float4` / `uint4`，单条指令搬 16 字节）** 协同把一整块连续内存一口气搬到 Shared Memory，随后各线程再在片上自由取用计算。

---

## 三、老卡 Volta (SM70 / Tesla V100) 的硬件枷锁

在现代 A100 / H100 上写算子有各种硬件指令兜底，但在 V100 这类老架构上写，到处都是坑。先对比一下代际差异：

| 架构特性 | Tesla V100 (Volta / SM70) | A100 (Ampere / SM80) | H100 (Hopper / SM90) |
| :--- | :--- | :--- | :--- |
| **Tensor Core 指令** | `mma.sync.m8n8k4` (小粒度 HMMA) | `mma.sync.m16n8k16` (吞吐翻倍) | WGMMA / TMA 硬件单元支持 |
| **低比特硬件支持** | **无原生 FP8 / 无原生 INT4** (仅 FP16/FP32) | 原生 INT8 / INT4 | 原生硬件 FP8 (E4M3 / E5M2) |
| **显存到共享内存** | **必须经过通用寄存器中转** (消耗寄存器) | `cp.async` 硬件异步直拷 | TMA 硬件级异步张量加速 |
| **Shared Memory 容量** | L1 与 Shared Memory 共享 128KB | 最高 164 KB，硬件异步屏障 | 最高 228 KB，集群间共享 (DSM) |

### 为什么在 V100 上写算子这么难受？

1. **没有异步直拷，寄存器容易溢出（Register Spilling）**：
   A100 之后一条 `cp.async` 就能直接让硬件把数据从显存拷进共享内存，不走通用寄存器。而 V100 必须走 `Global Memory -> 通用寄存器 -> Shared Memory`。线程里的每个寄存器都价值连城，一旦寄存器用超了，数据就会溢出到极慢的 Local Memory，活跃 Warp 骤降，延迟根本掩盖不住。
2. **HMMA 粒度太小，指令发射容易卡死前端**：
   Volta Tensor Core 硬件单次只能算 $8 \times 8 \times 4$ 矩阵。想要算完一个常规大小的矩阵块（Tile），CUDA 编译器必须发射密密麻麻的大量汇编指令，往往还没跑满算力，先撞到了 SM 的指令发射瓶颈。
3. **完全没有低比特硬件单元，全靠软件手搓解包**：
   要在 V100 上跑 W4A16 或者 FP8 量化模型，硬件根本没有对应的乘法指令。必须在 CUDA 代码里先用位运算手写解包（Unpack），在寄存器里动态转成 FP16，再塞给 Tensor Core 跑。

---

## 四、Attention 计算演进：从朴素实现到 Online Softmax

标准 Attention 公式大家都会背：
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### 1. 为什么朴素实现会吃爆显存？
传统深度学习框架的算子是按层顺序执行的：
$$S = QK^T \;\to\; P = \text{Softmax}(S) \;\to\; O = PV$$
- 每一个中间矩阵 $S$ 和 $P$ 的尺寸都是 $N \times N$。
- 这些矩阵必须老老实实写回显存（HBM），下一步再读出来。
- 显存空间占用与访存带宽往返都是 $O(N^2)$。当 Prompt 长度达到 8k、32k 时，还没开始算，显存就已经被撑爆了（OOM）。

### 2. FlashAttention 原理：Online Softmax 详细推导

FlashAttention 的破局思路很纯粹：**绝不把中间结果写回显存，全部在片上算完**。
但片上 Shared Memory 只有几十 KB，一次只能装下一小截 $K, V$。而为了数值安全防止溢出的 **Safe Softmax**，传统上需要看完全部数据：
1. 第一遍扫全量：找到全局最大值 $m = \max_{1 \le i \le N} x_i$；
2. 第二遍扫全量：求出归一化分母 $l = \sum_{i=1}^N e^{x_i - m}$；
3. 第三遍算概率：$P_i = \frac{e^{x_i - m}}{l}$ 并与 $V$ 做加权乘加。

**核心矛盾**：既然不能提前看完所有 Token，怎么在只看到眼前这小块数据时，就算出全局正确的结果？这就是 **Online Softmax** 的数学精髓。

---

#### 核心推导：两块数据的 Softmax 增量拼接

设序列被切分为两段（Block 1 代表已处理的历史累积，Block 2 代表刚载入的新块）：
$$x = \left[ x^{(1)}, x^{(2)} \right], \quad x^{(1)} \in \mathbb{R}^{B_1}, \; x^{(2)} \in \mathbb{R}^{B_2}$$

##### 步骤 1：两块各自的局部统计量（严格对称）
* **Block 1（历史累加态）**：
  * 局部最大值：$m^{(1)} = \max_{i \in B_1} x_i^{(1)}$
  * 局部指数分母：$l^{(1)} = \sum_{i \in B_1} e^{x_i^{(1)} - m^{(1)}}$
  * 局部归一化输出：$O^{(1)} = \sum_{i \in B_1} \frac{e^{x_i^{(1)} - m^{(1)}}}{l^{(1)}} V_i^{(1)}$
  * 未归一化分子为：$\tilde{O}^{(1)} = l^{(1)} \cdot O^{(1)}$
* **Block 2（新块数据）**：
  * 局部最大值：$m^{(2)} = \max_{j \in B_2} x_j^{(2)}$
  * 局部指数分母：$l^{(2)} = \sum_{j \in B_2} e^{x_j^{(2)} - m^{(2)}}$
  * 局部归一化输出：$O^{(2)} = \sum_{j \in B_2} \frac{e^{x_j^{(2)} - m^{(2)}}}{l^{(2)}} V_j^{(2)}$
  * 未归一化分子为：$\tilde{O}^{(2)} = l^{(2)} \cdot O^{(2)}$

##### 步骤 2：全局最大值更新
合体之后的真实最大值显而易见：
$$m^{\text{new}} = \max\left(m^{(1)}, m^{(2)}\right)$$

##### 步骤 3：归一化分母（指数和）的增量修正
全局正确的总分母，是以 $m^{\text{new}}$ 为基准的全部指数和：
$$l^{\text{new}} = \sum_{k \in B_1 \cup B_2} e^{x_k - m^{\text{new}}} = \sum_{i \in B_1} e^{x_i^{(1)} - m^{\text{new}}} + \sum_{j \in B_2} e^{x_j^{(2)} - m^{\text{new}}}$$

拆分指数项 $x_i - m^{\text{new}} = (x_i - m^{(1)}) + (m^{(1)} - m^{\text{new}})$，提公因式：
$$\sum_{i \in B_1} e^{x_i^{(1)} - m^{\text{new}}} = e^{m^{(1)} - m^{\text{new}}} \cdot \underbrace{\sum_{i \in B_1} e^{x_i^{(1)} - m^{(1)}}}_{l^{(1)}} = e^{m^{(1)} - m^{\text{new}}} \cdot l^{(1)}$$

同理，对 Block 2 提取缩放公因式：
$$\sum_{j \in B_2} e^{x_j^{(2)} - m^{\text{new}}} = e^{m^{(2)} - m^{\text{new}}} \cdot \underbrace{\sum_{j \in B_2} e^{x_j^{(2)} - m^{(2)}}}_{l^{(2)}} = e^{m^{(2)} - m^{\text{new}}} \cdot l^{(2)}$$

代回即可得到**分母更新递推公式**：
$$l^{\text{new}} = e^{m^{(1)} - m^{\text{new}}} \cdot l^{(1)} + e^{m^{(2)} - m^{\text{new}}} \cdot l^{(2)}$$

> **为什么数值绝对安全？**
> 因为 $m^{\text{new}} \ge m^{(1)}$ 且 $m^{\text{new}} \ge m^{(2)}$，所以指数差值 $m^{(1)} - m^{\text{new}} \le 0$。衰减系数 $e^{\Delta m} \in (0, 1]$，只做衰减不做放大，**绝对不会发生浮点上溢**。

##### 步骤 4：注意力输出向量 $O$ 的在线递推
最终全局加权输出为两部分分子求和后再除以总分母：
$$O^{\text{new}} = \frac{\sum_{i \in B_1} e^{x_i^{(1)} - m^{\text{new}}} V_i^{(1)} + \sum_{j \in B_2} e^{x_j^{(2)} - m^{\text{new}}} V_j^{(2)}}{l^{\text{new}}}$$

两块提取缩放系数的过程在数学上完全对称：
* **Block 1 分子缩放**：$e^{m^{(1)} - m^{\text{new}}} \cdot \left( l^{(1)} \cdot O^{(1)} \right)$
* **Block 2 分子缩放**：$e^{m^{(2)} - m^{\text{new}}} \cdot \left( l^{(2)} \cdot O^{(2)} \right)$

代入分母，即得**完全对称的输出增量递推公式**：
$$O^{\text{new}} = \frac{e^{m^{(1)} - m^{\text{new}}} \cdot l^{(1)} \cdot O^{(1)} \;+\; e^{m^{(2)} - m^{\text{new}}} \cdot l^{(2)} \cdot O^{(2)}}{l^{\text{new}}}$$

直观地看，这就是**“旧结果衰减更新 + 新结果加权补入”**的凸组合：
$$O^{\text{new}} = \left( \frac{l^{(1)} \cdot e^{m^{(1)} - m^{\text{new}}}}{l^{\text{new}}} \right) \cdot O^{(1)} \;+\; \left( \frac{l^{(2)} \cdot e^{m^{(2)} - m^{\text{new}}}}{l^{\text{new}}} \right) \cdot O^{(2)}$$

---

#### 深入剖析：为什么工程代码里可以省去除法？

如果你看底层 CUDA 源码，会发现一个“奇怪”的现象：公式里明明写着 $l^{(2)} \cdot O^{(2)}$，为什么真实代码里既看不到除以 $l^{(2)}$，也看不到乘以 $l^{(2)}$？

##### 1. 数学上的“直接约分”
根据定义：
* **未归一化的局部矩阵乘积**：定义 $\tilde{P}^{(2)} = \exp\left(Q (K^{(2)})^\top - m^{(2)}\right)$，其与 $V^{(2)}$ 的矩阵乘积就是原始分子：
  $$\text{未归一化分子} = \tilde{P}^{(2)} V^{(2)} = \sum_{j \in B_2} e^{x_j^{(2)} - m^{(2)}} V_j^{(2)}$$
* **局部归一化输出**：就是分子除以分母：
  $$O^{(2)} = \frac{\tilde{P}^{(2)} V^{(2)}}{l^{(2)}}$$

如果机械地代入公式：
$$l^{(2)} \cdot O^{(2)} = l^{(2)} \cdot \left( \frac{\tilde{P}^{(2)} V^{(2)}}{l^{(2)}} \right) = \tilde{P}^{(2)} V^{(2)}$$

**看到了吗？分母上的 $l^{(2)}$ 和外层的 $l^{(2)}$ 在数学上直接被抵消掉了！**  
因此，新块需要累加的分子项，纯粹就是：
$$e^{m^{(2)} - m^{\text{new}}} \cdot \left( \tilde{P}^{(2)} V^{(2)} \right)$$

##### 2. 硬件层面的关键收益（GPU 芯片特性）
在纸面上约分只是一笔划掉的事，但在 GPU 芯片上，这一步价值千金：
* **Tensor Core 只会做矩阵乘**：GPU 里的 Tensor Core 跑矩阵乘（MMA）极快，一枪发射就能算完 $\tilde{P}^{(2)} V^{(2)}$。
* **浮点除法（DIV）在 GPU 上极慢**：Tensor Core **原生不支持除法**！除法必须打断硬件流水线，交给通用算术单元（ALU/SFU）按标量串行做，延迟高达数十周期。
* **流水线对比**：
  * **机械低效做法**：Tensor Core 算完 $\tilde{P}^{(2)} V^{(2)} \to$ 打断流水线做慢速除法算 $O^{(2)} \to$ 再乘回 $l^{(2)} \to$ 累加；
  * **工程优化做法**：既然分子就是 $\tilde{P}^{(2)} V^{(2)}$，直接把 Tensor Core 的矩阵乘输出乘以缩放因子加进寄存器，**全程连一次多余除法都不用做**！

---

#### 硬件流水线与寄存器状态机

在 CUDA Kernel 运行期间，每个线程无需向显存申请任何缓冲区，全靠片上几个**通用寄存器**当作流式累加器：
* 标量 $m \in \mathbb{R}$（当前累加最大值，初值设为 $-\infty$）；
* 标量 $l \in \mathbb{R}$（当前累加指数和，初值设为 $0$）；
* 向量 $O \in \mathbb{R}^d$（当前累加加权结果，初值设为 $\vec{0}$）。

```text
片上通用寄存器 (维持当前流式状态):
   ┌─────────┐   ┌─────────┐   ┌─────────────────┐
   │  m = -∞ │   │  l = 0  │   │     O = 0       │
   └─────────┘   └─────────┘   └─────────────────┘
        │             │                 │
        ▼ 外层循环：依次搬运下一个 Physical Block 的 K, V
   ┌─────────────────────────────────────────────┐
   │ 1. GEMM-1:  S_tile = Q * K_tile^T           │
   │ 2. Max/Sum: 计算该 Tile 的局部 m_tile, l_tile│
   │ 3. Rescale: 按公式就地更新 m, l 寄存器       │
   │ 4. GEMM-2:  O 寄存器累加 P_tile * V_tile    │
   └─────────────────────────────────────────────┘
        │
   循环结束 ──> 寄存器中的 O 即为严格等于标准 Attention 的最终结果！
```

**两大核心收益**：
1. **显存访问复杂度直降**：中间 $N \times N$ 的打分矩阵 $S$ 和概率矩阵 $P$ 彻底不在显存落盘，访存复杂度从 $O(N^2)$ 压到 $O(N)$；
2. **片上算子完全融合（Fused Kernel）**：GEMM-1 ($QK^T$)、Softmax 和 GEMM-2 ($PV$) 在单次 Kernel 执行内流水线跑完。

---

## 五、1Cat-vLLM 是怎么在 V100 上榨出 60 TFLOP/s 的？（PR #286）

针对早期实现在 V100 上吞吐偏低的问题（当时只有 17.92 TFLOP/s，远没喂饱 125 TFLOP/s 的硬件能力），1Cat-vLLM 在 **PR #286** 中引入了两项核心优化：
- **Split-D**：较大的 Head Dimension 容易撑爆 Volta 的 Shared Memory 与寄存器，通过将其切分成适合 SM70 容量的小 Tile 流水线推进；
- **GQA-packed Wide QK/PV GEMM**：GQA 结构下多个 Query Head 共享同一组 KV。早期实现分开跑产生了大量重复读 KV 开销。优化后将 6 个 Q Head 打包成一个更宽的 GEMM 一起算，大幅降低了指令发射频率并抬高了 Tensor Core 计算密度，最终将实测吞吐提升到了 **≈60.8 TFLOP/s**（提升超过 3 倍）。

---

## 研讨记录与思考题

- [ ] **思考题 1**：在 GQA 中，多个 Q Head 共享同一个 KV Head。如果每个 Q 分别启动一个 Kernel，会产生哪些重复访存？打包成宽 GEMM 为什么能提高算力利用率？
- [ ] **思考题 2**：在计算 $QK^T$ 时，$K$ 原本按行存储，转置后按列读取。Shared Memory 的 32 个 Bank 会产生何种冲突？工业界通常采用什么排布方式（Padding 或 Swizzle）解决？

---
*源文件位于 `learn_notes/markdown/01_vllm_arch_and_sm70_hardware.md`*
