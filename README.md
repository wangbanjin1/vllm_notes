# vLLM 与 1Cat-vLLM 进阶学习笔记

本项目记录针对 [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)（*Make Volta Fast Again*）以及 vLLM 体系架构的高性能算子优化、CUDA 体系结构与系统调度的学习笔记。

每一讲包含：
* **Markdown 源文件**：位于 `markdown/`，支持编辑、批注与二次开发。
* **护眼 HTML 报告**：位于 `html/`，配备「暖阳米纸 / 薄暮夜读」低蓝光护眼主题、MathJax 公式渲染及夜间模式切换。
* **一键编译脚本**：使用纯 Python 标准库编写的 `build_notes.py`，支持修改 Markdown 后全自动重新生成 HTML 与导航索引。

---

## 目录结构

```text
vllm_notes/
├── markdown/                 # 核心 Markdown 源文件
│   └── 01_vllm_arch_and_sm70_hardware.md
├── html/                     # 护眼排版 HTML 文件
│   └── 01_vllm_arch_and_sm70_hardware.html
├── code/                     # 配套实验代码与测试脚本
├── templates/                # 护眼排版模板（含 MathJax、深浅主题）
│   └── template.html
├── index.html                # 笔记总目录导航首页
├── build_notes.py            # 一键构建同步脚本
├── build_notes.bat           # Windows 批处理执行脚本
└── README.md
```

---

## 章节概览

| 篇章 | 主题 | 核心内容 | 快速通道 |
| :--- | :--- | :--- | :--- |
| **第 01 讲** | **vLLM 架构全景与 Volta 硬件底座** | Prefill vs Decode 计算与访存瓶颈、PagedAttention 分页机制、SM70 Volta 硬件限制剖析、FlashAttention 与 Online Softmax 递推公式 | [阅读 HTML](html/01_vllm_arch_and_sm70_hardware.html) · [查看 Markdown](markdown/01_vllm_arch_and_sm70_hardware.md) |
| **第 02 讲** | *(更新中)* **PR #286 算子精读** | GQA-packed Wide QK/PV 打包、Split-D、Volta HMMA 从 17.9 冲向 60.8 TFLOP/s | 整理中 |

---

## 使用说明

1. **阅读**：
   在浏览器中打开 [`index.html`](index.html) 或直接进入 [`html/`](html/) 查看对应篇章。右上角支持切换米纸或暮色模式。
2. **编辑**：
   在 [`markdown/`](markdown/) 下找到对应的 `.md` 文件直接修改或添加心得批注。
3. **重新构建**：
   * Windows 下双击运行 [`build_notes.bat`](build_notes.bat)；
   * 或在终端运行：
     ```bash
     python build_notes.py
     ```
   脚本会自动扫描 `markdown/` 并全自动刷新所有 HTML 页面及 `index.html` 目录。
