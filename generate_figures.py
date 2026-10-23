#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
生成符合 wdkns/tensor-formula-viz 规范的高密度三段式矢量图 (SVG + 原生 LaTeX 公式渲染)
1. Top: 原生 LaTeX 核心数学公式 ($\displaystyle ...$)
2. Middle: 形状对齐的分块张量、条纹、算子与硬件映射流 + 原生 LaTeX 符号/维度标注
3. Bottom: 轴 (Axes) / 对象 (Objects) / 机制 (Mechanism) 三行语义账本 (含行内 LaTeX) + 底部署名行
包含自动化 Bounding-Box 碰撞审计（零文字重叠、零压矩阵格子、零压箭头线）。
"""

from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

COMMON_DEFS = """
  <defs>
    <style>
      .fig-bg { fill: #ffffff; stroke: #cbd5e1; stroke-width: 1; }
      .stage-title { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; font-size: 13px; font-weight: 700; fill: #0f172a; }
      .op-node { fill: #f8fafc; stroke: #64748b; stroke-width: 1.2; }
      .op-text { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; font-size: 11.5px; font-weight: 700; fill: #1e293b; text-anchor: middle; }
      .note-card { fill: #f8fafc; stroke: #cbd5e1; stroke-width: 1; rx: 3; }
      .note-title { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; font-size: 11.5px; font-weight: 700; fill: #0f172a; }
      .ledger-box { fill: #f8fafc; stroke: #94a3b8; stroke-width: 1; rx: 2; }
      .ledger-key { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; font-size: 12px; font-weight: 700; fill: #0f172a; }
      .sig-line { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; font-size: 11px; fill: #94a3b8; text-anchor: middle; }
      .bracket { fill: none; stroke: #475569; stroke-width: 1.3; stroke-linecap: square; }
      .arrow { stroke: #64748b; stroke-width: 1.3; fill: none; marker-end: url(#arrowhead); }
      .arrow-accent { stroke: #c95b5b; stroke-width: 1.4; fill: none; marker-end: url(#arrowhead-red); }
      .arrow-teal { stroke: #4f8fa5; stroke-width: 1.4; fill: none; marker-end: url(#arrowhead-teal); }
    </style>
    <marker id="arrowhead" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
      <polygon points="0 0, 7 2.5, 0 5" fill="#64748b" />
    </marker>
    <marker id="arrowhead-red" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
      <polygon points="0 0, 7 2.5, 0 5" fill="#c95b5b" />
    </marker>
    <marker id="arrowhead-teal" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
      <polygon points="0 0, 7 2.5, 0 5" fill="#4f8fa5" />
    </marker>
  </defs>
"""


def tex_box(cx, y_top, w, h, content, align="center", size=12.5, color="#0f172a", weight="normal", bg="none", wrap=False):
    """
    在 SVG 中放置一个承载原生 LaTeX ($...$) 与混排中文的 foreignObject 盒子。
    - 若 align == 'center'，则 cx 表示水平中心坐标 (自动计算 x = cx - w/2)；
    - 若 align == 'left'，则 cx 表示左边缘坐标 x = cx。
    """
    x = (cx - w / 2.0) if align == "center" else cx
    justify = {"center": "center", "left": "flex-start", "right": "flex-end"}[align]
    ws = "normal" if wrap else "nowrap"
    t_align = "center" if align == "center" else "left"
    bg_css = f"background:{bg};border-radius:2px;padding:0 4px;" if bg != "none" else ""
    return (
        f'<foreignObject x="{x:.1f}" y="{y_top:.1f}" width="{w:.1f}" height="{h:.1f}">\n'
        f'  <div xmlns="http://www.w3.org/1999/xhtml" class="svg-latex" '
        f'style="width:100%;height:100%;display:flex;align-items:center;justify-content:{justify};'
        f'font-size:{size}px;color:{color};font-weight:{weight};line-height:1.32;'
        f'white-space:{ws};text-align:{t_align};{bg_css}">'
        f'<span>{content}</span></div>\n'
        f'</foreignObject>'
    )


def draw_matrix_grid(x, y, rows, cols, cell_w, cell_h, gap, base_color, pattern="dense", bracket=True):
    """Draw a Kimi/wdkns-style matrix tile grid with brackets and 3-level lightness variation."""
    shades = {
        "teal":   ["#b9d6e0", "#7fb3c5", "#4f8fa5"],
        "orange": ["#f8d6ba", "#f3b787", "#ee995b"],
        "coral":  ["#eac0c0", "#da8e8e", "#c95b5b"],
        "violet": ["#d1c7e3", "#ae9ecf", "#8a74b5"],
        "green":  ["#bce0c4", "#89c195", "#5b9a68"],
        "gray":   ["#dcdfe3", "#b1b5ba", "#85898f"],
    }[base_color]
    out = []
    total_w = cols * cell_w + (cols - 1) * gap
    total_h = rows * cell_h + (rows - 1) * gap

    for r in range(rows):
        for c in range(cols):
            cx = x + c * (cell_w + gap)
            cy = y + r * (cell_h + gap)
            if pattern == "causal" and c > r:
                fill = "#ffffff"
                stroke = "#cbd5e1"
            elif pattern == "diag":
                fill = shades[2] if r == c else "#ffffff"
                stroke = "none" if r == c else "#e2e8f0"
            elif pattern == "stripe_row":
                fill = shades[r % 3]
                stroke = "none"
            elif pattern == "stripe_col":
                fill = shades[c % 3]
                stroke = "none"
            else:
                idx = (r * 7 + c * 13 + r * c) % 3
                fill = shades[idx]
                stroke = "none"
            st_attr = f' stroke="{stroke}" stroke-width="0.6"' if stroke != "none" else ""
            out.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cell_w}" height="{cell_h}" rx="1" fill="{fill}"{st_attr}/>')

    if bracket:
        pad = 3.5
        bw = 4
        bx0, by0 = x - pad, y - pad
        bx1, by1 = x + total_w + pad, y + total_h + pad
        out.append(f'<path class="bracket" d="M {bx0+bw} {by0} L {bx0} {by0} L {bx0} {by1} L {bx0+bw} {by1}"/>')
        out.append(f'<path class="bracket" d="M {bx1-bw} {by0} L {bx1} {by0} L {bx1} {by1} L {bx1-bw} {by1}"/>')
    return "\n".join(out), total_w, total_h


def generate_fig01():
    """图1: Roofline 算术强度与 PagedAttention 二级页表寻址图"""
    w, h = 980, 650
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%">', COMMON_DEFS]
    svg.append(f'<rect class="fig-bg" x="1" y="1" width="{w-2}" height="{h-2}" rx="4"/>')

    # Zone 1: Top Formula (True LaTeX)
    svg.append(tex_box(490, 6, 940, 32,
        r'$\displaystyle \text{Perf} = \min\!\left(P_{\max},\; I \cdot B_{\text{HBM}}\right), \qquad I = \frac{\text{FLOPs}}{\text{Bytes}_{\text{moved}}}, \qquad \text{Addr}(t) = \text{Base}_{\text{KV}} + \text{BlockTable}\!\left[\left\lfloor \frac{t}{B_{\text{blk}}} \right\rfloor\right] S_{\text{page}} + \left(t \bmod B_{\text{blk}}\right) S_{\text{row}}$',
        align="center", size=14.2, color="#0f172a"))
    svg.append(tex_box(490, 38, 940, 22,
        r'—— 左半场：Prefill ($M \gg 1$) 算力受限 vs. Decode ($M = 1$) 显存带宽受限；右半场：PagedAttention 逻辑块到物理页跳表寻址',
        align="center", size=12, color="#475569"))
    svg.append('<line x1="24" y1="64" x2="956" y2="64" stroke="#e2e8f0" stroke-width="1"/>')

    # Stage (a): Roofline & Prefill vs Decode
    svg.append(tex_box(28, 70, 720, 24,
        r'<strong>(a) 阶段算术强度对照</strong> —— Tesla V100 拐点 $I^* = 125\text{ TFLOP/s} \div 900\text{ GB/s} \approx 138.9\text{ FLOPs/Byte}$',
        align="left", size=12.8, color="#0f172a"))

    # Mini Roofline Plot (Left: x=36..356, y=100..255)
    px, py, pw, ph = 36, 100, 320, 155
    svg.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" fill="#fcfcfd" stroke="#cbd5e1" stroke-width="1" rx="2"/>')
    ox, oy = px + 44, py + 124
    svg.append(f'<line x1="{ox}" y1="{oy}" x2="{px+304}" y2="{oy}" stroke="#475569" stroke-width="1.3"/>')
    svg.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{py+16}" stroke="#475569" stroke-width="1.3"/>')
    svg.append(tex_box(ox - 22, py + 28, 38, 18, r'$125\text{T}$', size=10.5, color="#64748b"))
    svg.append(tex_box(ox - 16, oy - 10, 24, 18, r'$0$', size=10.5, color="#64748b"))

    rx_pt, ry_pt = px + 162, py + 38
    svg.append(f'<polyline points="{ox},{oy} {rx_pt},{ry_pt} {px+298},{ry_pt}" fill="none" stroke="#4f8fa5" stroke-width="2.4"/>')
    svg.append(f'<line x1="{rx_pt}" y1="{ry_pt}" x2="{rx_pt}" y2="{oy}" stroke="#94a3b8" stroke-width="1" stroke-dasharray="3,3"/>')
    svg.append(tex_box(rx_pt, oy + 4, 150, 20, r'拐点 $I^* \approx 138.9\text{ FLOPs/B}$', size=10.5, color="#64748b"))

    dx_pt, dy_pt = px + 68, py + 104
    svg.append(f'<circle cx="{dx_pt}" cy="{dy_pt}" r="4.5" fill="#c95b5b"/>')
    svg.append(tex_box(dx_pt + 74, dy_pt - 10, 132, 20, r'Decode ($M{=}1, I{\approx}1{\sim}2$)', size=10.5, color="#c95b5b", weight="700", bg="#ffffff"))

    fx_pt, fy_pt = px + 235, ry_pt
    svg.append(f'<circle cx="{fx_pt}" cy="{fy_pt}" r="4.5" fill="#ee995b"/>')
    svg.append(tex_box(px + 232, fy_pt - 24, 130, 20, r'Prefill ($M{=}2048$)', size=10.8, color="#b45309", weight="700"))
    svg.append(tex_box(px + 232, fy_pt + 4, 120, 18, r'Compute-Bound', size=10.2, color="#b45309"))

    # Tensor comparison for Prefill vs Decode (Middle of Stage a)
    y_sym_a, y_shp_a = 196, 218

    m_q_pre, qw, qh = draw_matrix_grid(390, 116, 6, 8, 8.5, 8.5, 1.5, "orange")
    svg.append(m_q_pre)
    svg.append(tex_box(390 + qw / 2, y_sym_a, 90, 20, r'$\mathbf{Q}_{\text{prefill}}$', size=13))
    svg.append(tex_box(390 + qw / 2, y_shp_a, 96, 18, r'$(M{=}2048,\, d)$', size=10.5, color="#64748b"))

    svg.append(tex_box(492, 138, 28, 20, r'$\text{vs.}$', size=12, color="#475569"))

    m_q_dec, qdw, qdh = draw_matrix_grid(518, 142, 1, 8, 8.5, 8.5, 1.5, "coral")
    svg.append(m_q_dec)
    svg.append(tex_box(518 + qdw / 2, y_sym_a, 90, 20, r'$\mathbf{q}_{\text{decode}}$', size=13))
    svg.append(tex_box(518 + qdw / 2, y_shp_a, 96, 18, r'$(M{=}1,\, d)$ 单行', size=10.5, color="#64748b"))

    svg.append(tex_box(615, 138, 22, 20, r'$\times$', size=14, color="#1e293b"))

    m_kv, kvw, kvh = draw_matrix_grid(635, 110, 7, 8, 8.5, 8.5, 1.5, "teal")
    svg.append(m_kv)
    svg.append(tex_box(635 + kvw / 2, y_sym_a, 90, 20, r'$\mathbf{K}_{\text{cache}}^\top$', size=13))
    svg.append(tex_box(635 + kvw / 2, y_shp_a, 98, 18, r'$(d,\, L_{\text{ctx}})$ 全量', size=10.5, color="#64748b"))

    # Note card on right of Stage (a)
    svg.append('<rect x="742" y="100" width="214" height="155" class="note-card"/>')
    svg.append('<text x="754" y="120" class="note-title">为什么 Decode 必然受限于显存？</text>')
    svg.append(tex_box(754, 128, 192, 118,
        r'1. Prefill 时权重与 KV 被 $M{=}2048$ 个 Token 共享复用，算术强度高。<br/>'
        r'2. Decode 每次仅 $M{=}1$，却必须从 HBM2 读出全部历史 KV，每字节仅做 $1{\sim}2$ 次 FMA，$98\%$ 时间计算单元在饥饿等待显存。',
        align="left", size=10.8, color="#334155", wrap=True))

    # Stage (b): PagedAttention 2-level lookup
    svg.append(tex_box(28, 276, 820, 24,
        r'<strong>(b) PagedAttention 二级页表寻址</strong> —— 以 $t = 35,\; B_{\text{page}} = 16,\; \text{block\_table} = [7, 12, 3, 9]$ 为例 (<code>tail.cu:L340</code>)',
        align="left", size=12.8, color="#0f172a"))

    y_top_b = 304
    y_mat_b = 332
    y_sym_b = 400
    y_shp_b = 422

    # 1. Logical token stream
    svg.append(tex_box(119, y_top_b, 160, 20, r'逻辑序列位置 $t = 35$', size=11, color="#334155"))
    m_log, lw, lh = draw_matrix_grid(52, y_mat_b, 3, 8, 15, 13, 2, "violet", pattern="stripe_row")
    svg.append(m_log)
    svg.append(tex_box(28, y_mat_b - 2, 34, 16, r'$\text{LB}_0$', size=10, color="#64748b"))
    svg.append(tex_box(28, y_mat_b + 13, 34, 16, r'$\text{LB}_1$', size=10, color="#64748b"))
    svg.append(tex_box(28, y_mat_b + 28, 34, 16, r'$\text{LB}_2$', size=10, color="#64748b"))
    hx = 52 + 3 * (15 + 2)
    hy = y_mat_b + 2 * (13 + 2)
    svg.append(f'<rect x="{hx-1}" y="{hy-1}" width="17" height="15" fill="none" stroke="#c95b5b" stroke-width="2"/>')
    svg.append(tex_box(52 + lw / 2, y_sym_b, 140, 20, r'$\text{Logical Tokens}$', size=12.5, weight="700"))
    svg.append(tex_box(52 + lw / 2, y_shp_b, 175, 20, r'$\lfloor 35/16 \rfloor = 2\text{ (块)},\; 35\bmod 16 = 3$', size=10.2, color="#64748b"))

    # Arrow 1 -> Div/Mod op
    svg.append('<line x1="194" y1="356" x2="232" y2="356" class="arrow"/>')
    svg.append('<rect x="236" y="326" width="136" height="60" rx="3" class="op-node"/>')
    svg.append('<text x="304" y="344" class="op-text">整除与取模拆解</text>')
    svg.append(tex_box(304, 349, 126, 18, r'$b_{\text{log}} = \lfloor 35 / 16 \rfloor = 2$', size=10.8, color="#334155"))
    svg.append(tex_box(304, 366, 126, 18, r'$\text{off} = 35 \bmod 16 = 3$', size=10.8, color="#334155"))

    # Arrow 2 -> Block Table
    svg.append('<line x1="376" y1="356" x2="408" y2="356" class="arrow"/>')
    svg.append(tex_box(468, y_top_b, 140, 20, r'页表 $\text{BlockTable}[4]$', size=11, color="#334155"))
    bt_vals = ["7", "12", "3", "9"]
    for idx, val in enumerate(bt_vals):
        bx = 414 + idx * 28
        by = 336
        fill = "#f3b787" if idx == 2 else "#e2e8f0"
        stroke = "#c95b5b" if idx == 2 else "#64748b"
        svg.append(f'<rect x="{bx}" y="{by}" width="25" height="32" rx="2" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
        svg.append(tex_box(bx + 12.5, by + 6, 24, 20, rf'${val}$', size=12, weight="700"))
        svg.append(tex_box(bx + 12.5, by + 34, 26, 16, rf'$[{idx}]$', size=9.8, color="#64748b"))
    svg.append(tex_box(468, y_sym_b, 120, 20, r'$\text{BlockTable}$', size=12.5, weight="700"))
    svg.append(tex_box(468, y_shp_b, 155, 20, r'$\text{BlockTable}[2] \to \text{页 } 3$', size=10.2, color="#64748b"))

    # Arrow 3 -> Physical KV Pool
    svg.append('<line x1="530" y1="356" x2="564" y2="356" class="arrow-accent"/>')
    svg.append(tex_box(665, y_top_b, 160, 20, r'HBM2 物理块池 $\text{KV}_{\text{pool}}$', size=11, color="#334155"))
    m_phys, pw2, ph2 = draw_matrix_grid(602, y_mat_b - 4, 4, 8, 14, 11, 2, "teal")
    svg.append(m_phys)
    svg.append(tex_box(581, y_mat_b - 5, 28, 15, r'$P_0$', size=9.8, color="#64748b"))
    svg.append(tex_box(581, y_mat_b + 8, 28, 15, r'$\mathbf{P_3}$', size=9.8, color="#c95b5b"))
    svg.append(tex_box(581, y_mat_b + 21, 28, 15, r'$P_7$', size=9.8, color="#64748b"))
    svg.append(tex_box(581, y_mat_b + 34, 32, 15, r'$P_{12}$', size=9.8, color="#64748b"))
    px_hit = 602 + 3 * (14 + 2)
    py_hit = (y_mat_b - 4) + 1 * (11 + 2)
    svg.append(f'<rect x="{px_hit-1}" y="{py_hit-1}" width="16" height="13" fill="none" stroke="#c95b5b" stroke-width="2.2"/>')
    svg.append(tex_box(602 + pw2 / 2, y_sym_b, 140, 20, r'$\text{Physical KV Cache}$', size=12.5, weight="700"))
    svg.append(tex_box(602 + pw2 / 2, y_shp_b, 175, 20, r'$128\text{-bit}$ 向量化合并加载', size=10.2, color="#64748b"))

    # Warning card on right of Stage (b)
    svg.append('<rect x="752" y="308" width="204" height="122" class="note-card"/>')
    svg.append('<text x="764" y="327" class="note-title">CUDA 寻址合并铁律</text>')
    svg.append(tex_box(764, 334, 182, 90,
        r'单线程算单 Token 地址会导致严重非合并访存。<code>tail.cu</code> 让 $8$ 个相邻线程协同读取同一行 $d_{\text{chunk}}{=}64$，凑齐 $128\text{-bit}$ 连续事务。',
        align="left", size=10.8, color="#334155", wrap=True))

    # Zone 3: Bottom Meaning Ledger
    ly = 456
    svg.append(f'<rect x="24" y="{ly}" width="932" height="156" class="ledger-box"/>')
    svg.append(f'<text x="38" y="{ly+24}" class="ledger-key">轴</text>')
    svg.append(tex_box(88, ly + 8, 850, 26,
        r'$M$：单步处理的 Token 行数（Prefill 时 $M = L_{\text{prompt}} \gg 1$，Decode 时 $M = 1$）；$L_{\text{ctx}}$：历史上下文总长；$B_{\text{blk}} = 16$：单物理页容纳槽位数；$d = 256$：单头特征维。',
        align="left", size=11.2, color="#334155"))
    svg.append(f'<line x1="36" y1="{ly+38}" x2="944" y2="{ly+38}" stroke="#e2e8f0" stroke-width="1"/>')

    svg.append(f'<text x="38" y="{ly+60}" class="ledger-key">对象</text>')
    svg.append(tex_box(88, ly + 44, 850, 26,
        r'$I\text{ (FLOPs/Byte)}$：算术强度；$\text{BlockTable} \in \mathbb{Z}^{B \times N_{\text{pages}}}$：逻辑块号到物理页号的映射表；$\text{KV}_{\text{pool}}$：HBM2 中按固定页大小预分配的显存块池。',
        align="left", size=11.2, color="#334155"))
    svg.append(f'<line x1="36" y1="{ly+74}" x2="944" y2="{ly+74}" stroke="#e2e8f0" stroke-width="1"/>')

    svg.append(f'<text x="38" y="{ly+98}" class="ledger-key">机制</text>')
    svg.append(tex_box(88, ly + 80, 850, 66,
        r'(i) Roofline 决定优化方向：Prefill 优化 Tensor Core 吞吐（宽矩阵打包），Decode 优化 HBM2 访存效率。<br/>'
        r'(ii) PagedAttention 将寻址拆为「查页表 $p_{\text{phys}} = \text{BlockTable}[\lfloor t / B_{\text{blk}} \rfloor]$」+「页内偏移 $(t \bmod B_{\text{blk}}) S_{\text{row}}$」，页内保持连续以支撑 $128\text{-bit}$ (<code>float4</code>) 合并访存。',
        align="left", size=11.2, color="#334155", wrap=True))

    svg.append(f'<text x="490" y="{h-12}" class="sig-line">Roofline 算术强度与 PagedAttention 二级页表寻址机制 @ 1Cat-vLLM 源码精读</text>')
    svg.append('</svg>')
    (ASSETS_DIR / "fig01_roofline_and_paging.svg").write_text("\n".join(svg), encoding="utf-8")


def generate_fig02():
    """图2: Online Softmax 分块递推与浮点除法约分消解机制"""
    w, h = 980, 635
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%">', COMMON_DEFS]
    svg.append(f'<rect class="fig-bg" x="1" y="1" width="{w-2}" height="{h-2}" rx="4"/>')

    # Zone 1: Top Formula (True LaTeX)
    svg.append(tex_box(490, 6, 940, 32,
        r'$\displaystyle m^{\text{new}} = \max\!\left(m^{(1)}, m^{(2)}\right), \qquad l^{\text{new}} = e^{m^{(1)} - m^{\text{new}}} l^{(1)} + e^{m^{(2)} - m^{\text{new}}} l^{(2)}, \qquad \tilde{\mathbf{O}}^{\text{new}} = e^{m^{(1)} - m^{\text{new}}} \tilde{\mathbf{O}}^{(1)} + \exp\!\left(\mathbf{S}^{(2)} - m^{\text{new}}\right)\mathbf{V}^{(2)}$',
        align="center", size=14.2, color="#0f172a"))
    svg.append(tex_box(490, 38, 940, 22,
        r'—— 核心洞察：片上寄存器仅维护未归一化分子 $\tilde{\mathbf{O}}$ 与统计量 $(m, l)$，中间块 $l^{(2)}$ 乘除直接代数约分抵消，全程零中间除法！',
        align="center", size=12, color="#475569"))
    svg.append('<line x1="24" y1="64" x2="956" y2="64" stroke="#e2e8f0" stroke-width="1"/>')

    # Stage (a): Naive 3-pass vs Online 1-pass
    svg.append(tex_box(28, 70, 780, 24,
        r'<strong>(a) 传统 Safe Softmax 的全局屏障 vs. Online Softmax 在线流式递推</strong>',
        align="left", size=12.8, color="#0f172a"))

    y_sym_a, y_shp_a = 196, 218

    # Left: Naive N x N materialized in HBM
    m_s, sw, sh = draw_matrix_grid(36, 104, 6, 6, 11, 11, 2, "coral", pattern="causal")
    svg.append(m_s)
    svg.append(tex_box(36 + sw / 2, y_sym_a, 96, 20, r'$\mathbf{S} = \mathbf{Q}\mathbf{K}^\top$', size=12.8))
    svg.append(tex_box(36 + sw / 2, y_shp_a, 106, 18, r'$(N \times N)$ 落盘 HBM', size=10.2, color="#64748b"))

    svg.append('<line x1="120" y1="144" x2="192" y2="144" class="arrow-accent"/>')
    svg.append(tex_box(156, 118, 68, 20, r'$\mathcal{O}(N^2)$ 读写', size=10.2, color="#c95b5b", weight="700"))

    m_p, pw, ph = draw_matrix_grid(200, 104, 6, 6, 11, 11, 2, "orange", pattern="causal")
    svg.append(m_p)
    svg.append(tex_box(200 + pw / 2, y_sym_a, 116, 20, r'$\mathbf{P} = \text{Softmax}(\mathbf{S})$', size=12.5))
    svg.append(tex_box(200 + pw / 2, y_shp_a, 120, 18, r'三遍扫全行求 $m, l$', size=10.2, color="#64748b"))

    # Vertical separator
    svg.append('<line x1="296" y1="98" x2="296" y2="236" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4,4"/>')

    # Right: Online Tiled Flow in SRAM/Registers
    m_q, qw, qh = draw_matrix_grid(316, 116, 4, 6, 10.5, 10.5, 2, "orange")
    svg.append(m_q)
    svg.append(tex_box(316 + qw / 2, y_sym_a, 80, 20, r'$\mathbf{Q}_{\text{tile}}$', size=12.8))
    svg.append(tex_box(316 + qw / 2, y_shp_a, 92, 18, r'$(B_r \times d)$ 常驻', size=10.2, color="#64748b"))

    svg.append(tex_box(404, 134, 22, 20, r'$\times$', size=14, color="#1e293b"))

    m_kt, ktw, kth = draw_matrix_grid(420, 104, 6, 4, 10.5, 10.5, 2, "teal")
    svg.append(m_kt)
    svg.append(tex_box(420 + ktw / 2, y_sym_a, 82, 20, r'$\left(\mathbf{K}^{(j)}\right)^\top$', size=12.5))
    svg.append(tex_box(420 + ktw / 2, y_shp_a, 90, 18, r'$(d \times B_c)$ 流式', size=10.2, color="#64748b"))

    svg.append('<line x1="476" y1="142" x2="508" y2="142" class="arrow-teal"/>')

    m_st, stw, sth = draw_matrix_grid(516, 116, 4, 4, 10.5, 10.5, 2, "violet")
    svg.append(m_st)
    svg.append(tex_box(516 + stw / 2, y_sym_a, 70, 20, r'$\mathbf{S}^{(j)}$', size=12.8))
    svg.append(tex_box(516 + stw / 2, y_shp_a, 98, 18, r'$(B_r \times B_c)$ 寄存器', size=10.2, color="#64748b"))

    svg.append('<line x1="572" y1="142" x2="604" y2="142" class="arrow-teal"/>')

    m_ml, mlw, mlh = draw_matrix_grid(614, 116, 4, 2, 10.5, 10.5, 2, "green", pattern="stripe_col")
    svg.append(m_ml)
    svg.append(tex_box(614 + mlw / 2, y_sym_a, 54, 20, r'$m,\; l$', size=12.8))
    svg.append(tex_box(614 + mlw / 2, y_shp_a, 62, 18, r'$(B_r \times 1)$', size=10.2, color="#64748b"))

    m_ot, otw, oth = draw_matrix_grid(682, 116, 4, 6, 10.5, 10.5, 2, "teal")
    svg.append(m_ot)
    svg.append(tex_box(682 + otw / 2, y_sym_a, 70, 20, r'$\tilde{\mathbf{O}}$', size=13))
    svg.append(tex_box(682 + otw / 2, y_shp_a, 96, 18, r'$(B_r \times d)$ 累加器', size=10.2, color="#64748b"))

    svg.append('<rect x="784" y="100" width="172" height="134" class="note-card"/>')
    svg.append('<text x="796" y="120" class="note-title">零 HBM 中间写回</text>')
    svg.append(tex_box(796, 128, 152, 98,
        r'• $\mathbf{S}^{(j)}$ 与 $\tilde{\mathbf{P}}^{(j)}$ 算完即刻在片上寄存器消费；<br/>'
        r'• 显存读写复杂度从 $\mathcal{O}(N^2)$ 彻底降至 $\mathcal{O}(N)$！',
        align="left", size=10.8, color="#334155", wrap=True))

    # Stage (b): Algebraic cancellation of l^(2) and SFU division avoidance
    svg.append(tex_box(28, 262, 860, 24,
        r'<strong>(b) 为什么工程代码中看不到「先除 $l^{(2)}$ 再乘 $l^{(2)}$」？</strong> —— 代数约分与 Tensor Core 流水线保护',
        align="left", size=12.8, color="#0f172a"))

    # Bad pipeline box
    svg.append('<rect x="36" y="290" width="436" height="132" rx="3" fill="#fff5f5" stroke="#c95b5b" stroke-width="1.2"/>')
    svg.append('<text x="50" y="312" class="note-title" fill="#991b1b">❌ 机械照搬归一化公式 (触发 SFU 标量除法停顿 Stall)</text>')
    svg.append(tex_box(254, 318, 400, 36,
        r'$\displaystyle \mathbf{O}^{(2)} = \frac{\tilde{\mathbf{P}}^{(2)}\mathbf{V}^{(2)}}{\color{#c95b5b}{l^{(2)}}} \quad\longrightarrow\quad {\color{#c95b5b}{l^{(2)}}} \cdot \mathbf{O}^{(2)}$',
        align="center", size=13.5, color="#0f172a"))
    svg.append(tex_box(50, 356, 410, 58,
        r'• Tensor Core 算完后打断流水线，交由 SFU 逐元素做浮点除法 ($\div\, l^{(2)}$)；<br/>'
        r'• 随后合并时又要再乘回 $l^{(2)}$，白白浪费数十周期指令延迟与精度。',
        align="left", size=10.8, color="#334155", wrap=True))

    # Good pipeline box
    svg.append('<rect x="492" y="290" width="464" height="132" rx="3" fill="#f0fdf4" stroke="#5b9a68" stroke-width="1.2"/>')
    svg.append('<text x="506" y="312" class="note-title" fill="#166534">✅ 1Cat-vLLM / FlashAttention 真实寄存器更新路径 (tail.cu:L285--L325)</text>')
    svg.append(tex_box(724, 318, 436, 36,
        r'$\displaystyle \tilde{\mathbf{O}}^{\text{new}} = 2^{(m_{\text{old}} - m_{\text{new}})\log_2 e} \cdot \tilde{\mathbf{O}}^{\text{old}} + \tilde{\mathbf{P}}^{(2)}\mathbf{V}^{(2)}$',
        align="center", size=13.5, color="#0f172a"))
    svg.append(tex_box(506, 356, 438, 60,
        r'• 分母 $l^{(2)}$ 与外层系数 $l^{(2)}$ 代数上严格抵消，内层循环完全不求局部 $\mathbf{O}^{(2)}$；<br/>'
        r'• 仅对旧累加器乘缩放因子 $\alpha = 2^{(m_{\text{old}}-m_{\text{new}})\cdot s}$，直到全部 $K,V$ 块结束写回 HBM 前才执行唯一一次 $\div\, l_{\text{final}}$！',
        align="left", size=10.8, color="#334155", wrap=True))

    # Zone 3: Bottom Meaning Ledger
    ly = 442
    svg.append(f'<rect x="24" y="{ly}" width="932" height="156" class="ledger-box"/>')
    svg.append(f'<text x="38" y="{ly+24}" class="ledger-key">轴</text>')
    svg.append(tex_box(88, ly + 8, 850, 26,
        r'$B_r\;(\texttt{kBlockM}=64)$：单次驻留片上的 Query 行块；$B_c\;(\texttt{kBlockN}=32)$：流式扫过的 Key/Value 列块；$d\;(\texttt{kHeadDim}=256)$：注意力头特征维度。',
        align="left", size=11.2, color="#334155"))
    svg.append(f'<line x1="36" y1="{ly+38}" x2="944" y2="{ly+38}" stroke="#e2e8f0" stroke-width="1"/>')

    svg.append(f'<text x="38" y="{ly+60}" class="ledger-key">对象</text>')
    svg.append(tex_box(88, ly + 44, 850, 26,
        r'$m \in \mathbb{R}^{B_r}$：已扫描前缀的行最大值；$l \in \mathbb{R}^{B_r}$：以当前 $m$ 为基准的指数和分母；$\tilde{\mathbf{O}} \in \mathbb{R}^{B_r \times d}$：未除以 $l$ 的分子累加器；$\alpha = e^{m_{\text{old}} - m_{\text{new}}} \in (0, 1]$：衰减因子。',
        align="left", size=11.2, color="#334155"))
    svg.append(f'<line x1="36" y1="{ly+74}" x2="944" y2="{ly+74}" stroke="#e2e8f0" stroke-width="1"/>')

    svg.append(f'<text x="38" y="{ly+98}" class="ledger-key">机制</text>')
    svg.append(tex_box(88, ly + 80, 850, 66,
        r'(i) 指数恒等式 $e^{x - m_{\text{new}}} = e^{x - m_{\text{old}}} \cdot e^{m_{\text{old}} - m_{\text{new}}}$ 使得历史统计量 $(l, \tilde{\mathbf{O}})$ 只需乘以标量 $\alpha$ 即可对齐到新基准 $m_{\text{new}}$，且因 $m_{\text{old}} \le m_{\text{new}}$ 保证 $\alpha \le 1$ 永不上溢。<br/>'
        r'(ii) 1Cat-vLLM 在 <code>tail.cu</code> 中用 $\exp_2\!\big((m_{\text{old}} - m_{\text{new}})\cdot s_{\log_2}\big)$ 替代自然指数，直接映射为单条 GPU 硬件 <code>MUFU.EX2</code> 指令。',
        align="left", size=11.2, color="#334155", wrap=True))

    svg.append(f'<text x="490" y="{h-12}" class="sig-line">FlashAttention Online Softmax 在线递推与代数约分消解机制 @ 1Cat-vLLM 源码精读</text>')
    svg.append('</svg>')
    (ASSETS_DIR / "fig02_online_softmax_flow.svg").write_text("\n".join(svg), encoding="utf-8")


def generate_fig03():
    """图3: 1Cat-vLLM SM70 Split-D (d=256 -> 4x64) 双 Warp 协同与 45,568 B Smem 布局图"""
    w, h = 980, 735
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%">', COMMON_DEFS]
    svg.append(f'<rect class="fig-bg" x="1" y="1" width="{w-2}" height="{h-2}" rx="4"/>')

    # Zone 1: Top Formula (True LaTeX)
    svg.append(tex_box(490, 6, 940, 32,
        r'$\displaystyle \mathbf{S}_{\text{group}} = \sum_{c=0}^{3} \mathbf{Q}_{\text{group}, c} \left(\mathbf{K}_c\right)^\top, \qquad \left[\mathbf{O}_{\text{group}, [0:128]} \;\middle|\; \mathbf{O}_{\text{group}, [128:256]}\right] \;\longleftarrow\; \left[\text{Warp}_0:\, \mathbf{P}_{\text{group}}\mathbf{V}_{[0:128]} \;\middle|\; \text{Warp}_1:\, \mathbf{P}_{\text{group}}\mathbf{V}_{[128:256]}\right]$',
        align="center", size=14, color="#0f172a"))
    svg.append(tex_box(490, 38, 940, 22,
        r'—— <code>Sm70D256SplitDTraits</code> (<code>tail.cu</code>)：QK 阶段双 Warp 按行切分 (各算 $8\text{行} \times D$)，PV 阶段共享 $16$ 行 $\mathbf{P}$ 并按列切分 $D/2$ (各管 $2$ 个 $d_{\text{chunk}}=64$)',
        align="center", size=12, color="#475569"))
    svg.append('<line x1="24" y1="64" x2="956" y2="64" stroke="#e2e8f0" stroke-width="1"/>')

    # Stage (a): QK Phase (Row-split within Warp Pair)
    svg.append(tex_box(28, 70, 820, 24,
        r'<strong>(a) QK 阶段：组内双 Warp 按行分工（$\texttt{kQkWarpRows} = 8$），流式累加 $4$ 个 $d_{\text{chunk}} = 64$</strong>',
        align="left", size=12.8, color="#0f172a"))

    y_top_a = 94
    y_mat_a = 114
    y_sym_a = 162
    y_shp_a = 182

    # Q_group (16 rows x 4 chunks)
    for c_idx in range(4):
        bx = 42 + c_idx * 48
        m_w0, cw, ch = draw_matrix_grid(bx, y_mat_a, 2, 4, 8, 8, 1.5, "orange", bracket=False)
        m_w1, _, _ = draw_matrix_grid(bx, y_mat_a + ch + 3, 2, 4, 8, 8, 1.5, "coral", bracket=False)
        svg.append(m_w0)
        svg.append(m_w1)
        svg.append(tex_box(bx + cw / 2, y_top_a, 42, 16, rf'$c={c_idx}$', size=10.2, color="#64748b"))
    svg.append(f'<path class="bracket" d="M 43 {y_mat_a-3} L 38 {y_mat_a-3} L 38 {y_mat_a+41} L 43 {y_mat_a+41}"/>')
    svg.append(f'<path class="bracket" d="M 225 {y_mat_a-3} L 230 {y_mat_a-3} L 230 {y_mat_a+41} L 225 {y_mat_a+41}"/>')
    svg.append(tex_box(134, y_sym_a, 150, 20, r'$\mathbf{Q}_{\text{group}}\; (16 \times 256)$', size=12.5))
    svg.append(tex_box(134, y_shp_a, 195, 18, r'橙: $\text{Warp}_0\,(8\text{行})$ | 红: $\text{Warp}_1\,(8\text{行})$', size=10.2, color="#64748b"))

    svg.append(tex_box(248, 124, 22, 20, r'$\times$', size=14, color="#1e293b"))

    # K_chunk^T (64 x 32)
    m_kc, kcw, kch = draw_matrix_grid(268, y_mat_a, 4, 4, 9, 9, 1.5, "teal")
    svg.append(m_kc)
    svg.append(tex_box(268 + kcw / 2, y_sym_a, 70, 20, r'$\left(\mathbf{K}_c\right)^\top$', size=12.5))
    svg.append(tex_box(268 + kcw / 2, y_shp_a, 96, 18, r'$(64 \times 32)$ 双缓冲', size=10.2, color="#64748b"))

    svg.append('<line x1="322" y1="134" x2="358" y2="134" class="arrow"/>')
    svg.append(tex_box(340, 108, 48, 22, r'$\sum_{c=0}^{3}$', size=11, color="#475569"))

    # S_group (Warp 0 top 8x32, Warp 1 bottom 8x32)
    m_sw0, sww, swh = draw_matrix_grid(368, y_mat_a, 2, 4, 9, 8, 1.5, "orange", bracket=False)
    m_sw1, _, _ = draw_matrix_grid(368, y_mat_a + swh + 3, 2, 4, 9, 8, 1.5, "coral", bracket=False)
    svg.append(m_sw0)
    svg.append(m_sw1)
    svg.append(f'<path class="bracket" d="M 369 {y_mat_a-3} L 365 {y_mat_a-3} L 365 {y_mat_a+41} L 369 {y_mat_a+41}"/>')
    svg.append(f'<path class="bracket" d="M {368+sww-1} {y_mat_a-3} L {368+sww+3} {y_mat_a-3} L {368+sww+3} {y_mat_a+41} L {368+sww-1} {y_mat_a+41}"/>')
    svg.append(tex_box(368 + sww / 2, y_sym_a, 70, 20, r'$\mathbf{S}_{\text{acc}}$', size=12.5))
    svg.append(tex_box(368 + sww / 2, y_shp_a, 86, 18, r'各持 $(8 \times 32)$', size=10.2, color="#64748b"))

    svg.append('<line x1="422" y1="134" x2="462" y2="134" class="arrow-accent"/>')

    # Smem exchange box (sP + row_scale_exchange)
    svg.append('<rect x="468" y="98" width="178" height="66" rx="3" fill="#fef3c7" stroke="#d97706" stroke-width="1.2"/>')
    svg.append('<text x="557" y="117" class="op-text" fill="#92400e">Smem 跨 Warp 同步桥</text>')
    svg.append(tex_box(557, 123, 168, 18, r'1. 写满完整 $\mathbf{sP}\;(16 \times 32)$', size=10.5, color="#334155"))
    svg.append(tex_box(557, 141, 168, 18, r'2. $\texttt{row\_scale\_exchange}[64]$', size=10.5, color="#334155"))
    svg.append(tex_box(557, y_shp_a, 180, 18, r'交换各自 $8$ 行的 $2^{\Delta m \cdot s}$ 衰减因子', size=10.2, color="#64748b"))

    # Note card on right of Stage (a)
    svg.append('<rect x="662" y="94" width="294" height="104" class="note-card"/>')
    svg.append('<text x="674" y="113" class="note-title">为什么需要 row_scale_exchange？(tail.cu:L295)</text>')
    svg.append(tex_box(674, 120, 272, 72,
        r'• QK 阶段 $\text{Warp}_0$ 算前 $8$ 行，$\text{Warp}_1$ 算后 $8$ 行（零重复）；<br/>'
        r'• 但 PV 阶段每个 Warp 都要负责全部 $16$ 行的 $D/2$ 输出列！<br/>'
        r'• 故通过 $512\text{B}$ Smem 交换行缩放因子来更新 $16$ 行 $\tilde{\mathbf{O}}$。',
        align="left", size=10.6, color="#334155", wrap=True))

    # Stage (b): PV Phase (Column-split D/2 across Warp Pair)
    svg.append(tex_box(28, 214, 860, 24,
        r'<strong>(b) PV 阶段：双 Warp 共享完整 $\mathbf{sP}\;(16 \times 32)$，按特征维 $D$ 对半切分（$\texttt{kOwnedDChunks} = 2$，零寄存器溢出）</strong>',
        align="left", size=12.8, color="#0f172a"))

    y_grp_b = 240
    y_sub_b = 258
    y_mat_b = 278
    y_sym_b = 324
    y_shp_b = 344

    # sP (16 x 32) full 4 rows
    m_sp, spw, sph = draw_matrix_grid(48, y_mat_b, 4, 4, 10, 9, 1.5, "violet")
    svg.append(m_sp)
    svg.append(tex_box(48 + spw / 2, y_sym_b, 86, 20, r'$\mathbf{sP}\text{ (Smem)}$', size=12.5))
    svg.append(tex_box(48 + spw / 2, y_shp_b, 92, 18, r'$(16 \times 32)$ 共享', size=10.2, color="#64748b"))

    svg.append(tex_box(114, y_mat_b + 10, 22, 20, r'$\times$', size=14, color="#1e293b"))

    # V chunks: V0, V1 (Warp 0 in teal) and V2, V3 (Warp 1 in green)
    svg.append(tex_box(182, y_grp_b, 92, 18, r'$\text{Warp}_0\;(D/2)$', size=10.5, color="#0f766e", weight="700"))
    svg.append(tex_box(286, y_grp_b, 92, 18, r'$\text{Warp}_1\;(D/2)$', size=10.5, color="#15803d", weight="700"))
    for c_idx in range(4):
        bx = 136 + c_idx * 52
        col_theme = "teal" if c_idx < 2 else "green"
        m_vc, vcw, vch = draw_matrix_grid(bx, y_mat_b, 4, 4, 9, 9, 1.5, col_theme, bracket=True)
        svg.append(m_vc)
        svg.append(tex_box(bx + vcw / 2, y_sub_b, 38, 16, rf'$\mathbf{{V}}_{c_idx}$', size=10.5, color="#475569"))
    svg.append(tex_box(234, y_sym_b, 160, 20, r'$\mathbf{V}_{0..3}\; (32 \times 64 \times 4)$', size=12.5))
    svg.append(tex_box(234, y_shp_b, 200, 18, r'分两拍双缓冲加载 $\mathbf{sV}_0, \mathbf{sV}_1$ (TT)', size=10.2, color="#64748b"))

    svg.append(f'<line x1="348" y1="{y_mat_b+20}" x2="394" y2="{y_mat_b+20}" class="arrow"/>')

    # Output o_storage: Warp 0 owns d_chunk 0,1; Warp 1 owns d_chunk 2,3
    svg.append(tex_box(454, y_grp_b, 96, 18, r'$\text{Warp}_0\text{ 寄存器}$', size=10.5, color="#b45309", weight="700"))
    svg.append(tex_box(558, y_grp_b, 96, 18, r'$\text{Warp}_1\text{ 寄存器}$', size=10.5, color="#991b1b", weight="700"))
    for c_idx in range(4):
        bx = 408 + c_idx * 52
        col_theme = "orange" if c_idx < 2 else "coral"
        m_oc, ocw, och = draw_matrix_grid(bx, y_mat_b, 4, 4, 9, 9, 1.5, col_theme, bracket=True)
        svg.append(m_oc)
        svg.append(tex_box(bx + ocw / 2, y_sub_b, 38, 16, rf'$\tilde{{\mathbf{{O}}}}_{c_idx}$', size=10.5, color="#475569"))
    svg.append(tex_box(506, y_sym_b, 190, 20, r'$\texttt{o\_storage}[2][\texttt{kOElements}]$', size=12))
    svg.append(tex_box(506, y_shp_b, 210, 18, r'每 Warp 仅驻留 $(16 \times 128)$ 即 $D/2$', size=10.2, color="#64748b"))

    # Comparison card on right of Stage (b)
    svg.append('<rect x="630" y="242" width="326" height="114" class="note-card"/>')
    svg.append('<text x="642" y="261" class="note-title">寄存器减半效应：从 Local Memory Spill 到零溢出</text>')
    svg.append(tex_box(642, 268, 304, 80,
        r'• 若单 Warp 强行持有全部 $D{=}256$ 列 $\tilde{\mathbf{O}}$，单线程累加器膨胀触发 NVCC 溢出至 HBM2 Local Memory (延迟暴增 $200\times$)；<br/>'
        r'• Split-D 让每个 Warp 只管 $2$ 个 $d_{\text{chunk}}{=}64$，彻底归零 Spill！',
        align="left", size=10.6, color="#334155", wrap=True))

    # Stage (c): Exact 45,568 Bytes Shared Memory Layout
    svg.append(tex_box(28, 374, 860, 24,
        r'<strong>(c) 静态锁死的 $45,568\text{ Bytes}\;(44.5\text{ KB})$ Shared Memory 物理排布与生命周期复用 (<code>tail.cu:L111--L127</code>)</strong>',
        align="left", size=12.8, color="#0f172a"))

    bar_x, bar_y, bar_h = 36, 402, 56
    # 1. sQ: width 480
    svg.append(f'<rect x="{bar_x}" y="{bar_y}" width="480" height="{bar_h}" fill="#f8d6ba" stroke="#ee995b" stroke-width="1.3"/>')
    svg.append(tex_box(bar_x + 240, bar_y + 6, 440, 22, r'<strong>$\mathbf{sQ}$ (Query Tile 常驻区)</strong>', size=11.5, color="#1e293b"))
    svg.append(tex_box(bar_x + 240, bar_y + 28, 440, 22, r'$64 \times 256 \times 2\text{B} = 32,768\text{ Bytes}\;(\texttt{Swizzle}\langle 3,3,3\rangle)$', size=10.5, color="#334155"))

    # 2. kv_smem: width 200
    x2 = bar_x + 480
    svg.append(f'<rect x="{x2}" y="{bar_y}" width="200" height="{bar_h}" fill="#b9d6e0" stroke="#4f8fa5" stroke-width="1.3"/>')
    svg.append(tex_box(x2 + 100, bar_y + 6, 190, 22, r'<strong>$\texttt{kv\_smem}$ ($\mathbf{sK}/\mathbf{sV}$ 复用)</strong>', size=11.2, color="#1e293b"))
    svg.append(tex_box(x2 + 100, bar_y + 28, 190, 22, r'$2 \times (32 \times 64) \times 2\text{B} = 8,192\text{ B}$', size=10.2, color="#334155"))

    # 3. sP: width 136
    x3 = x2 + 200
    svg.append(f'<rect x="{x3}" y="{bar_y}" width="136" height="{bar_h}" fill="#d1c7e3" stroke="#8a74b5" stroke-width="1.3"/>')
    svg.append(tex_box(x3 + 68, bar_y + 6, 128, 22, r'<strong>$\mathbf{sP}$ (概率矩阵)</strong>', size=11.2, color="#1e293b"))
    svg.append(tex_box(x3 + 68, bar_y + 28, 128, 22, r'$64 \times 32 \times 2\text{B} = 4,096\text{ B}$', size=10.2, color="#334155"))

    # 4. row_scale_exchange: width 104
    x4 = x3 + 136
    svg.append(f'<rect x="{x4}" y="{bar_y}" width="104" height="{bar_h}" fill="#bce0c4" stroke="#5b9a68" stroke-width="1.3"/>')
    svg.append(tex_box(x4 + 52, bar_y + 6, 98, 22, r'<strong>scale 交换</strong>', size=11.2, color="#1e293b"))
    svg.append(tex_box(x4 + 52, bar_y + 28, 98, 22, r'$128 \times 4\text{B} = 512\text{ B}$', size=10, color="#334155"))

    # Below memory bar: two non-overlapping callout strips!
    svg.append(f'<rect x="36" y="{bar_y+bar_h+8}" width="460" height="24" rx="2" fill="#f0fdf4" stroke="#5b9a68" stroke-width="1"/>')
    svg.append(tex_box(266, bar_y + bar_h + 9, 448, 22,
        r'$\texttt{static\_assert(kSmemBytes == 45568)} \longrightarrow \text{留出 }{\sim}19.5\text{ KB 给 V100 L1 Cache！}$',
        size=10.5, color="#166534"))

    svg.append(f'<rect x="508" y="{bar_y+bar_h+8}" width="448" height="24" rx="2" fill="#fff1f2" stroke="#c95b5b" stroke-width="1" stroke-dasharray="3,2"/>')
    svg.append(tex_box(732, bar_y + bar_h + 9, 436, 22,
        r'生命周期别名复用：QK 阶段 $\mathbf{sK}$ 双缓冲 ($2{\times}2240\text{ half}, \text{Pitch}{=}68$) 借用 $\mathbf{sP}$ 头部',
        size=10.5, color="#991b1b"))

    # Zone 3: Bottom Meaning Ledger
    ly = 504
    svg.append(f'<rect x="24" y="{ly}" width="932" height="196" class="ledger-box"/>')
    svg.append(f'<text x="38" y="{ly+24}" class="ledger-key">轴</text>')
    svg.append(tex_box(88, ly + 8, 850, 26,
        r'$\texttt{kBlockM}=64$ ($4$ 个 MMA Group $\times$ 每组 $\texttt{kGroupRows}=16$ 行)；$\texttt{kBlockN}=32$；$\texttt{kHeadDim}=256$ 切为 $4$ 个 $d_{\text{chunk}}=64$；$\texttt{kNThreads}=256$ ($8$ Warps)。',
        align="left", size=11.2, color="#334155"))
    svg.append(f'<line x1="36" y1="{ly+38}" x2="944" y2="{ly+38}" stroke="#e2e8f0" stroke-width="1"/>')

    svg.append(f'<text x="38" y="{ly+60}" class="ledger-key">对象</text>')
    svg.append(tex_box(88, ly + 44, 850, 26,
        r'$\texttt{SM70\_8x8x4\_F32F16F16F32\_TN/TT}$：Volta HMMA 硬件原子；$\mathbf{S}_{\text{acc}}$：单 Warp 负责 $8$ 行的 QK 累加器；$\texttt{o\_storage}[2][\texttt{kOElements}]$：单 Warp 负责 $16\text{行} \times 2$ 个 $d_{\text{chunk}}$ 的输出累加器。',
        align="left", size=11.2, color="#334155"))
    svg.append(f'<line x1="36" y1="{ly+74}" x2="944" y2="{ly+74}" stroke="#e2e8f0" stroke-width="1"/>')

    svg.append(f'<text x="38" y="{ly+98}" class="ledger-key">机制</text>')
    svg.append(tex_box(88, ly + 80, 850, 108,
        r'(i) 正交分工：QK 按行切分（$\text{Warp}_0$ 算 $0..7$ 行，$\text{Warp}_1$ 算 $8..15$ 行，避免重复计算 QK），Online Softmax 后通过 $\mathbf{sP}$ 与 $\texttt{row\_scale\_exchange}$ 同步，PV 转为按 $D$ 列切分（各管 $D/2$）。<br/>'
        r'(ii) Bank Conflict 消除：Volta $\texttt{HMMA.884}$ 以 $8$ 线程 Quadpair 访问，$\texttt{SmemLayoutK}$ 采用 $\text{Pitch} = 68\;(d_{\text{chunk}}+4)$ 并在每 $16$ 行额外偏移 $16\text{ half}$，使每个 Half-warp 均匀覆盖全部 $16$ 个 Bank Pair。<br/>'
        r'(iii) 时空复用：第 $2$ 个 K Stage 仅在 $\mathbf{P}$ 物化前存活，故越界借用 $\mathbf{sP}$ 起始区，将总 Smem 严控在 $45,568\text{ B}$。',
        align="left", size=11.1, color="#334155", wrap=True))

    svg.append(f'<text x="490" y="{h-12}" class="sig-line">1Cat-vLLM SM70 Split-D 双 Warp 正交分工与 45,568B Smem 内存布局 @ 1Cat-vLLM 源码精读</text>')
    svg.append('</svg>')
    (ASSETS_DIR / "fig03_sm70_splitd_pipeline.svg").write_text("\n".join(svg), encoding="utf-8")


def generate_fig04():
    """图4: GQA-packed Wide GEMM (6Q:1KV) 与 DeepSeek MLA 矩阵吸收张量流"""
    w, h = 980, 665
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%">', COMMON_DEFS]
    svg.append(f'<rect class="fig-bg" x="1" y="1" width="{w-2}" height="{h-2}" rx="4"/>')

    # Zone 1: Top Formula (True LaTeX)
    svg.append(tex_box(490, 6, 940, 32,
        r'$\displaystyle \text{GQA-Packed: } \left[\mathbf{Q}_1;\, \dots;\, \mathbf{Q}_G\right]_{(G\cdot M)\times d} \mathbf{K}^\top_{d\times N} \to \mathbf{S}_{(G\cdot M)\times N}, \qquad \text{MLA Absorbed: } \mathbf{q}_i \mathbf{K}_i^\top = \mathbf{q}_i \left(\mathbf{W}_i^{UK}\mathbf{c}_t^{KV}\right)^\top = \left(\mathbf{q}_i \mathbf{W}_i^{UK}\right)\!\left(\mathbf{c}_t^{KV}\right)^\top = \tilde{\mathbf{q}}_i \left(\mathbf{c}_t^{KV}\right)^\top$',
        align="center", size=13.8, color="#0f172a"))
    svg.append(tex_box(490, 38, 940, 22,
        r'—— 两条正交的 KV 访存突围路线：(a) 1Cat-vLLM GQA 行维度打包喂饱 SM70 Tensor Core；(b) MLA 低秩潜向量缓存 + 升维矩阵结合律吸收',
        align="center", size=12, color="#475569"))
    svg.append('<line x1="24" y1="64" x2="956" y2="64" stroke="#e2e8f0" stroke-width="1"/>')

    # Stage (a): GQA-packed Wide GEMM (prefill.cu)
    svg.append(tex_box(28, 70, 860, 24,
        r'<strong>(a) 1Cat-vLLM GQA-Packed Wide GEMM (<code>prefill.cu:L455</code>)：将同组 $G=6$ 个 Query 头沿行轴拼成宽矩阵 ($\texttt{kRows} = 6Q$)</strong>',
        align="left", size=12.8, color="#0f172a"))

    y_sym_a, y_shp_a = 198, 220

    # Unpacked 6 small Q heads vs Packed tall Q matrix
    for h_i in range(6):
        by = 102 + h_i * 15.5
        m_qh, qhw, qhh = draw_matrix_grid(52, by, 1, 6, 8.5, 8.5, 1.5, "orange", bracket=False)
        svg.append(m_qh)
        svg.append(tex_box(31, by - 4, 30, 16, rf'$\mathbf{{Q}}_{h_i}$', size=9.8, color="#64748b"))
    svg.append(tex_box(82, y_sym_a, 110, 20, r'$6$ 个独立窄头', size=12.5, weight="700"))
    svg.append(tex_box(82, y_shp_a, 130, 18, r'各自读 KV $6$ 次 ($M$ 极窄)', size=10.2, color="#64748b"))

    # Arrow with plenty of clearance above/below text
    svg.append('<line x1="118" y1="146" x2="184" y2="146" class="arrow-teal"/>')
    svg.append(tex_box(151, 120, 66, 20, r'行轴 Pack', size=10.5, color="#334155", weight="700"))
    svg.append(tex_box(151, 150, 76, 18, r'$M_{\text{pack}} = 6Q$', size=10.2, color="#0f766e"))

    # Packed tall Q (6x6)
    m_qp, qpw, qph = draw_matrix_grid(194, 102, 6, 6, 11, 13, 2, "orange", pattern="stripe_row")
    svg.append(m_qp)
    svg.append(tex_box(194 + qpw / 2, y_sym_a, 90, 20, r'$\mathbf{Q}_{\text{packed}}$', size=12.8))
    svg.append(tex_box(194 + qpw / 2, y_shp_a, 86, 18, r'$(6Q \times 256)$', size=10.2, color="#64748b"))

    svg.append(tex_box(286, 138, 22, 20, r'$\times$', size=14, color="#1e293b"))

    # Shared Single KV head (6x5)
    m_ks, ksw, ksh = draw_matrix_grid(304, 108, 6, 5, 11, 11, 2, "teal")
    svg.append(m_ks)
    svg.append(tex_box(304 + ksw / 2, y_sym_a, 90, 20, r'$\mathbf{K}_{\text{shared}}^\top$', size=12.8))
    svg.append(tex_box(304 + ksw / 2, y_shp_a, 106, 18, r'$(256 \times N)$ 仅读 $1$ 次', size=10.2, color="#64748b"))

    svg.append('<line x1="378" y1="146" x2="418" y2="146" class="arrow"/>')

    m_spack, spw, sph = draw_matrix_grid(428, 102, 6, 5, 11, 13, 2, "violet", pattern="stripe_row")
    svg.append(m_spack)
    svg.append(tex_box(428 + spw / 2, y_sym_a, 86, 20, r'$\mathbf{S}_{\text{packed}}$', size=12.8))
    svg.append(tex_box(428 + spw / 2, y_shp_a, 80, 18, r'$(6Q \times N)$', size=10.2, color="#64748b"))

    svg.append(tex_box(508, 138, 22, 20, r'$\times$', size=14, color="#1e293b"))

    m_vs, vsw, vsh = draw_matrix_grid(526, 108, 5, 6, 11, 11, 2, "green")
    svg.append(m_vs)
    svg.append(tex_box(526 + vsw / 2, y_sym_a, 86, 20, r'$\mathbf{V}_{\text{shared}}$', size=12.8))
    svg.append(tex_box(526 + vsw / 2, y_shp_a, 106, 18, r'$(N \times 256)$ 仅读 $1$ 次', size=10.2, color="#64748b"))

    svg.append('<line x1="612" y1="146" x2="648" y2="146" class="arrow"/>')

    m_opack, opw, oph = draw_matrix_grid(658, 102, 6, 6, 11, 13, 2, "orange", pattern="stripe_row")
    svg.append(m_opack)
    svg.append(tex_box(658 + opw / 2, y_sym_a, 86, 20, r'$\mathbf{O}_{\text{packed}}$', size=12.8))
    svg.append(tex_box(658 + opw / 2, y_shp_a, 86, 18, r'$(6Q \times 256)$', size=10.2, color="#64748b"))

    svg.append('<rect x="754" y="96" width="202" height="142" class="note-card"/>')
    svg.append('<text x="766" y="115" class="note-title">17.9 → 60.8+ TFLOP/s 跃升根因</text>')
    svg.append(tex_box(766, 122, 182, 108,
        r'1. KV 显存带宽立省 $5/6$ (片上复用 $6\times$)；<br/>'
        r'2. $M$ 轴扩大 $6$ 倍，彻底消除 Volta $\texttt{mma.sync.m8n8k4}$ 小矩阵发射饥饿，喂饱 CUTLASS 流水线！',
        align="left", size=10.6, color="#334155", wrap=True))

    # Stage (b): MLA Low-Rank Joint Compression & Matrix Absorption
    svg.append(tex_box(28, 258, 880, 24,
        r'<strong>(b) DeepSeek MLA 低秩潜向量缓存与推理期矩阵吸收：在 HBM 中只存 $\left[\mathbf{c}_t^{KV} \;\middle|\; \mathbf{k}_t^R\right]$，不展开多头 $\mathbf{K}, \mathbf{V}$</strong>',
        align="left", size=12.8, color="#0f172a"))

    y_sym_b, y_shp_b = 360, 382

    # Input h_t
    m_ht, htw, hth = draw_matrix_grid(42, 314, 2, 7, 10, 10, 1.5, "gray")
    svg.append(m_ht)
    svg.append(tex_box(42 + htw / 2, y_sym_b, 60, 20, r'$\mathbf{h}_t$', size=13))
    svg.append(tex_box(42 + htw / 2, y_shp_b, 96, 18, r'$(B, T, d_{\text{model}})$', size=10.2, color="#64748b"))

    svg.append('<line x1="128" y1="325" x2="176" y2="325" class="arrow"/>')
    svg.append(tex_box(152, 299, 52, 20, r'$\mathbf{W}^{DKV}$', size=10.5, color="#475569"))

    # Compressed KV Cache [c_t^KV (4 cols) | k_t^R (1 col)]
    m_ckv, ckw, ckh = draw_matrix_grid(186, 306, 3, 4, 11, 11, 2, "teal", bracket=False)
    m_kr, krw, krh = draw_matrix_grid(186 + ckw + 4, 306, 3, 1, 11, 11, 2, "coral", bracket=False)
    svg.append(m_ckv)
    svg.append(m_kr)
    svg.append('<path class="bracket" d="M 187 303 L 183 303 L 183 346 L 187 346"/>')
    svg.append(f'<path class="bracket" d="M {186+ckw+4+krw-1} 303 L {186+ckw+4+krw+3} 303 L {186+ckw+4+krw+3} 346 L {186+ckw+4+krw-1} 346"/>')
    svg.append(tex_box(186 + (ckw + krw) / 2, y_sym_b, 110, 22, r'$\left[\mathbf{c}_t^{KV} \;\middle|\; \mathbf{k}_t^R\right]$', size=12.2))
    svg.append(tex_box(186 + (ckw + krw) / 2, y_shp_b, 120, 18, r'仅存 $(512 + 64)$ 维', size=10.2, color="#64748b"))

    # Connector arrow to Weight Absorption box
    svg.append('<line x1="258" y1="325" x2="302" y2="325" class="arrow-teal"/>')
    svg.append(tex_box(280, 301, 44, 18, r'点积', size=10.2, color="#0f766e"))

    # Weight Absorption branch
    svg.append('<rect x="308" y="292" width="380" height="106" rx="3" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.2"/>')
    svg.append('<text x="498" y="312" class="op-text">推理期矩阵结合律吸收 (Weight Absorbing)</text>')
    svg.append(tex_box(498, 318, 360, 28,
        r'$\displaystyle \mathbf{q}_i\,(1 \times d_h) \;\cdot\; \mathbf{W}_i^{UK}\,(d_h \times d_c) \;\longrightarrow\; \tilde{\mathbf{q}}_i\,(1 \times d_c)$',
        align="center", size=13, color="#0f172a"))
    svg.append(tex_box(498, 348, 364, 20,
        r'无需把 $\mathbf{c}_t^{KV}$ 升维成 $128$ 头 $\mathbf{K}$（省去 $30,720$ 维片上展开）！',
        align="center", size=10.6, color="#334155"))
    svg.append(tex_box(498, 370, 364, 20,
        r'直接用吸收后的 $\tilde{\mathbf{q}}_i\;(512\text{维})$ 与缓存的 $\mathbf{c}_t^{KV}$ 做点积',
        align="center", size=10.6, color="#334155"))

    # Decoupled RoPE note card
    svg.append('<rect x="704" y="292" width="252" height="106" class="note-card"/>')
    svg.append('<text x="716" y="311" class="note-title">为什么 RoPE 必须解耦外挂？</text>')
    svg.append(tex_box(716, 318, 232, 74,
        r'若对 $\mathbf{K}$ 直接施加旋转矩阵 $\mathcal{R}_t$，中间夹入依赖位置 $t$ 的 $\mathcal{R}_{s-t}$ 会阻断 $\mathbf{q}_i$ 与 $\mathbf{W}_i^{UK}$ 静态结合！故拆出 $1$ 个共享头 $\mathbf{k}_t^R\;(64\text{维})$ 专职携带 RoPE。',
        align="left", size=10.6, color="#334155", wrap=True))

    # Zone 3: Bottom Meaning Ledger
    ly = 424
    svg.append(f'<rect x="24" y="{ly}" width="932" height="202" class="ledger-box"/>')
    svg.append(f'<text x="38" y="{ly+24}" class="ledger-key">轴</text>')
    svg.append(tex_box(88, ly + 8, 850, 26,
        r'$G = n_q / n_{kv}$：单组共享同一 KV 头的 Query 头数（1Cat-vLLM v37 中 $G=6$）；$d_c = 512$：MLA KV 低秩压缩潜空间维；$d_R = 64$：解耦 RoPE 单头维。',
        align="left", size=11.2, color="#334155"))
    svg.append(f'<line x1="36" y1="{ly+38}" x2="944" y2="{ly+38}" stroke="#e2e8f0" stroke-width="1"/>')

    svg.append(f'<text x="38" y="{ly+60}" class="ledger-key">对象</text>')
    svg.append(tex_box(88, ly + 44, 850, 26,
        r'$\mathbf{Q}_{\text{packed}} \in \mathbb{R}^{6Q \times 256}$：沿 $M$ 轴连续排布的 $6$ 头 Query；$\mathbf{c}_t^{KV} \in \mathbb{R}^{512}$：无位置编码联合压缩潜向量；$\mathbf{k}_t^R \in \mathbb{R}^{64}$：全头广播共享位置键；$\tilde{\mathbf{q}}_i = \mathbf{q}_i\mathbf{W}_i^{UK}$：吸收后查询。',
        align="left", size=11.2, color="#334155"))
    svg.append(f'<line x1="36" y1="{ly+74}" x2="944" y2="{ly+74}" stroke="#e2e8f0" stroke-width="1"/>')

    svg.append(f'<text x="38" y="{ly+98}" class="ledger-key">机制</text>')
    svg.append(tex_box(88, ly + 80, 850, 110,
        r'(i) GQA 算子层优化：利用同一 Group 内 $G$ 个 Q 头共享同一 $(\mathbf{K}, \mathbf{V})$ 的代数结构，将 $G$ 次窄矩阵乘 $(Q \times d)\cdot(d \times N)$ 合并为 $1$ 次宽矩阵乘 $(GQ \times d)\cdot(d \times N)$，既消除 $(G{-}1)$ 次冗余 HBM 读取，又解决 Volta Tensor Core 行维度不足的发射瓶颈。<br/>'
        r'(ii) MLA 架构层优化：通过低秩瓶颈 $\mathbf{c}_t^{KV}$ 将每 Token 缓存压至 $576$ 维，并借助矩阵乘法结合律把升维权重 $\mathbf{W}_i^{UK}, \mathbf{W}_i^{UV}$ 分别吸收到 Query 端与输出端。',
        align="left", size=11.1, color="#334155", wrap=True))

    svg.append(f'<text x="490" y="{h-12}" class="sig-line">GQA-Packed 宽矩阵打包算子与 MLA 低秩矩阵吸收机制对比 @ 1Cat-vLLM 源码精读</text>')
    svg.append('</svg>')
    (ASSETS_DIR / "fig04_gqa_wide_gemm_and_mla.svg").write_text("\n".join(svg), encoding="utf-8")


def audit_svg_text_collisions(svg_path: Path):
    """静态扫描 SVG 中所有 <text> 与 <foreignObject> 节点的 Bounding Box 是否互相重叠、是否压矩阵格子或箭头线"""
    import xml.etree.ElementTree as ET
    tree = ET.parse(svg_path)
    root = tree.getroot()
    label_boxes = []
    tile_boxes = []
    arrow_lines = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "text":
            x = float(elem.attrib.get("x", "0"))
            y = float(elem.attrib.get("y", "0"))
            cls = elem.attrib.get("class", "")
            anchor = elem.attrib.get("text-anchor", "")
            if not anchor and cls in ("sym-label", "shape-label", "code-pill", "op-text", "sig-line"):
                anchor = "middle"
            txt = "".join(elem.itertext()).strip()
            est_w = sum(11.2 if ord(ch) > 127 else 6.2 for ch in txt)
            est_h = 13.0
            x0 = x - est_w / 2 if anchor == "middle" else x
            x1 = x0 + est_w
            y0 = y - est_h + 2
            y1 = y + 2
            label_boxes.append((x0, y0, x1, y1, txt[:32]))
        elif tag == "foreignObject":
            x0 = float(elem.attrib.get("x", "0"))
            y0 = float(elem.attrib.get("y", "0"))
            w = float(elem.attrib.get("width", "0"))
            h = float(elem.attrib.get("height", "0"))
            txt = "".join(elem.itertext()).strip()
            label_boxes.append((x0, y0, x0 + w, y0 + h, txt[:32]))
        elif tag == "rect":
            w = float(elem.attrib.get("width", "0"))
            h = float(elem.attrib.get("height", "0"))
            if 5 <= w <= 20 and 5 <= h <= 20:
                rx = float(elem.attrib.get("x", "0"))
                ry = float(elem.attrib.get("y", "0"))
                tile_boxes.append((rx, ry, rx + w, ry + h))
        elif tag == "line":
            cls = elem.attrib.get("class", "")
            if "arrow" in cls:
                x1 = float(elem.attrib.get("x1", "0"))
                y1 = float(elem.attrib.get("y1", "0"))
                x2 = float(elem.attrib.get("x2", "0"))
                y2 = float(elem.attrib.get("y2", "0"))
                arrow_lines.append((min(x1, x2), min(y1, y2) - 2, max(x1, x2), max(y1, y2) + 2))

    collisions = []
    for i in range(len(label_boxes)):
        for j in range(i + 1, len(label_boxes)):
            a = label_boxes[i]
            b = label_boxes[j]
            if a[0] < b[2] - 2 and a[2] > b[0] + 2 and a[1] < b[3] - 2 and a[3] > b[1] + 2:
                collisions.append(f"Label-Label: '{a[4]}' <--> '{b[4]}'")

    for a in label_boxes:
        for t in tile_boxes:
            if a[0] < t[2] and a[2] > t[0] and a[1] < t[3] and a[3] > t[1]:
                collisions.append(f"Label-Tile: '{a[4]}' overlaps matrix cell at ({t[0]},{t[1]})")
                break
        for ln in arrow_lines:
            if a[0] < ln[2] and a[2] > ln[0] and a[1] < ln[3] and a[3] > ln[1]:
                collisions.append(f"Label-Arrow: '{a[4]}' overlaps arrow at ({ln[0]}..{ln[2]}, {ln[1]})")

    if collisions:
        print(f"[WARN] {svg_path.name} has {len(collisions)} collisions:")
        for c in collisions:
            print("  -", c)
    else:
        print(f"[AUDIT PASS] {svg_path.name}: 0 collisions across {len(label_boxes)} LaTeX/text boxes & {len(tile_boxes)} matrix cells.")


if __name__ == "__main__":
    generate_fig01()
    generate_fig02()
    generate_fig03()
    generate_fig04()
    for p in sorted(ASSETS_DIR.glob("*.svg")):
        audit_svg_text_collisions(p)
