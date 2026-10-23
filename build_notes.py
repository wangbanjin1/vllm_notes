#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1Cat-vLLM 学习笔记构建脚本 (wdkns-skills 讲义风格版，纯 Python 标准库，零外部依赖)
支持：
1. 首页 Titlepage + tcolorbox 仓库元数据框 (YAML Frontmatter)
2. 四大动态标题高亮盒：importantbox (黄)、knowledgebox (蓝悬浮标)、warningbox (红)、dialoguebox (绿)
3. LaTeX lstlisting 风格代码块（带行号、底部 Listing 编号图注与源码行号溯源）
4. 自动编号配图（图 N + 底部源码定位/机制溯源角标）
5. 单行与多行 $$...$$ 独立公式块渲染
"""

import os
import sys
import re
import html
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent
MD_DIR = BASE_DIR / "markdown"
HTML_DIR = BASE_DIR / "html"
TEMPLATE_FILE = BASE_DIR / "templates" / "template.html"


def parse_inline(text: str) -> str:
    # 保护行内公式 $...$
    math_placeholders = []
    def save_math(m):
        math_placeholders.append(m.group(0))
        return f"__MATH_{len(math_placeholders)-1}__"

    text = re.sub(r'(?<!\\)\$([^\$\n]+?)(?<!\\)\$', save_math, text)

    # 行内代码 `code`
    code_placeholders = []
    def save_code(m):
        code_text = html.escape(m.group(1))
        code_placeholders.append(f"<code>{code_text}</code>")
        return f"__CODE_{len(code_placeholders)-1}__"

    text = re.sub(r'`([^`\n]+)`', save_code, text)

    # 粗体 **bold**
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # 斜体 *italic*
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    # 链接 [text](url)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)

    for i, c in enumerate(code_placeholders):
        text = text.replace(f"__CODE_{i}__", c)

    for i, m in enumerate(math_placeholders):
        text = text.replace(f"__MATH_{i}__", m)

    return text


def make_slug(text: str, used_ids: dict) -> str:
    clean = re.sub(r'[`\*_#\[\]\(\)\$\<\>\&\"]', '', text).strip()
    slug = re.sub(r'[\s\t\n]+', '-', clean)
    slug = re.sub(r'[^\w\u4e00-\u9fa5\-]+', '', slug).strip('-')
    if not slug:
        slug = "section"
    base = slug
    count = used_ids.get(base, 0)
    used_ids[base] = count + 1
    if count > 0:
        return f"{base}-{count}"
    return base


def extract_frontmatter(md_text: str) -> tuple[dict, str]:
    """解析顶部 --- ... --- 元数据块"""
    lines = md_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, md_text
    meta = {}
    end_idx = -1
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
        m = re.match(r'^([a-zA-Z0-9_]+)\s*:\s*(.*)$', lines[idx])
        if m:
            k, v = m.group(1).strip(), m.group(2).strip().strip('"').strip("'")
            meta[k] = v
    if end_idx == -1:
        return {}, md_text
    rest = "\n".join(lines[end_idx + 1:])
    return meta, rest


def render_titlepage(meta: dict) -> str:
    if not meta:
        return ""
    series = meta.get("series", "1Cat-vLLM 源码精读中文讲义")
    title = meta.get("title", "课程讲义")
    subtitle = meta.get("subtitle", "")
    authors = meta.get("authors", "五道口纳什风格 & Antigravity")
    date = meta.get("date", "2026-10-23")

    meta_rows = []
    if meta.get("repo"):
        meta_rows.append(f'<div class="lecture-meta-row"><strong>目标仓库 / 分支</strong>：{parse_inline(meta["repo"])}</div>')
    if meta.get("hardware"):
        meta_rows.append(f'<div class="lecture-meta-row"><strong>目标硬件架构</strong>：{parse_inline(meta["hardware"])}</div>')
    if meta.get("source_files"):
        meta_rows.append(f'<div class="lecture-meta-row"><strong>核心源码定位</strong>：{parse_inline(meta["source_files"])}</div>')
    if meta.get("topics"):
        meta_rows.append(f'<div class="lecture-meta-row"><strong>核心问题主线</strong>：{parse_inline(meta["topics"])}</div>')
    meta_rows.append(
        '<div class="lecture-meta-row"><strong>讲义制作范式</strong>：'
        '<a href="https://github.com/wdkns/wdkns-skills">wdkns-skills</a> '
        '(<code>codebase-render-notes</code> &amp; <code>tensor-formula-viz</code>)</div>'
    )

    subtitle_html = f'<div class="lecture-subtitle">{parse_inline(subtitle)}</div>' if subtitle else ""

    return (
        '<div class="lecture-titlepage">\n'
        f'  <div class="lecture-series-tag">{html.escape(series)}</div>\n'
        f'  <h1 class="lecture-main-title">{parse_inline(title)}</h1>\n'
        f'  {subtitle_html}\n'
        f'  <div class="lecture-author-date">整理：{html.escape(authors)} &nbsp;·&nbsp; 日期：{html.escape(date)}</div>\n'
        '  <div class="lecture-meta-box">\n'
        + "\n".join("    " + r for r in meta_rows) + "\n"
        '  </div>\n'
        '</div>'
    )


def md_to_html(md_text: str) -> tuple[str, str, str]:
    meta, body_md = extract_frontmatter(md_text)
    lines = body_md.splitlines()
    html_out = []
    title = meta.get("title", "学习笔记")

    if meta:
        html_out.append(render_titlepage(meta))

    in_code_block = False
    code_lang = ""
    code_caption = ""
    code_source = ""
    code_buf = []

    in_math_block = False
    math_buf = []

    in_table = False
    table_buf = []

    in_list = False
    list_type = "ul"

    in_callout = False
    callout_type = "knowledge"
    callout_title = "背景知识"
    callout_buf = []
    callout_in_list = False
    callout_list_type = "ul"

    used_ids = {}
    headings = []
    fig_counter = 0
    listing_counter = 0

    def flush_list():
        nonlocal in_list, html_out
        if in_list:
            html_out.append(f"</{list_type}>")
            in_list = False

    def flush_table():
        nonlocal in_table, table_buf, html_out
        if in_table and table_buf:
            html_out.append("<table>")
            is_header = True
            for row in table_buf:
                cols = [c.strip() for c in row.strip("|").split("|")]
                if any(re.match(r'^:?-+:?$', c) for c in cols):
                    is_header = False
                    continue
                tag = "th" if is_header else "td"
                html_out.append("  <tr>" + "".join(f"<{tag}>{parse_inline(c)}</{tag}>" for c in cols) + "</tr>")
                if is_header:
                    is_header = False
            html_out.append("</table>")
            table_buf = []
            in_table = False

    def flush_callout():
        nonlocal in_callout, callout_buf, callout_in_list, html_out
        if in_callout:
            if callout_in_list:
                callout_buf.append(f"</{callout_list_type}>")
                callout_in_list = False
            inner_html = "\n".join(callout_buf)
            html_out.append(f'<div class="tcolorbox tcolorbox-{callout_type}">')
            html_out.append(f'  <div class="tcolorbox-title">{parse_inline(callout_title)}</div>')
            html_out.append(f'  <div class="tcolorbox-body">\n{inner_html}\n  </div>')
            html_out.append('</div>')
            callout_buf = []
            in_callout = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # 1. 代码块 ```lang | caption=... | source=...
        if line.strip().startswith("```"):
            if not in_code_block:
                flush_list()
                flush_table()
                flush_callout()
                in_code_block = True
                header_spec = line.strip()[3:].strip()
                parts = [p.strip() for p in header_spec.split("|")]
                code_lang = parts[0].lower() if parts else ""
                code_caption = ""
                code_source = ""
                for p in parts[1:]:
                    if p.startswith("caption="):
                        code_caption = p[len("caption="):].strip()
                    elif p.startswith("source="):
                        code_source = p[len("source="):].strip()
                code_buf = []
            else:
                in_code_block = False
                escaped = html.escape("\n".join(code_buf))
                display_lang = code_lang.upper() if code_lang else "CODE"
                hl_class = f"language-{code_lang}" if code_lang else "language-plaintext"
                num_lines = max(1, len(code_buf))
                line_nums_html = "<br>".join(str(n) for n in range(1, num_lines + 1))

                caption_html = ""
                if code_caption:
                    listing_counter += 1
                    caption_html = f'<div class="lstlisting-caption">Listing {listing_counter}: {parse_inline(code_caption)}</div>'
                prov_html = ""
                if code_source:
                    prov_html = f'<div class="lstlisting-provenance">源码定位：{parse_inline(code_source)}</div>'

                block_html = (
                    f'<div class="lstlisting-wrapper">\n'
                    f'  <div class="code-block">\n'
                    f'    <div class="code-header">\n'
                    f'      <span class="code-lang">{display_lang}</span>\n'
                    f'      <button class="copy-btn" onclick="copyCode(this)" title="复制代码">\n'
                    f'        <span>复制</span>\n'
                    f'      </button>\n'
                    f'    </div>\n'
                    f'    <div class="code-table-wrap">\n'
                    f'      <div class="line-numbers">{line_nums_html}</div>\n'
                    f'      <pre><code class="{hl_class}">{escaped}</code></pre>\n'
                    f'    </div>\n'
                    f'  </div>\n'
                    f'  {caption_html}\n'
                    f'  {prov_html}\n'
                    f'</div>'
                )
                html_out.append(block_html)
                code_buf = []
            i += 1
            continue

        if in_code_block:
            code_buf.append(line)
            i += 1
            continue

        # 2. 多行独立公式块 $$
        if line.strip() == "$$":
            if not in_math_block:
                flush_list()
                flush_table()
                flush_callout()
                in_math_block = True
                math_buf = ["$$"]
            else:
                in_math_block = False
                math_buf.append("$$")
                html_out.append(f'<div class="math-display-block">{" ".join(math_buf)}</div>')
                math_buf = []
            i += 1
            continue

        if in_math_block:
            math_buf.append(line.strip())
            i += 1
            continue

        # 3. 空行
        if not line.strip():
            flush_list()
            flush_table()
            if in_callout and callout_in_list:
                callout_buf.append(f"</{callout_list_type}>")
                callout_in_list = False
            i += 1
            continue

        # 4. Callout 处理 > [!IMPORTANT|KNOWLEDGE|NOTE|TIP|WARNING|CAUTION|DIALOGUE] 标题
        callout_m = re.match(r'^>\s*\[!(IMPORTANT|KNOWLEDGE|NOTE|TIP|WARNING|CAUTION|DIALOGUE)\](.*)', line, re.IGNORECASE)
        if callout_m:
            flush_list()
            flush_table()
            flush_callout()
            in_callout = True
            ctype_raw = callout_m.group(1).upper()
            callout_type = {
                "IMPORTANT": "important",
                "KNOWLEDGE": "knowledge",
                "NOTE": "knowledge",
                "TIP": "knowledge",
                "WARNING": "warning",
                "CAUTION": "warning",
                "DIALOGUE": "dialogue"
            }.get(ctype_raw, "knowledge")
            default_titles = {
                "important": "核心结论",
                "knowledge": "背景与直觉",
                "warning": "常见误区与陷阱",
                "dialogue": "核心讨论片段"
            }
            custom_t = callout_m.group(2).strip()
            callout_title = custom_t if custom_t else default_titles[callout_type]
            i += 1
            continue

        if in_callout and line.startswith(">"):
            content = line.lstrip(">").strip()
            if not content:
                if callout_in_list:
                    callout_buf.append(f"</{callout_list_type}>")
                    callout_in_list = False
                i += 1
                continue
            c_list_m = re.match(r'^(\*|-|\d+\.)\s+(.*)', content)
            if c_list_m:
                marker, ltext = c_list_m.groups()
                curr_lt = "ol" if marker[0].isdigit() else "ul"
                if not callout_in_list or callout_list_type != curr_lt:
                    if callout_in_list:
                        callout_buf.append(f"</{callout_list_type}>")
                    callout_in_list = True
                    callout_list_type = curr_lt
                    callout_buf.append(f"<{callout_list_type}>")
                callout_buf.append(f"  <li>{parse_inline(ltext)}</li>")
            elif content.startswith("$$") and content.endswith("$$"):
                if callout_in_list:
                    callout_buf.append(f"</{callout_list_type}>")
                    callout_in_list = False
                callout_buf.append(f'<div class="math-display-block">{content}</div>')
            else:
                if callout_in_list:
                    callout_buf.append(f"</{callout_list_type}>")
                    callout_in_list = False
                callout_buf.append(f"<p>{parse_inline(content)}</p>")
            i += 1
            continue
        elif in_callout and not line.startswith(">"):
            flush_callout()

        # 5. 普通引用块
        if line.startswith(">"):
            flush_list()
            flush_table()
            html_out.append(f"<blockquote>{parse_inline(line.lstrip('> '))}</blockquote>")
            i += 1
            continue

        # 6. 独立配图行 ![caption|provenance](path)
        fig_m = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)\s*$', line.strip())
        if fig_m:
            flush_list()
            flush_table()
            flush_callout()
            fig_counter += 1
            alt_raw, img_src_raw = fig_m.group(1).strip(), fig_m.group(2).strip()
            img_src = "../" + img_src_raw if img_src_raw.startswith("assets/") else img_src_raw
            if "|" in alt_raw:
                f_cap, f_prov = [x.strip() for x in alt_raw.split("|", 1)]
            else:
                f_cap, f_prov = alt_raw, ""
            prov_div = f'  <div class="lecture-fig-provenance">{parse_inline(f_prov)}</div>\n' if f_prov else ""

            local_svg_path = BASE_DIR / img_src_raw
            if img_src_raw.lower().endswith(".svg") and local_svg_path.exists():
                svg_raw = local_svg_path.read_text(encoding="utf-8")
                # Make marker IDs unique per figure so multiple inline SVGs never clash
                svg_raw = re.sub(r'id="(arrowhead[^"]*)"', rf'id="fig{fig_counter}_\1"', svg_raw)
                svg_raw = re.sub(r'url\(#(arrowhead[^)]*)\)', rf'url(#fig{fig_counter}_\1)', svg_raw)
                media_html = f'  <div class="inline-svg-wrap">{svg_raw}</div>\n'
            else:
                media_html = f'  <img src="{html.escape(img_src)}" alt="{html.escape(f_cap)}">\n'

            html_out.append(
                f'<figure class="lecture-figure">\n'
                f'{media_html}'
                f'  <figcaption class="lecture-figcaption">图 {fig_counter}: {parse_inline(f_cap)}</figcaption>\n'
                f'{prov_div}'
                f'</figure>'
            )
            i += 1
            continue

        # 7. 表格 | ... |
        if line.strip().startswith("|") and line.strip().endswith("|"):
            flush_list()
            in_table = True
            table_buf.append(line.strip())
            i += 1
            continue
        else:
            flush_table()

        # 8. 标题 # ~ ######
        heading_match = re.match(r'^(#{1,6})\s+(.*)', line)
        if heading_match:
            flush_list()
            flush_table()
            flush_callout()
            level = len(heading_match.group(1))
            htext = heading_match.group(2).strip()
            if level == 1 and title == "学习笔记":
                title = re.sub(r'[`\*_#\[\]\(\)\$]', '', htext).strip()

            slug = make_slug(htext, used_ids)
            html_out.append(f'<h{level} id="{slug}">{parse_inline(htext)}</h{level}>')

            if level in (2, 3, 4):
                headings.append({
                    "level": level,
                    "text": htext,
                    "id": slug
                })
            i += 1
            continue

        # 9. 水平分割线
        if re.match(r'^(-{3,}|\*{3,}|_{3,})$', line.strip()):
            flush_list()
            i += 1
            continue

        # 10. 列表项 (- 或 1.)
        list_match = re.match(r'^\s*(\*|-|\d+\.)\s+(.*)', line)
        if list_match:
            marker, text = list_match.groups()
            curr_type = "ol" if marker[0].isdigit() else "ul"
            if not in_list or list_type != curr_type:
                flush_list()
                in_list = True
                list_type = curr_type
                html_out.append(f"<{list_type}>")

            if text.startswith("[ ] "):
                text = '<input type="checkbox" disabled style="margin-right: 6px;"> ' + text[4:]
            elif text.startswith("[x] "):
                text = '<input type="checkbox" checked disabled style="margin-right: 6px;"> ' + text[4:]

            html_out.append(f"  <li>{parse_inline(text)}</li>")
            i += 1
            continue
        else:
            flush_list()

        # 11. 单行独立公式块 $$ ... $$
        if line.strip().startswith("$$") and line.strip().endswith("$$") and len(line.strip()) > 2:
            html_out.append(f'<div class="math-display-block">{line.strip()}</div>')
            i += 1
            continue

        # 12. 普通段落
        html_out.append(f"<p>{parse_inline(line)}</p>")
        i += 1

    flush_list()
    flush_table()
    flush_callout()

    toc_items = []
    for h in headings:
        clean_text = re.sub(r'[`\*_#\[\]\(\)\$]', '', h["text"]).strip()
        escaped_title = html.escape(clean_text)
        inline_html = parse_inline(h["text"])
        toc_items.append(
            f'  <li class="toc-item toc-level-{h["level"]}">'
            f'<a href="#{h["id"]}" class="toc-link" title="{escaped_title}">{inline_html}</a></li>'
        )

    if toc_items:
        toc_html = '<ul class="toc-list">\n' + '\n'.join(toc_items) + '\n</ul>'
    else:
        toc_html = '<div class="toc-empty">暂无章节大纲</div>'

    return title, "\n".join(html_out), toc_html


def build_all():
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
        template = f.read()

    md_files = sorted(MD_DIR.glob("*.md"))
    print(f"[*] 发现 {len(md_files)} 个 Markdown 源文件")

    articles = []

    for md_path in md_files:
        print(f" -> 正在转换: {md_path.name}")
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        title, body_html, toc_html = md_to_html(md_content)
        rel_path = f"markdown/{md_path.name}"

        page_html = template.replace("{{TITLE}}", title)
        page_html = page_html.replace("{{BODY_CONTENT}}", body_html)
        page_html = page_html.replace("{{TOC_CONTENT}}", toc_html)
        page_html = page_html.replace("{{SOURCE_REL_PATH}}", rel_path)

        out_html_path = HTML_DIR / f"{md_path.stem}.html"
        with open(out_html_path, "w", encoding="utf-8") as f:
            f.write(page_html)

        articles.append({
            "title": title,
            "filename": f"{md_path.stem}.html",
            "md_name": md_path.name
        })

    build_index(articles)
    print("[OK] All notes built successfully in wdkns-skills lecture style!")


def build_index(articles):
    index_path = BASE_DIR / "index.html"

    categories = {
        "核心系统与硬件架构实战": [],
        "知识点汇总": []
    }

    for item in articles:
        if item['md_name'].startswith("knowledge_") or "知识点汇总" in item['title']:
            categories["知识点汇总"].append(item)
        else:
            categories["核心系统与硬件架构实战"].append(item)

    sections_html = []
    category_icons = {
        "核心系统与硬件架构实战": "⚡",
        "知识点汇总": "💡"
    }

    for cat_name, cat_articles in categories.items():
        if not cat_articles:
            continue
        items_html = []
        for item in cat_articles:
            items_html.append(f'''
            <li style="margin-bottom: 14px; padding: 16px 20px; background: var(--surface-color); border: 1.5px solid var(--border-color); border-radius: 2px;">
                <a href="html/{item['filename']}" style="font-size: 1.15rem; font-weight: 700; color: var(--accent-primary); text-decoration: none;">
                    {item['title']}
                </a>
                <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 6px;">
                    源文件：<code>markdown/{item['md_name']}</code> · 讲义目标：<code>html/{item['filename']}</code> · 风格：<code>wdkns-skills</code>
                </div>
            </li>
            ''')

        sections_html.append(f'''
        <section style="margin-bottom: 36px;">
            <h2 style="font-size: 1.35rem; font-weight: 700; margin-bottom: 16px; border-bottom: 2px solid var(--border-color); padding-bottom: 8px; color: var(--text-primary);">
                {category_icons.get(cat_name, '📌')} {cat_name}
            </h2>
            <ul style="list-style: none; padding: 0; margin: 0;">
                {"".join(items_html)}
            </ul>
        </section>
        ''')

    index_html = f'''<!DOCTYPE html>
<html lang="zh-CN" data-theme="paper">
<head>
    <meta charset="UTF-8">
    <title>1Cat-vLLM 源码精读讲义导航 (wdkns-skills 风格)</title>
    <style>
        :root[data-theme="paper"] {{
            --bg-color: #ffffff;
            --surface-color: #f8fafc;
            --border-color: #cbd5e1;
            --text-primary: #111827;
            --text-muted: #64748b;
            --accent-primary: #1d4ed8;
            --code-bg: #f1f5f9;
        }}
        body {{
            font-family: "Noto Serif CJK SC", -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            line-height: 1.8;
            padding: 50px 24px;
        }}
        .container {{ max-width: 880px; margin: 0 auto; }}
        code {{ background: var(--code-bg); padding: 2px 6px; border-radius: 3px; font-size: 0.88em; font-family: "JetBrains Mono", Consolas, monospace; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 style="margin-bottom: 8px; font-size: 2.05rem;">1Cat-vLLM 源码精读中文讲义导航</h1>
        <p style="color: var(--text-muted); margin-bottom: 35px;">
            讲义全面采用 <code>wdkns-skills</code>（五步教学法 + <code>tcolorbox</code> 视觉路标 + <code>tensor-formula-viz</code> 三段式张量图 + 源码精确行号溯源）构建。
        </p>
        {"".join(sections_html)}
    </div>
</body>
</html>
'''
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)


if __name__ == "__main__":
    build_all()
