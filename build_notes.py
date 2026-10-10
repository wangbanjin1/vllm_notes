#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1Cat-vLLM 学习笔记构建脚本 (纯 Python 标准库，零外部依赖)
将 learn_notes/markdown/*.md 转换为具备护眼主题与公式支持的 HTML 页面
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
    # 保护已转义的行内公式 $...$
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

    # 还原代码
    for i, c in enumerate(code_placeholders):
        text = text.replace(f"__CODE_{i}__", c)

    # 还原行内公式
    for i, m in enumerate(math_placeholders):
        text = text.replace(f"__MATH_{i}__", m)

    return text


def make_slug(text: str, used_ids: dict) -> str:
    # 移除 Markdown / 行内语法及标点符号
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


def md_to_html(md_text: str) -> tuple[str, str, str]:
    """简单的纯 Python Markdown 解析器，支持代码块、表格、Callouts、公式与大纲目录"""
    lines = md_text.splitlines()
    html_out = []
    title = "学习笔记"
    
    in_code_block = False
    code_lang = ""
    code_buf = []

    in_table = False
    table_buf = []

    in_list = False
    list_type = "ul"

    in_callout = False
    callout_type = "note"
    callout_buf = []

    used_ids = {}
    headings = []

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
                # 忽略分隔线行 |---|---|
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
        nonlocal in_callout, callout_buf, html_out
        if in_callout:
            c_title = {"note": "说明与背景", "warning": "核心挑战与限制", "important": "核心公式与结论"}.get(callout_type, "提示")
            inner_html = "\n".join(callout_buf)
            html_out.append(f'<div class="callout callout-{callout_type}">')
            html_out.append(f'  <div class="callout-title">{c_title}</div>')
            html_out.append(inner_html)
            html_out.append('</div>')
            callout_buf = []
            in_callout = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # 1. 代码块 ```
        if line.strip().startswith("```"):
            if not in_code_block:
                flush_list()
                flush_table()
                flush_callout()
                in_code_block = True
                code_lang = line.strip()[3:].strip().lower()
                code_buf = []
            else:
                in_code_block = False
                escaped = html.escape("\n".join(code_buf))
                display_lang = code_lang.upper() if code_lang else "TEXT"
                hl_class = f"language-{code_lang}" if code_lang else "language-plaintext"
                block_html = (
                    f'<div class="code-block">\n'
                    f'  <div class="code-header">\n'
                    f'    <span class="code-lang">{display_lang}</span>\n'
                    f'    <button class="copy-btn" onclick="copyCode(this)" title="复制代码">\n'
                    f'      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">\n'
                    f'        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>\n'
                    f'        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>\n'
                    f'      </svg>\n'
                    f'      <span>复制</span>\n'
                    f'    </button>\n'
                    f'  </div>\n'
                    f'  <pre><code class="{hl_class}">{escaped}</code></pre>\n'
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

        # 2. 空行
        if not line.strip():
            flush_list()
            flush_table()
            if in_callout:
                callout_buf.append("<p></p>")
            i += 1
            continue

        # 3. Callout 处理 > [!NOTE], > [!WARNING], > [!IMPORTANT]
        callout_m = re.match(r'^>\s*\[!(NOTE|WARNING|IMPORTANT|TIP)\](.*)', line, re.IGNORECASE)
        if callout_m:
            flush_list()
            flush_table()
            flush_callout()
            in_callout = True
            ctype_raw = callout_m.group(1).upper()
            callout_type = {"NOTE": "note", "WARNING": "warning", "IMPORTANT": "important", "TIP": "note"}.get(ctype_raw, "note")
            extra = callout_m.group(2).strip()
            if extra:
                callout_buf.append(f"<p>{parse_inline(extra)}</p>")
            i += 1
            continue

        if in_callout and line.startswith(">"):
            content = line.lstrip("> ").strip()
            if content.startswith("$$") and content.endswith("$$"):
                callout_buf.append(f'<div style="text-align:center; margin: 12px 0;">{content}</div>')
            elif content:
                callout_buf.append(f"<p>{parse_inline(content)}</p>")
            i += 1
            continue
        elif in_callout and not line.startswith(">"):
            flush_callout()

        # 4. 普通引用块
        if line.startswith(">"):
            flush_list()
            flush_table()
            html_out.append(f"<blockquote>{parse_inline(line.lstrip('> '))}</blockquote>")
            i += 1
            continue

        # 5. 表格 | ... |
        if line.strip().startswith("|") and line.strip().endswith("|"):
            flush_list()
            in_table = True
            table_buf.append(line.strip())
            i += 1
            continue
        else:
            flush_table()

        # 6. 标题 # ~ ######
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

        # 7. 水平分割线
        if re.match(r'^(-{3,}|\*{3,}|_{3,})$', line.strip()):
            flush_list()
            html_out.append("<hr style='border: 0; border-top: 1px solid var(--border-color); margin: 32px 0;'>")
            i += 1
            continue

        # 8. 列表项 (- 或 1.)
        list_match = re.match(r'^\s*(\*|-|\d+\.)\s+(.*)', line)
        if list_match:
            marker, text = list_match.groups()
            curr_type = "ol" if marker[0].isdigit() else "ul"
            if not in_list or list_type != curr_type:
                flush_list()
                in_list = True
                list_type = curr_type
                html_out.append(f"<{list_type}>")
            
            # 复选框支持 - [ ] / - [x]
            if text.startswith("[ ] "):
                text = f'<input type="checkbox" disabled style="margin-right: 6px;"> ' + text[4:]
            elif text.startswith("[x] "):
                text = f'<input type="checkbox" checked disabled style="margin-right: 6px;"> ' + text[4:]
            
            html_out.append(f"  <li>{parse_inline(text)}</li>")
            i += 1
            continue
        else:
            flush_list()

        # 9. 独立公式块 $$ ... $$
        if line.strip().startswith("$$") and line.strip().endswith("$$") and len(line.strip()) > 2:
            html_out.append(f'<div style="text-align: center; margin: 18px 0; font-size: 1.15em;">{line.strip()}</div>')
            i += 1
            continue

        # 10. 普通段落
        html_out.append(f"<p>{parse_inline(line)}</p>")
        i += 1

    flush_list()
    flush_table()
    flush_callout()

    # 构建左侧大纲 HTML
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

    # 生成 index.html 总目录
    build_index(articles)
    print("[OK] All notes built successfully!")


def build_index(articles):
    index_path = BASE_DIR / "index.html"
    
    # 按栏目分类
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
            <li style="margin-bottom: 14px; padding: 14px 18px; background: var(--surface-color); border: 1px solid var(--border-color); border-radius: 8px; transition: border-color 0.2s ease;">
                <a href="html/{item['filename']}" style="font-size: 1.15rem; font-weight: 600; color: var(--accent-primary); text-decoration: none;">
                    {item['title']}
                </a>
                <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 6px;">
                    源文件：<code>markdown/{item['md_name']}</code> · 生成目标：<code>html/{item['filename']}</code>
                </div>
            </li>
            ''')
        
        sections_html.append(f'''
        <section style="margin-bottom: 36px;">
            <h2 style="font-size: 1.35rem; font-weight: 700; margin-bottom: 16px; border-bottom: 2px solid var(--border-color); padding-bottom: 8px; color: var(--accent-primary);">
                {category_icons.get(cat_name, '📌')} {cat_name}
            </h2>
            <ul style="list-style: none; padding: 0; margin: 0;">
                {"".join(items_html)}
            </ul>
        </section>
        ''')

    index_html = f'''<!DOCTYPE html>
<html lang="zh-CN" data-theme="parchment">
<head>
    <meta charset="UTF-8">
    <title>1Cat-vLLM 学习笔记与知识库索引</title>
    <style>
        :root[data-theme="parchment"] {{
            --bg-color: #f7f4ec;
            --surface-color: #eee9dc;
            --border-color: #dcd4c0;
            --text-primary: #2d3748;
            --text-muted: #798696;
            --accent-primary: #a35d28;
            --code-bg: #eae3d2;
            --toggle-bg: #dfd6c2;
        }}
        :root[data-theme="twilight"] {{
            --bg-color: #1a1e24;
            --surface-color: #232931;
            --border-color: #39424e;
            --text-primary: #dce1e8;
            --text-muted: #707b88;
            --accent-primary: #e09f67;
            --code-bg: #15181d;
            --toggle-bg: #323b47;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            line-height: 1.8;
            padding: 50px 24px;
        }}
        .container {{ max-width: 860px; margin: 0 auto; }}
        code {{ background: var(--code-bg); padding: 2px 6px; border-radius: 4px; font-size: 0.88em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 style="margin-bottom: 8px; font-size: 2rem;">1Cat-vLLM 源码精读与知识库导航</h1>
        <p style="color: var(--text-muted); margin-bottom: 35px;">
            笔记体系已划分为「核心系统与硬件架构实战」与「知识点汇总」两大独立栏目。修改 <code>markdown/*.md</code> 后运行 <code>build_notes.py</code> 即可重新编译生成。
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
