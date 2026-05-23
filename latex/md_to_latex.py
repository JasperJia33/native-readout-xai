#!/usr/bin/env python3
"""
md_to_latex.py — Convert the journal-manuscript Markdown report to a
well-formatted LaTeX document suitable for compilation with TeX Live
(lualatex or xelatex).

Usage:
    python md_to_latex.py \\
        --input  outputs/markdown_journal/report.md \\
        --output latex/report.tex

The converter is hand-rolled (not pandoc) so figure widths, table
formatting, and equation numbering follow the project conventions.

Figure width policy (chosen per figure based on aspect ratio and
logical content):
  - fig_00 (Figure 1: class distribution)              --> 0.55
  - fig_01 (Figure 2: sample wafer maps, 4×9 grid)    --> 0.55
  - fig_02 (Figure 3: F1 radar chart)                  --> 0.55
  - fig_03 (Figure 4a: ResNet explain_class4)          --> 0.98
  - fig_04 (Figure 4b: DenseNet explain_class4)        --> 0.98
  - fig_05 (Figure 4c: ViT explain_class4)             --> 0.98
  - fig_06 (Figure 5: qualitative heatmap panel, 9×3)  --> 0.98
  - fig_07 (Figure 6: Del/Ins curves, 2 panels)        --> 0.98
  - fig_08 (Figure 7: stability boxplot)               --> 0.60
  - fig_09 (Supp S1: training curves, 3 panels)        --> 0.60
  - fig_10 (Supp S2: raw heatmaps, no overlay)         --> 0.98

All figures are horizontally centered via the `figure` environment
with `\\centering`. All tables use `booktabs` and are wrapped in a
centered `table` environment.
"""
from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path


# -----------------------------------------------------------------------------
# Figure width policy
# -----------------------------------------------------------------------------
FIGURE_WIDTHS = {
    "fig_00.png": 0.55,   # Figure 1: class distribution
    "fig_01.png": 0.55,   # Figure 2: sample wafer maps
    "fig_02.png": 0.55,   # Figure 3: F1 radar
    "fig_03.png": 0.98,   # Figure 4a: ResNet explain_class4
    "fig_04.png": 0.98,   # Figure 4b: DenseNet explain_class4
    "fig_05.png": 0.98,   # Figure 4c: ViT explain_class4
    "fig_06.png": 0.60,   # Figure 5: qualitative heatmap panel
    "fig_07.png": 0.98,   # Figure 6: Del/Ins curves
    "fig_08.png": 0.60,   # Figure 7: stability boxplot
    "fig_09.png": 0.70,   # Supp S1: training curves
    "fig_10.png": 0.60,   # Supp S2: raw heatmaps
}
DEFAULT_WIDTH = 0.70


# -----------------------------------------------------------------------------
# LaTeX escape / inline text handling
# -----------------------------------------------------------------------------
_ESCAPE_MAP = {
    "\\": r"\textbackslash{}",
    "&":  r"\&",
    "%":  r"\%",
    "$":  r"\$",
    "#":  r"\#",
    "_":  r"\_",
    "{":  r"\{",
    "}":  r"\}",
    "~":  r"\textasciitilde{}",
    "^":  r"\textasciicircum{}",
}


def latex_escape(s: str) -> str:
    """Escape plain text for LaTeX. Does NOT process markdown or math."""
    out = []
    for ch in s:
        out.append(_ESCAPE_MAP.get(ch, ch))
    return "".join(out)


# Inline-markdown transformations applied AFTER math segments are
# protected. Order matters: bold before italic.
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_RE = re.compile(r"(?<![*])\*([^*\n]+)\*(?![*])")
_EMPH_UNDER_RE = re.compile(r"(?<![A-Za-z0-9_])_([^_\n]+)_(?![A-Za-z0-9_])")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _strip_inline_html(s: str) -> str:
    """Convert a small set of inline HTML tags to markdown equivalents.

    Handles <i>, <b>, <em>, <strong>, <br>, and strips other tags.
    """
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = re.sub(r"<(i|em)>(.*?)</(i|em)>", r"*\2*", s, flags=re.I | re.DOTALL)
    s = re.sub(r"<(b|strong)>(.*?)</(b|strong)>", r"**\2**",
               s, flags=re.I | re.DOTALL)
    # Strip any remaining tags.
    s = re.sub(r"</?[a-zA-Z][^>]*>", "", s)
    return s


def convert_inline(text: str) -> str:
    """Convert a span of markdown (no block structures) to LaTeX.

    Handles: inline `$...$` math, inline code, bold, italic, underscores as
    emphasis, markdown links. Plain text outside math is escaped.
    """
    # Protect math segments ($...$ but not $$ and not already inside \text).
    pieces = []
    i = 0
    n = len(text)
    placeholders = []
    while i < n:
        if text[i] == "$":
            j = text.find("$", i + 1)
            if j == -1:
                pieces.append(text[i:])
                break
            math = text[i:j + 1]
            placeholders.append(math)
            pieces.append(f"\x00MATH{len(placeholders) - 1}\x00")
            i = j + 1
        else:
            pieces.append(text[i])
            i += 1
    protected = "".join(pieces)

    # Protect inline code (treat contents literally).
    def _code_sub(m):
        placeholders.append(("code", m.group(1)))
        return f"\x00CODE{len(placeholders) - 1}\x00"

    protected = _INLINE_CODE_RE.sub(_code_sub, protected)

    # Protect markdown links (will re-insert as \href).
    def _link_sub(m):
        placeholders.append(("link", m.group(1), m.group(2)))
        return f"\x00LINK{len(placeholders) - 1}\x00"

    protected = _MD_LINK_RE.sub(_link_sub, protected)

    # Extract bold / italic tokens BEFORE escaping, as markers we can
    # safely re-insert after escape. Inner text will be recursively
    # processed by _process_text (below) so nested bold/italic work.
    def _bold_sub(m):
        placeholders.append(("bold", m.group(1)))
        return f"\x00BOLD{len(placeholders) - 1}\x00"

    def _ital_sub(m):
        placeholders.append(("italic", m.group(1)))
        return f"\x00ITAL{len(placeholders) - 1}\x00"

    protected = _BOLD_RE.sub(_bold_sub, protected)
    protected = _ITALIC_RE.sub(_ital_sub, protected)
    protected = _EMPH_UNDER_RE.sub(_ital_sub, protected)

    # Escape remaining plain text.
    escaped = latex_escape(protected)

    # Re-insert placeholders, closing over the LOCAL `placeholders` list.
    pattern = re.compile(r"\x00(MATH|CODE|LINK|BOLD|ITAL)(\d+)\x00")

    def _resolve_marker(tag, idx):
        """Return LaTeX for a single placeholder."""
        p = placeholders[idx]
        if tag == "MATH":
            return p
        if isinstance(p, tuple):
            if p[0] == "code":
                return r"\texttt{" + latex_escape(p[1]) + "}"
            if p[0] == "link":
                label, url = p[1], p[2]
                # If the label is a bare placeholder like "link" / "链接",
                # display the actual URL so a print reader can see/copy it.
                # \url{} handles most special chars automatically via url.sty.
                if label.strip().lower() in ("link", "链接"):
                    return r"\url{" + url + "}"
                return r"\href{" + url + "}{" + latex_escape(label) + "}"
            if p[0] == "bold":
                # Inner text may contain markers that refer to THIS scope's
                # placeholders list. Resolve those before recursing so the
                # recursive call sees clean input.
                inner = pattern.sub(
                    lambda m: _resolve_marker(m.group(1), int(m.group(2))),
                    p[1],
                )
                return r"\textbf{" + convert_inline(inner) + "}"
            if p[0] == "italic":
                inner = pattern.sub(
                    lambda m: _resolve_marker(m.group(1), int(m.group(2))),
                    p[1],
                )
                return r"\emph{" + convert_inline(inner) + "}"
        return ""

    def _reinsert(match):
        return _resolve_marker(match.group(1), int(match.group(2)))

    # Resolve all placeholders in this scope.
    return pattern.sub(_reinsert, escaped)


# -----------------------------------------------------------------------------
# HTML table parser (for DataFrame HTML dumps)
# -----------------------------------------------------------------------------
class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []              # list[list[(tag, text)]] for thead+tbody
        self.current_row = None
        self.current_cell = None
        self.current_tag = None
        self.in_thead = False
        self.in_tbody = False
        self.thead_rows = []
        self.tbody_rows = []

    def handle_starttag(self, tag, attrs):
        if tag == "thead":
            self.in_thead = True
        elif tag == "tbody":
            self.in_tbody = True
        elif tag == "tr":
            self.current_row = []
        elif tag in ("th", "td"):
            self.current_cell = []
            self.current_tag = tag

    def handle_endtag(self, tag):
        if tag == "thead":
            self.in_thead = False
        elif tag == "tbody":
            self.in_tbody = False
        elif tag == "tr":
            if self.current_row is not None:
                if self.in_thead:
                    self.thead_rows.append(self.current_row)
                else:
                    self.tbody_rows.append(self.current_row)
            self.current_row = None
        elif tag in ("th", "td"):
            if self.current_cell is not None and self.current_row is not None:
                text = "".join(self.current_cell).strip()
                self.current_row.append((self.current_tag, text))
            self.current_cell = None
            self.current_tag = None

    def handle_data(self, data):
        if self.current_cell is not None:
            self.current_cell.append(data)


def html_tables_to_latex(html: str) -> str:
    """Convert a <div>...<table>...</table>...</div> block to a LaTeX tabular.

    Returns a LaTeX `table` environment string. The caller is expected to
    emit a preceding `\\caption` via the <center><b>Table N.</b>...</center>
    marker, which we capture separately as `caption`.
    """
    # Extract table block.
    m = re.search(r"<table[^>]*>(.*?)</table>", html, re.DOTALL)
    if not m:
        return ""
    parser = _TableParser()
    parser.feed(m.group(0))

    # Flatten header rows: the pandas HTML dump commonly has two thead rows
    # (column names on row 1, an empty row 2 where the <th> names the index).
    headers = []
    if parser.thead_rows:
        # Merge pair of header rows where second row is the index-name row.
        if len(parser.thead_rows) == 1:
            headers = [c[1] for c in parser.thead_rows[0]]
        else:
            # If the second row contains a non-empty th in column 0 and
            # all other cells empty, treat it as the index-name row.
            first, second = parser.thead_rows[0], parser.thead_rows[1]
            if (len(second) > 0 and second[0][0] == "th"
                    and all(c[1] == "" for c in second[1:])):
                # Replace the leading blank header with the index name.
                merged = [(c[0], c[1]) for c in first]
                if merged and merged[0][1] == "" and second[0][1]:
                    merged[0] = ("th", second[0][1])
                headers = [c[1] for c in merged]
            else:
                headers = [c[1] for c in first]

    body = [[c[1] for c in row] for row in parser.tbody_rows]

    # If the table has a pandas row-numbering column as the first column
    # (where header is "" and body cells are "0", "1", ...), drop it.
    if headers and headers[0] == "" and body:
        if all(row and row[0].isdigit() for row in body):
            headers = headers[1:]
            body = [row[1:] for row in body]

    if not headers:
        return ""

    ncols = len(headers)
    # Column format: first column left-aligned (often Family), remaining
    # right-aligned for numeric-looking columns, left-aligned for text.
    # A column is "numeric" if all its values start with a digit, minus
    # sign, or '[' (confidence interval), or are wrapped in parentheses.
    def _is_numeric_col(col_idx):
        for row in body:
            if col_idx >= len(row):
                continue
            v = row[col_idx].strip()
            if not v:
                continue
            # Accept leading digit, minus sign, bracket, or 0.xxx ± y.zzz style
            first = v.lstrip('[(').strip()[:1]
            if first and (first.isdigit() or first in '-+.'):
                continue
            return False
        return True

    col_fmt = "l"
    for i in range(1, ncols):
        col_fmt += "r" if _is_numeric_col(i) else "l"

    # Wide tables: wrap in \resizebox so they fit \linewidth.
    wide = ncols >= 6

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    # caption is inserted by caller
    lines.append(r"%CAPTION%")
    if wide:
        lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(convert_inline(h) for h in headers) + r" \\")
    lines.append(r"\midrule")
    for row in body:
        lines.append(" & ".join(convert_inline(v) for v in row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if wide:
        lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main Markdown -> LaTeX conversion
# -----------------------------------------------------------------------------
FIG_LINE_RE = re.compile(r"!\[\]\(images/([^)]+)\)")
DISPLAY_MATH_RE = re.compile(r"^\$\$(.+?)\$\$\s*$")
CAPTION_RE = re.compile(
    r"<center>\s*<b>(Figure|Table|Supplementary Figure|Supplementary Table|"
    r"Supplementary Diagram|Diagram|"
    r"图|表|补充图|补充表|补充示意图|示意图)"
    r"\s*(S?\d+[a-z]?)\.</b>\s*(.*?)</center>",
    re.DOTALL,
)


def convert(md: str) -> str:
    """Convert the full markdown document to LaTeX body text."""
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)

    # Title + subtitle are the first `# Title` and the first `### subtitle`
    # following it; we capture them to insert as \title / subtitle.
    title = None
    subtitle = None
    # Track whether we are inside the Supplementary section so that
    # "### Sx. Heading" preserves the explicit "Sx." label rather than
    # relying on LaTeX's section counter (which otherwise renders ".1"
    # after a starred section).
    in_supplementary = False
    # Track whether we are inside the Supplementary section so that
    # "### Sx. Heading" keeps its explicit "Sx." label.
    in_supplementary = False

    # Walk through blocks.
    while i < n:
        line = lines[i]
        stripped = line.rstrip()

        # --- Top-level title ---
        if stripped.startswith("# ") and title is None:
            title = stripped[2:].strip()
            i += 1
            continue

        # --- Subtitle (### before any ## section) ---
        if stripped.startswith("### ") and subtitle is None and all(
                not lines[j].startswith("## ") for j in range(i)):
            subtitle = stripped[4:].strip()
            i += 1
            continue

        # --- Horizontal rule ---
        if stripped == "---":
            i += 1
            continue

        # --- Section headings ---
        if stripped.startswith("#### "):
            body = stripped[5:].strip()
            # Strip leading numeric prefix like "3.2.1 " since LaTeX numbers.
            body = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", body)
            out.append(r"\subsubsection{" + convert_inline(body) + "}")
            i += 1
            continue
        if stripped.startswith("### "):
            body = stripped[4:].strip()
            # Sections like "### Abstract" / "### 摘要"
            if body.lower() == "abstract" or body.strip() == "摘要":
                out.append(r"\begin{abstract}")
                # Consume following paragraph(s) until next heading or hr.
                j = i + 1
                abs_lines = []
                while j < n:
                    nxt = lines[j]
                    if (nxt.startswith("#") or nxt.strip() == "---"
                            or nxt.startswith("<") or nxt.startswith("```")):
                        break
                    abs_lines.append(nxt)
                    j += 1
                paragraph = " ".join(ln.strip() for ln in abs_lines).strip()
                if paragraph:
                    out.append(convert_inline(paragraph))
                out.append(r"\end{abstract}")
                i = j
                continue
            if (body.strip().startswith("参考文献")
                    or body.lower().startswith("references")):
                out.append(r"\subsection*{" + convert_inline(body) + "}")
            elif in_supplementary:
                # Preserve explicit supplementary numbering: the source uses
                # "### S1. Experimental workflow", "### S2. Code overview",
                # etc. Emit as a starred (unnumbered) subsection so the
                # "Sx." prefix in the heading is not overridden by LaTeX.
                m_sup = re.match(r"^(S\d+)\.\s*(.*)$", body)
                if m_sup:
                    label, rest = m_sup.group(1), m_sup.group(2)
                    # Insert a page break before "S2.", "S6.", and "S9." for readability.
                    if label in ("S2", "S6", "S9"):
                        out.append(r"\clearpage")
                    heading = label + ". " + convert_inline(rest)
                    out.append(r"\phantomsection")
                    out.append(r"\subsection*{" + heading + "}")
                    out.append(r"\addcontentsline{toc}{subsection}{" +
                               heading + "}")
                else:
                    out.append(r"\phantomsection")
                    out.append(r"\subsection*{" + convert_inline(body) + "}")
            else:
                # Strip leading numeric prefix like "1.1 Motivation".
                stripped_body = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", body)
                # Strip supplementary "S1. "-style prefix.
                stripped_body = re.sub(r"^S\d+\.\s+", "", stripped_body)
                # Page break before §5.6 for readability.
                if re.match(r"^5\.6\b", body):
                    out.append(r"\clearpage")
                out.append(r"\subsection{" + convert_inline(stripped_body) + "}")
            i += 1
            continue
        if stripped.startswith("## "):
            body = stripped[3:].strip()
            if body.lower() in ("supplementary", "supplementary material",
                                "补充材料"):
                in_supplementary = True
                # Page break so the Supplementary starts on a fresh page.
                out.append(r"\clearpage")
                out.append(r"\appendix")
                # \phantomsection prevents a duplicate-anchor hyperref warning
                # between the starred section and its first starred subsection.
                out.append(r"\phantomsection")
                out.append(r"\section*{" + convert_inline(body) + "}")
                out.append(r"\addcontentsline{toc}{section}{" +
                           convert_inline(body) + "}")
            else:
                # Strip leading "N." prefix.
                stripped_body = re.sub(r"^\d+\.\s+", "", body)
                out.append(r"\section{" + convert_inline(stripped_body) + "}")
            i += 1
            continue

        # --- Figures ---
        mfig = FIG_LINE_RE.match(stripped)
        if mfig:
            img = mfig.group(1)
            width = FIGURE_WIDTHS.get(img, DEFAULT_WIDTH)
            # Look ahead for the figure caption (next non-empty line).
            j = i + 1
            while j < n and lines[j].strip() == "":
                j += 1
            caption_line = lines[j] if j < n else ""
            mc = CAPTION_RE.search(caption_line)
            if mc:
                kind, num, rest = mc.group(1), mc.group(2), mc.group(3)
                rest = _strip_inline_html(rest.strip()).rstrip(".。")
                caption_bold = r"\textbf{" + kind + " " + num + ".}"
                caption_text = (caption_bold + " " + convert_inline(rest))
                i = j + 1
            else:
                caption_text = ""
            out.append(r"\begin{figure}[htbp]")
            out.append(r"\centering")
            out.append(
                r"\includegraphics[width=" + f"{width:.2f}" +
                r"\linewidth]{" + img + "}"
            )
            if caption_text:
                out.append(r"\caption{" + caption_text + "}")
            out.append(r"\end{figure}")
            i += 1
            continue

        # --- Standalone caption (for the following HTML table) ---
        mc = CAPTION_RE.search(stripped)
        if mc:
            kind, num, rest = mc.group(1), mc.group(2), mc.group(3)
            rest = _strip_inline_html(rest.strip()).rstrip(".。")
            caption_bold = r"\textbf{" + kind + " " + num + ".}"
            caption_text = (caption_bold + " " + convert_inline(rest))
            # Peek ahead for <div>...<table>...</table>...</div>
            j = i + 1
            while j < n and lines[j].strip() == "":
                j += 1
            if j < n and lines[j].strip().startswith("<div>"):
                # Accumulate HTML until closing </div> that balances depth.
                html_parts = []
                depth = 0
                k = j
                while k < n:
                    html_parts.append(lines[k])
                    depth += lines[k].count("<div>") - lines[k].count("</div>")
                    if depth == 0 and "</div>" in lines[k]:
                        break
                    k += 1
                html = "\n".join(html_parts)
                tex_table = html_tables_to_latex(html)
                tex_table = tex_table.replace(
                    "%CAPTION%",
                    r"\caption{" + caption_text + "}"
                )
                out.append(tex_table)
                i = k + 1
                continue
            # Standalone caption without a following HTML table. Try to
            # retrofit onto the most-recently-emitted orphan table (which
            # placed a %CAPTION% marker) so the caption attaches correctly.
            for back in range(len(out) - 1, -1, -1):
                if "%CAPTION%" in out[back]:
                    out[back] = out[back].replace(
                        "%CAPTION%",
                        r"\caption{" + caption_text + "}"
                    )
                    break
            else:
                # Otherwise emit as a standalone centered bold paragraph.
                out.append(r"\begin{center}")
                out.append(r"\small " + caption_text)
                out.append(r"\end{center}")
            i += 1
            continue

        # --- Orphan HTML table (no preceding caption). Leaves %CAPTION%
        #     marker in place so a subsequent standalone caption can be
        #     retrofitted onto this table (see caption branch above).
        if stripped.startswith("<div>"):
            html_parts = []
            depth = 0
            k = i
            while k < n:
                html_parts.append(lines[k])
                depth += lines[k].count("<div>") - lines[k].count("</div>")
                if depth == 0 and "</div>" in lines[k]:
                    break
                k += 1
            html = "\n".join(html_parts)
            tex_table = html_tables_to_latex(html)
            out.append(tex_table)
            i = k + 1
            continue

        # --- Display math ---
        mm = DISPLAY_MATH_RE.match(stripped)
        if mm:
            body = mm.group(1).strip()
            # Convert ``\tag{N}`` into equation-numbered equation.
            tag_m = re.search(r"\\tag\{([^}]+)\}", body)
            if tag_m:
                body_no_tag = body.replace(tag_m.group(0), "").strip()
                out.append(r"\begin{equation}")
                out.append(body_no_tag)
                out.append(r"\tag{" + tag_m.group(1) + "}")
                out.append(r"\end{equation}")
            else:
                out.append(r"\begin{equation*}")
                out.append(body)
                out.append(r"\end{equation*}")
            i += 1
            continue

        # --- Code fence ---
        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            j = i + 1
            code_lines = []
            while j < n and not lines[j].startswith("```"):
                code_lines.append(lines[j])
                j += 1
            # Use fancyvrb Verbatim for better styling + size control.
            out.append(r"\begin{Verbatim}")
            out.extend(code_lines)
            out.append(r"\end{Verbatim}")
            i = j + 1
            continue

        # --- Blank line ---
        if stripped == "":
            out.append("")
            i += 1
            continue

        # --- Unordered list ---
        if re.match(r"^\s*[-*]\s+", line):
            items = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                item = re.sub(r"^\s*[-*]\s+", "", lines[i])
                # continuation lines
                j = i + 1
                while (j < n and lines[j].strip() != ""
                       and not re.match(r"^\s*[-*]\s+", lines[j])
                       and not lines[j].startswith("#")
                       and not lines[j].startswith("<")
                       and not FIG_LINE_RE.match(lines[j].strip())):
                    item += " " + lines[j].strip()
                    j += 1
                items.append(item)
                i = j
            out.append(r"\begin{itemize}")
            for it in items:
                out.append(r"\item " + convert_inline(it))
            out.append(r"\end{itemize}")
            continue

        # --- Ordered list ---
        if re.match(r"^\s*\d+\.\s+", line):
            items = []  # list of (num, text)
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                m = re.match(r"^\s*(\d+)\.\s+(.*)$", lines[i])
                num = m.group(1)
                item = m.group(2)
                j = i + 1
                while (j < n and lines[j].strip() != ""
                       and not re.match(r"^\s*\d+\.\s+", lines[j])
                       and not lines[j].startswith("#")
                       and not lines[j].startswith("<")):
                    item += " " + lines[j].strip()
                    j += 1
                items.append((num, item))
                i = j
            # Use description-style so the explicit numbering in the source
            # (e.g. 13., 14., ..., 18. for the second reference group) is
            # preserved rather than restarted at 1 by LaTeX.
            out.append(r"\begin{list}{}{\setlength{\leftmargin}{2.5em}"
                       r"\setlength{\labelwidth}{2em}"
                       r"\setlength{\labelsep}{0.3em}"
                       r"\setlength{\itemsep}{0.2ex}}")
            for num, it in items:
                out.append(r"\item[" + num + ".] " + convert_inline(it))
            out.append(r"\end{list}")
            continue

        # --- Markdown pipe table ---
        if stripped.startswith("|") and "|" in stripped[1:]:
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            # Parse: first line = header, second = separator, rest = data
            if len(table_lines) >= 3:
                header_cells = [c.strip() for c in table_lines[0].split("|")[1:-1]]
                ncols = len(header_cells)
                # Determine alignment from separator row
                sep_cells = [c.strip() for c in table_lines[1].split("|")[1:-1]]
                aligns = []
                for sc in sep_cells:
                    if sc.startswith(":") and sc.endswith(":"):
                        aligns.append("c")
                    elif sc.endswith(":"):
                        aligns.append("r")
                    else:
                        aligns.append("l")
                col_fmt = "".join(aligns[:ncols])
                wide_table = ncols >= 6
                if wide_table:
                    out.append(r"\noindent\resizebox{\linewidth}{!}{%")
                out.append(r"\begin{tabular}{" + col_fmt + "}")
                out.append(r"\toprule")
                out.append(" & ".join(convert_inline(c) for c in header_cells) + r" \\")
                out.append(r"\midrule")
                for row_line in table_lines[2:]:
                    cells_raw = [c.strip() for c in row_line.split("|")[1:-1]]
                    out.append(" & ".join(convert_inline(c) for c in cells_raw[:ncols]) + r" \\")
                out.append(r"\bottomrule")
                out.append(r"\end{tabular}")
                if wide_table:
                    out.append(r"}")
                out.append("")
            else:
                # Fallback: just emit as text
                for tl in table_lines:
                    out.append(convert_inline(tl))
            continue

        # --- Paragraph: accumulate until blank line or block marker ---
        para_lines = [line]
        j = i + 1
        while j < n:
            nxt = lines[j]
            if nxt.strip() == "":
                break
            if (nxt.startswith("#") or nxt.startswith("<")
                    or nxt.startswith("```")
                    or nxt.strip() == "---"
                    or FIG_LINE_RE.match(nxt.strip())
                    or DISPLAY_MATH_RE.match(nxt.strip())
                    or re.match(r"^\s*[-*]\s+", nxt)
                    or re.match(r"^\s*\d+\.\s+", nxt)):
                break
            para_lines.append(nxt)
            j += 1
        paragraph = " ".join(ln.strip() for ln in para_lines)
        out.append(convert_inline(paragraph))
        i = j

    body_tex = "\n".join(out)
    # Any %CAPTION% markers that were never retrofitted belong to truly
    # orphan tables — remove the marker line so LaTeX compiles cleanly.
    body_tex = re.sub(r"^%CAPTION%\n", "", body_tex, flags=re.M)
    return body_tex, title, subtitle


# -----------------------------------------------------------------------------
# Preamble
# -----------------------------------------------------------------------------
PREAMBLE_EN = r"""\documentclass[11pt,a4paper]{article}

% ---------- encoding & language ----------
% pdflatex-safe: uses inputenc/fontenc instead of fontspec.
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[english]{babel}

% ---------- typography & layout ----------
\usepackage[margin=1in]{geometry}
\usepackage{setspace}
\setstretch{1.15}
\usepackage[expansion=false]{microtype}
\usepackage{parskip}
\widowpenalty=10000
\clubpenalty=10000
\raggedbottom

% ---------- math ----------
\usepackage{amsmath,amssymb,amsthm}
\usepackage{mathtools}

% ---------- text symbols ----------
\usepackage{textcomp}

% ---------- figures & tables ----------
\usepackage{graphicx}
% arXiv: all figures in same directory as .tex
\graphicspath{{./}}
\usepackage{float}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{caption}
% Disable automatic "Figure N.:" / "Table N.:" prefixes so the explicit
% numbering written in the captions (Table 1., Figure 2., Supplementary
% Table S1., etc.) matches the numbers used in the prose cross-references.
\captionsetup{font=small,labelformat=empty,justification=centering,
              singlelinecheck=false}

% ---------- hyperlinks & references ----------
\usepackage[hidelinks,breaklinks=true]{hyperref}
% Allow URLs to break at any character so long references don't overflow
% the margin (the default only breaks at "/" and a handful of punctuation).
\usepackage{xurl}

% ---------- section spacing ----------
\usepackage{titlesec}
\titlespacing*{\section}{0pt}{1.4ex plus .2ex}{0.8ex plus .2ex}
\titlespacing*{\subsection}{0pt}{1.0ex plus .2ex}{0.6ex plus .2ex}
\titlespacing*{\subsubsection}{0pt}{0.8ex plus .2ex}{0.4ex plus .2ex}

% ---------- verbatim blocks ----------
\usepackage{fancyvrb}
\fvset{fontsize=\footnotesize,frame=single,framesep=2mm}

% ---------- centered figures (belt-and-braces) ----------
\makeatletter
\g@addto@macro\@floatboxreset\centering
\makeatother
"""

PREAMBLE_ZH = r"""\documentclass[11pt,a4paper]{article}

% ---------- encoding & CJK language ----------
% Compile with xelatex. arXiv supports xelatex via 00README.XXX.
\usepackage{fontspec}
\usepackage{xeCJK}
% Use Fandol fonts (shipped with TeX Live, available on arXiv).
% Reference by filename so xelatex finds them via kpathsea without fontconfig.
\setCJKmainfont{FandolSong-Regular}[
  Extension=.otf,
  BoldFont=FandolSong-Bold]
\setCJKsansfont{FandolHei-Regular}[
  Extension=.otf,
  BoldFont=FandolHei-Bold]
\setCJKmonofont{FandolFang-Regular}[
  Extension=.otf]

% ---------- typography & layout ----------
\usepackage[margin=1in]{geometry}
\usepackage{setspace}
\setstretch{1.25}
\usepackage{parskip}
\widowpenalty=10000
\clubpenalty=10000
\raggedbottom

% ---------- math ----------
\usepackage{amsmath,amssymb,amsthm}
\usepackage{mathtools}

% ---------- unicode glyph coverage for non-CJK prose ----------
\usepackage{newunicodechar}
\newunicodechar{σ}{\ensuremath{\sigma}}
\newunicodechar{Σ}{\ensuremath{\Sigma}}
\newunicodechar{α}{\ensuremath{\alpha}}
\newunicodechar{β}{\ensuremath{\beta}}
\newunicodechar{γ}{\ensuremath{\gamma}}
\newunicodechar{δ}{\ensuremath{\delta}}
\newunicodechar{λ}{\ensuremath{\lambda}}
\newunicodechar{μ}{\ensuremath{\mu}}
\newunicodechar{ϵ}{\ensuremath{\epsilon}}
\newunicodechar{ε}{\ensuremath{\varepsilon}}
\newunicodechar{θ}{\ensuremath{\theta}}
\newunicodechar{π}{\ensuremath{\pi}}
\newunicodechar{ρ}{\ensuremath{\rho}}
\newunicodechar{τ}{\ensuremath{\tau}}
\newunicodechar{φ}{\ensuremath{\varphi}}
\newunicodechar{ψ}{\ensuremath{\psi}}
\newunicodechar{ω}{\ensuremath{\omega}}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{≠}{\ensuremath{\neq}}
\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{±}{\ensuremath{\pm}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{·}{\ensuremath{\cdot}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{←}{\ensuremath{\leftarrow}}
\newunicodechar{↔}{\ensuremath{\leftrightarrow}}
\newunicodechar{∞}{\ensuremath{\infty}}
\newunicodechar{∑}{\ensuremath{\sum}}
\newunicodechar{∏}{\ensuremath{\prod}}
\newunicodechar{∫}{\ensuremath{\int}}
\newunicodechar{∈}{\ensuremath{\in}}
\newunicodechar{∉}{\ensuremath{\notin}}
\newunicodechar{⊂}{\ensuremath{\subset}}
\newunicodechar{⊆}{\ensuremath{\subseteq}}
\newunicodechar{∪}{\ensuremath{\cup}}
\newunicodechar{∩}{\ensuremath{\cap}}
\newunicodechar{∅}{\ensuremath{\emptyset}}
\newunicodechar{↓}{\ensuremath{\downarrow}}
\newunicodechar{↑}{\ensuremath{\uparrow}}
\newunicodechar{—}{---}
\newunicodechar{–}{--}
\newunicodechar{¹}{\textsuperscript{1}}
\newunicodechar{²}{\textsuperscript{2}}
\newunicodechar{³}{\textsuperscript{3}}

% ---------- figures & tables ----------
\usepackage{graphicx}
\graphicspath{{./}}
\usepackage{float}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{caption}
\captionsetup{font=small,labelformat=empty,justification=centering,
              singlelinecheck=false}

% ---------- hyperlinks & references ----------
\usepackage[hidelinks,breaklinks=true]{hyperref}
\usepackage{xurl}

% ---------- section spacing ----------
\usepackage{titlesec}
\titlespacing*{\section}{0pt}{1.4ex plus .2ex}{0.8ex plus .2ex}
\titlespacing*{\subsection}{0pt}{1.0ex plus .2ex}{0.6ex plus .2ex}
\titlespacing*{\subsubsection}{0pt}{0.8ex plus .2ex}{0.4ex plus .2ex}

% ---------- verbatim blocks ----------
\usepackage{fancyvrb}
\fvset{fontsize=\footnotesize,frame=single,framesep=2mm}

% ---------- centered figures ----------
\makeatletter
\g@addto@macro\@floatboxreset\centering
\makeatother
"""


# Back-compat alias: PREAMBLE used by legacy callers.
PREAMBLE = PREAMBLE_EN


# -----------------------------------------------------------------------------
# Unicode -> LaTeX substitution for pdflatex safety
# -----------------------------------------------------------------------------
_UNICODE_TO_LATEX = {
    # Punctuation / typography
    '\u2014': '---',          # em-dash —
    '\u2013': '--',           # en-dash –
    '\u2212': '$-$',          # minus sign −
    '\u00a7': r'\S{}',        # section sign §
    '\u00b0': r'\textdegree{}',  # degree °
    '\u00bd': r'\textonehalf{}', # ½
    '\u00bc': r'\textonequarter{}', # ¼
    '\u00be': r'\textthreequarters{}', # ¾
    '\u00b9': r'\textsuperscript{1}',  # ¹
    '\u00b2': r'\textsuperscript{2}',  # ²
    '\u00b3': r'\textsuperscript{3}',  # ³
    '\u2713': r'$\checkmark$',  # ✓
    '\u2717': r'$\times$',      # ✗
    '\u2022': r'\textbullet{}', # •
    '\u2026': r'\ldots{}',      # …
    # Greek letters (text mode)
    '\u03b1': r'$\alpha$',    # α
    '\u03b2': r'$\beta$',     # β
    '\u03b3': r'$\gamma$',    # γ
    '\u03b4': r'$\delta$',    # δ
    '\u0394': r'$\Delta$',    # Δ
    '\u03b5': r'$\varepsilon$', # ε
    '\u03f5': r'$\epsilon$',  # ϵ
    '\u03b8': r'$\theta$',    # θ
    '\u03bb': r'$\lambda$',   # λ
    '\u03bc': r'$\mu$',       # μ
    '\u03c0': r'$\pi$',       # π
    '\u03c1': r'$\rho$',      # ρ
    '\u03c3': r'$\sigma$',    # σ
    '\u03a3': r'$\Sigma$',    # Σ
    '\u03c4': r'$\tau$',      # τ
    '\u03c6': r'$\varphi$',   # φ
    '\u03c8': r'$\psi$',      # ψ
    '\u03c9': r'$\omega$',    # ω
    # Math symbols (text mode)
    '\u2248': r'$\approx$',   # ≈
    '\u2260': r'$\neq$',      # ≠
    '\u2264': r'$\leq$',      # ≤
    '\u2265': r'$\geq$',      # ≥
    '\u00b1': r'$\pm$',       # ±
    '\u00d7': r'$\times$',    # ×
    '\u00b7': r'$\cdot$',     # ·
    '\u2192': r'$\rightarrow$',  # →
    '\u2190': r'$\leftarrow$',   # ←
    '\u2194': r'$\leftrightarrow$', # ↔
    '\u2191': r'$\uparrow$',     # ↑
    '\u2193': r'$\downarrow$',   # ↓
    '\u221e': r'$\infty$',    # ∞
    '\u2211': r'$\sum$',      # ∑
    '\u220f': r'$\prod$',     # ∏
    '\u222b': r'$\int$',      # ∫
    '\u2208': r'$\in$',       # ∈
    '\u2209': r'$\notin$',    # ∉
    '\u2282': r'$\subset$',   # ⊂
    '\u2286': r'$\subseteq$', # ⊆
    '\u222a': r'$\cup$',      # ∪
    '\u2229': r'$\cap$',      # ∩
    '\u2205': r'$\emptyset$', # ∅
    # Arrows used in diagrams
    '\u25b6': r'$\triangleright$',  # ▶
    '\u25c0': r'$\triangleleft$',   # ◀
    '\u25bc': r'$\triangledown$',   # ▼
    '\u25b2': r'$\triangle$',       # ▲
    # Box-drawing -> ASCII
    '\u250c': '+', '\u2510': '+', '\u2514': '+', '\u2518': '+',
    '\u251c': '+', '\u2524': '+', '\u252c': '+', '\u2534': '+',
    '\u253c': '+',
    '\u2500': '-', '\u2502': '|',
    '\u2550': '=', '\u2551': '|',
    # Accented characters (common in references)
    '\u00e9': r"\'e",         # é
    '\u00e8': r"\`e",         # è
    '\u00ea': r"\^e",         # ê
    '\u00e1': r"\'a",         # á
    '\u00e0': r"\`a",         # à
    '\u00e2': r"\^a",         # â
    '\u00f3': r"\'o",         # ó
    '\u00f2': r"\`o",         # ò
    '\u00f4': r"\^o",         # ô
    '\u00fc': r'\"u',         # ü
    '\u00fa': r"\'u",         # ú
    '\u00ed': r"\'i",         # í
    '\u00f1': r"\~n",         # ñ
    '\u00e7': r"\c{c}",       # ç
    '\u0107': r"\'c",         # ć
    '\u010d': r"\v{c}",       # č
    '\u0161': r"\v{s}",       # š
    '\u017e': r"\v{z}",       # ž
    '\u00e4': r'\"a',         # ä
    '\u00f6': r'\"o',         # ö
    '\u00e5': r'\aa{}',       # å
    '\u00c9': r"\'E",         # É
}


def sanitize_for_pdflatex(tex: str) -> str:
    """Replace Unicode characters with LaTeX commands for pdflatex safety.
    
    Skips content inside \\begin{Verbatim}...\\end{Verbatim} blocks,
    where only box-drawing -> ASCII substitutions are applied.
    """
    # Split on Verbatim blocks
    parts = re.split(r'(\\begin\{Verbatim\}.*?\\end\{Verbatim\})', tex, flags=re.DOTALL)
    result = []
    # Box-drawing only map (safe inside Verbatim)
    _BOX_ONLY = {
        '\u250c': '+', '\u2510': '+', '\u2514': '+', '\u2518': '+',
        '\u251c': '+', '\u2524': '+', '\u252c': '+', '\u2534': '+',
        '\u253c': '+',
        '\u2500': '-', '\u2502': '|',
        '\u2550': '=', '\u2551': '|',
        '\u25b6': '>', '\u25c0': '<', '\u25bc': 'v', '\u25b2': '^',
        '\u2014': '--', '\u2013': '-', '\u2212': '-',
        '\u2713': '[ok]',
        '\u2193': 'v', '\u2191': '^', '\u2192': '->', '\u2190': '<-',
        '\u00d7': 'x', '\u00b7': '.',
    }
    for i, part in enumerate(parts):
        if part.startswith('\\begin{Verbatim}'):
            # Inside Verbatim: only do box-drawing -> ASCII
            out = []
            for ch in part:
                out.append(_BOX_ONLY.get(ch, ch))
            result.append(''.join(out))
        else:
            # Normal text: full substitution
            out = []
            for ch in part:
                out.append(_UNICODE_TO_LATEX.get(ch, ch))
            result.append(''.join(out))
    return ''.join(result)


def wrap_long_lines(tex: str, max_len: int = 900) -> str:
    """Wrap lines longer than max_len at word boundaries."""
    result = []
    for line in tex.split('\n'):
        if len(line) <= max_len or line.startswith('\\'):
            result.append(line)
        else:
            # Wrap at spaces
            words = line.split(' ')
            current = ''
            for w in words:
                if current and len(current) + 1 + len(w) > max_len:
                    result.append(current)
                    current = w
                else:
                    current = current + ' ' + w if current else w
            if current:
                result.append(current)
    return '\n'.join(result)


def sanitize_verbatim_only(tex: str) -> str:
    """For xelatex/ZH: only replace box-drawing chars inside Verbatim blocks."""
    _BOX_ONLY = {
        '\u250c': '+', '\u2510': '+', '\u2514': '+', '\u2518': '+',
        '\u251c': '+', '\u2524': '+', '\u252c': '+', '\u2534': '+',
        '\u253c': '+',
        '\u2500': '-', '\u2502': '|',
        '\u2550': '=', '\u2551': '|',
        '\u25b6': '>', '\u25c0': '<', '\u25bc': 'v', '\u25b2': '^',
        '\u2193': 'v', '\u2191': '^', '\u2192': '->', '\u2190': '<-',
        '\u00d7': 'x', '\u00b7': '.',
        '\u2014': '--', '\u2013': '-', '\u2212': '-',
        '\u2713': '[ok]',
    }
    parts = re.split(r'(\\begin\{Verbatim\}.*?\\end\{Verbatim\})', tex, flags=re.DOTALL)
    result = []
    for part in parts:
        if part.startswith('\\begin{Verbatim}'):
            out = []
            for ch in part:
                out.append(_BOX_ONLY.get(ch, ch))
            result.append(''.join(out))
        else:
            result.append(part)
    return ''.join(result)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Path to input report.md")
    ap.add_argument("--output", required=True, help="Path to output .tex")
    ap.add_argument("--lang", default="auto", choices=("auto", "en", "zh"),
                    help="Language preamble; 'auto' detects Chinese from "
                         "the input filename (_zh in the path).")
    args = ap.parse_args()

    md = Path(args.input).read_text(encoding="utf-8")
    body, title, subtitle = convert(md)

    lang = args.lang
    if lang == "auto":
        lang = "zh" if "_zh" in args.input else "en"
    preamble = PREAMBLE_ZH if lang == "zh" else PREAMBLE_EN

    title_tex = convert_inline(title) if title else "Untitled"
    if subtitle:
        title_tex += r" \\ \large " + convert_inline(subtitle)

    doc = []
    doc.append(preamble)
    doc.append(r"\title{" + title_tex + "}")
    doc.append(r"\author{}")
    doc.append(r"\date{}")
    doc.append(r"\begin{document}")
    doc.append(r"\maketitle")
    doc.append(r"\thispagestyle{empty}")
    doc.append("")
    doc.append(body)
    doc.append("")
    doc.append(r"\end{document}")

    final = "\n".join(doc)

    # For EN (pdflatex): sanitize Unicode and wrap long lines
    if lang == "en":
        final = sanitize_for_pdflatex(final)
        final = wrap_long_lines(final)
    else:
        # For ZH (xelatex): only sanitize Verbatim blocks (box-drawing)
        final = sanitize_verbatim_only(final)
        final = wrap_long_lines(final)

    Path(args.output).write_text(final, encoding="utf-8")
    print(f"Wrote {args.output} (lang={lang})")


if __name__ == "__main__":
    main()
