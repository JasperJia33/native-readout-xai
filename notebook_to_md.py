"""Convert executed notebook to a standalone Markdown file.

Extracts markdown cells and code-cell outputs (text + images) into a
single .md file with embedded images.

Usage:
    python notebook_to_md.py
    python notebook_to_md.py --input outputs/report/analysis_executed.ipynb --output outputs/markdown/
"""
import argparse
import base64
import json
import os
import re


def extract_text_output(output):
    """Extract plain text from a cell output."""
    if output.get('output_type') == 'stream':
        return ''.join(output.get('text', []))
    if output.get('output_type') in ('execute_result', 'display_data'):
        data = output.get('data', {})
        if 'text/plain' in data:
            return ''.join(data['text/plain'])
    return ''


def extract_image(output, img_dir, img_idx):
    """Save embedded image and return markdown reference."""
    data = output.get('data', {})
    for mime in ('image/png', 'image/jpeg'):
        if mime in data:
            ext = mime.split('/')[1]
            img_name = f"fig_{img_idx:02d}.{ext}"
            img_path = os.path.join(img_dir, img_name)
            raw = data[mime]
            if isinstance(raw, list):
                raw = ''.join(raw)
            with open(img_path, 'wb') as f:
                f.write(base64.b64decode(raw))
            return f"![](images/{img_name})"
    return None


def convert(nb_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    with open(nb_path) as f:
        nb = json.load(f)

    lines = []
    img_idx = 0

    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            src = ''.join(cell['source'])
            lines.append(src)
            lines.append('')

        elif cell['cell_type'] == 'code':
            outputs = cell.get('outputs', [])
            for out in outputs:
                # Images
                img_ref = extract_image(out, img_dir, img_idx)
                if img_ref:
                    img_idx += 1
                    lines.append(img_ref)
                    lines.append('')
                    continue

                data = out.get('data', {})
                has_html_table = ('text/html' in data and
                                  '<table' in ''.join(data['text/html']))

                # HTML table output (prefer over plain text)
                if has_html_table:
                    html = ''.join(data['text/html'])
                    if len(html) < 50000:
                        lines.append(html)
                        lines.append('')
                    continue

                # Plain text output (only if no HTML table)
                text = extract_text_output(out)
                if text.strip():
                    if '<table' in text or '<style' in text:
                        continue
                    lines.append('```')
                    lines.append(text.rstrip())
                    lines.append('```')
                    lines.append('')

    md_path = os.path.join(out_dir, 'report.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Wrote {md_path} ({img_idx} images saved to {img_dir}/)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='outputs/report/analysis_executed.ipynb')
    ap.add_argument('--output', default='outputs/markdown')
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: {args.input} not found. Execute the notebook first:")
        print("  jupyter nbconvert --to notebook --execute analysis.ipynb "
              "--output outputs/report/analysis_executed.ipynb")
        return

    convert(args.input, args.output)


if __name__ == '__main__':
    main()
