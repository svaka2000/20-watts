#!/usr/bin/env python3
"""Render the 20 Watts papers (Markdown) to polished academic PDFs via weasyprint.
Usage: python src/build_pdf.py   (builds every paper/*.md that exists)
"""
import os, sys, markdown, weasyprint

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "..", "paper")
OUTDIR = os.path.join(HERE, "..", "pdf")
os.makedirs(OUTDIR, exist_ok=True)

CSS = """
@page { size: letter; margin: 1.8cm 1.9cm;
        @bottom-center { content: counter(page); font: 9pt Georgia, serif; color:#555; } }
* { box-sizing: border-box; }
body { font: 10.5pt/1.46 Georgia, 'Times New Roman', serif; color:#111; }
h1 { font-size: 19pt; text-align:center; margin:0 0 2pt; line-height:1.2; }
h1 + p, body > p:first-of-type { text-align:center; }
h2 { font-size: 13.5pt; border-bottom:1px solid #ccc; padding-bottom:2px; margin-top:16pt; }
h3 { font-size: 11.5pt; margin-top:12pt; color:#0b132b; }
p, li { text-align: justify; }
a { color:#1b4965; text-decoration:none; }
code { font: 8.6pt 'SF Mono','Menlo',monospace; background:#f4f6f8; padding:0 2px; }
pre { background:#f4f6f8; border:1px solid #e2e8f0; border-radius:4px; padding:7px 9px;
      font: 8.4pt 'SF Mono','Menlo',monospace; white-space:pre-wrap; overflow:hidden; }
pre code { background:none; padding:0; }
table { border-collapse:collapse; width:100%; margin:8pt 0; font-size:9.4pt; }
th { border-bottom:2px solid #333; text-align:left; padding:3px 6px; }
td { border-bottom:1px solid #ddd; padding:3px 6px; }
img { display:block; margin:8pt auto; max-width:86%; }
blockquote { margin:8pt 0; padding:4pt 12pt; border-left:3px solid #5bc0be;
             color:#333; background:#f7fafb; font-style:italic; }
hr { border:none; border-top:1px solid #ddd; margin:12pt 0; }
strong { color:#0b132b; }
"""

PAPERS = [
    ("PAPER.md", "20Watts_Ep1_SparseFiring.pdf"),
    ("PAPER_EP2.md", "20Watts_Ep2_PredictiveCoding.pdf"),
    ("PAPER_EP3.md", "20Watts_Ep3_FoveatedMemory.pdf"),
    ("THESIS.md", "20Watts_Thesis_Overview.pdf"),
]


def build(md_name, out_name):
    md_path = os.path.join(PAPER, md_name)
    if not os.path.exists(md_path):
        return None
    text = open(md_path).read()
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    out = os.path.join(OUTDIR, out_name)
    weasyprint.HTML(string=html, base_url=PAPER + "/").write_pdf(out)
    sz = os.path.getsize(out) // 1024
    print(f"  wrote {out_name} ({sz} KB)")
    return out


if __name__ == "__main__":
    print("Building PDFs ->", os.path.abspath(OUTDIR))
    for md, out in PAPERS:
        build(md, out)
