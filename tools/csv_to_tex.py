# tools/csv_to_tex.py
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
resdir = os.path.join(ROOT, "results")

pairs = [
    ("cs_table_off.csv", "cs_table_off.tex"),
    ("cs_table_on.csv",  "cs_table_on.tex"),
]

preamble = (
    "% Auto-generated from CSV by tools/csv_to_tex.py\n"
    "\\begin{table}[H]\n\\centering\n"
    "\\caption{Comparative statics (toggle state shown in filename).}\n"
    "\\label{tab:cs}\n"
)

postamble = "\\end{table}\n"

for csv_name, tex_name in pairs:
    csv_path = os.path.join(resdir, csv_name)
    df = pd.read_csv(csv_path)

    # Pandas -> LaTeX; you can tweak formatting here if needed
    latex_tab = df.to_latex(index=False, escape=False)  # keep glyphs like ↑ ↓ ∅ Amb.

    tex_path = os.path.join(resdir, tex_name)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(preamble)
        f.write(latex_tab)
        f.write(postamble)

    print(f"Wrote {tex_path}")
