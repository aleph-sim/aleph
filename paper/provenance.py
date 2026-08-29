#!/usr/bin/env python3
"""Generate sections/A-provenance.tex from the `% src:` comments in the section sources.

Every number in the paper carries a `% src: <file> <heading>` comment; this collects, per section,
the distinct source files (and the headings quoted) so a reader can find the originating report.
Run from paper/: python3 provenance.py
"""
import re, glob, collections
order = ["02-decoder", "03-architecture", "04-silicon", "05-scaling", "06-limitations", "07-reproducibility"]
titles = {"02-decoder": "\\S2 The decoder", "03-architecture": "\\S3 Hardware architecture",
          "04-silicon": "\\S4 Results on FPGA silicon", "05-scaling": "\\S5 Scaling and predictive ASIC",
          "06-limitations": "\\S6 Limitations", "07-reproducibility": "\\S7 Reproducibility"}
def esc(s): return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
out = ["\\section{Provenance of every number}", "\\label{app:prov}", "",
       "Each numeric claim in the text carries, in the \\LaTeX{} source, a comment naming the repository",
       "file and heading it was copied from. This table lists those files per section (headings",
       "abbreviated); the reports themselves are in \\texttt{docs/} of the repository \\citep{alephrepo}",
       "at the commit tagged \\texttt{preprint-v1}; the section headings quoted in each comment locate the",
       "exact table or paragraph.", "",
       "\\begin{footnotesize}", "\\begin{tabular}{@{}p{0.16\\textwidth}>{\\raggedright\\arraybackslash}p{0.80\\textwidth}@{}}", "\\toprule",
       "Section & Source files (\\texttt{docs/} unless noted) \\\\", "\\midrule"]
for sec in order:
    files = collections.OrderedDict()
    for line in open(f"sections/{sec}.tex"):
        m = re.search(r"%\s*src:\s*(.*)", line)
        if not m: continue
        for part in re.split(r";\s*", m.group(1)):
            fm = re.match(r"\s*([\w./-]+\.(?:md|csv|sv|sh|yml))\s*(.*)", part)
            if not fm: continue
            f = fm.group(1); f = f[5:] if f.startswith("docs/") else f
            files.setdefault(f, set())
            h = fm.group(2).strip().strip('"').strip("§").strip()
            if h: files[f].add(h[:40])
    cells = []
    for f, hs in files.items():
        hs = sorted(hs)[:3]
        cells.append("\\texttt{" + esc(f) + "}")
    out.append(titles[sec] + " & " + " ".join(c + ";" for c in cells)[:-1] + " \\\\[2pt]")
out += ["\\bottomrule", "\\end{tabular}", "\\end{footnotesize}"]
open("sections/A-provenance.tex", "w").write("\n".join(out) + "\n")
print(f"wrote sections/A-provenance.tex")
