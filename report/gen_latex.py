import os
from datetime import datetime

latex_content = f"""
\\documentclass{{article}}
\\usepackage{{graphicx}}
\\usepackage{{geometry}}
\\geometry{{a4paper, margin=1in}}

\\title{{Uncertainty-Gated LoRA: Phase 1 Baseline Report}}
\\author{{Final Year Project Team}}
\\date{{{datetime.now().strftime('%B %d, %Y')}}}

\\begin{{document}}

\\maketitle

\\section{{Introduction}}
This report establishes the baseline performance of a standard ResNet-50 model across three domains: Sunny, Rain, and Night. The goal is to demonstrate the need for domain adaptation (LoRA).

\\section{{Methodology}}
We measure model uncertainty using Shannon Entropy. A score above \\textbf{{1.0}} indicates the model is "Confused" (High Uncertainty).

\\section{{Results}}
The figure below visualizes the entropy shift when the model encounters Out-of-Distribution (OOD) data.

\\begin{{figure}}[h!]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{entropy_plot.png}}
    \\caption{{Average Entropy across domains. Note the spike in uncertainty for Rain and Night scenarios.}}
    \\label{{fig:entropy}}
\\end{{figure}}

\\section{{Conclusion}}
The baseline model fails to maintain confidence in adverse weather. This confirms the hypothesis that an Uncertainty-Gated LoRA system is required to stabilize performance.

\\end{{document}}
"""

with open("report/baseline_report.tex", "w") as f:
    f.write(latex_content)

print("✅ LaTeX Report generated at report/baseline_report.tex")
