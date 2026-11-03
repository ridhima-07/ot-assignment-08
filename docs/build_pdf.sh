#!/bin/bash
# Render REPORT.md + figures + docs/RESULTS_SUMMARY.md into REPORT.pdf (needs pandoc and wkhtmltopdf).
set -e
cd "$(dirname "$0")/.."
cat > /tmp/a8_style.css <<'EOF'
body { font-family: "DejaVu Sans", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; max-width: 980px; margin: auto; color: #1d1d1f; }
h1 { font-size: 20pt; border-bottom: 2px solid #234; padding-bottom: 4px; } h2 { font-size: 14pt; color: #234; margin-top: 22px; }
h3 { font-size: 11.5pt; } table { border-collapse: collapse; margin: 8px 0; font-size: 9pt; }
th, td { border: 1px solid #bbb; padding: 3px 6px; vertical-align: top; } th { background: #eef2f6; }
code { font-size: 9pt; background: #f4f4f4; padding: 0 2px; } img { max-width: 100%; margin: 6px 0; }
blockquote { border-left: 3px solid #8aa; margin-left: 0; padding-left: 10px; color: #444; }
EOF
{
  cat REPORT.md
  echo -e "\n\n## Appendix A — Key figures\n"
  for f in trackE/E_rosters trackE/E_quality_runtime trackA/A1_convergence trackA/A2_saddle trackC/C4_vqe trackC/C5_barren \
           trackD/D_sched_sim trackD/D_fedavg trackF/F_speedup trackG/G_sched trackG/G_cow trackH/H_index_selection \
           trackI/I3_grover trackJ/J_minimization; do
    [ -f "results/$f.png" ] && echo -e "**${f#*/}**\n\n![](results/$f.png)\n"
  done
  echo -e "\n\n# Appendix B — Results summary\n"
  tail -n +2 docs/RESULTS_SUMMARY.md
} > /tmp/a8_report.md
pandoc /tmp/a8_report.md -f gfm -t html5 -s --metadata pagetitle="Optimize Anything - Assignment 8 - Ridhima Trivedi (241071062)" \
       --css /tmp/a8_style.css --resource-path=. -o /tmp/a8_report.html
cp /tmp/a8_report.html ./.report_tmp.html
wkhtmltopdf --quiet --enable-local-file-access -s A4 --margin-top 14mm --margin-bottom 14mm --margin-left 12mm --margin-right 12mm \
            --footer-center "[page] / [topage]" --footer-font-size 8 ./.report_tmp.html REPORT.pdf || true
rm -f ./.report_tmp.html
ls -la REPORT.pdf
