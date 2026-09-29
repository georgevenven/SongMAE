#!/usr/bin/env bash
# paper.md + latex/blocks/*.tex -> latex/paper.tex -> latex/paper.pdf.
# "Figure 4", "Supplemental Table 2", "Appendix A.5" etc. are linked automatically; "REVISION: ..." paragraphs become comments.
set -euo pipefail
cd "$(dirname "$0")"
perl -pe '
  my %label = ("Figure" => "fig:", "Table" => "tab:", "Supplemental Figure" => "fig:s", "Supplemental Table" => "tab:s",
               "Appendix Table" => "tab:s", "Appendix A." => "app:a");
  my $caption = s/^((?:Supplemental )?(?:Figure|Table) \d+:)// ? $1 : "";
  s{\b(Supplemental Figure|Supplemental Table|Appendix Table|Figure|Table|Appendix A\.)( ?)(\d+)}{`\\hyperref[$label{$1}$3]{$1$2$3}`{=latex}}g;
  $_ = $caption . $_;
  s/^REVISION: (.*)/```{=latex}\n% REVISION: $1\n```/;
' paper.md | docker run --rm -i --volume "$PWD:/data" --workdir /data --user "$(id -u):$(id -g)" --entrypoint pandoc \
  tinybird-tmlr-compiler:2026 -f markdown --natbib --template latex/template.tex --lua-filter latex/paper.lua -o latex/paper.tex
latex/compile.sh
