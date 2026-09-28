#!/usr/bin/env bash
# Build paper.pdf with the TMLR compiler image (see SongMAE_TMLR_Paper_Materials/TMLR_compiler/Dockerfile).
set -euo pipefail
cd "$(dirname "$0")"
docker run --rm --volume "$PWD:/data" --workdir /data --user "$(id -u):$(id -g)" --entrypoint sh \
  tinybird-tmlr-compiler:2026 -c 'pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null && bibtex paper >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null'
rm -f paper.aux paper.bbl paper.blg paper.log paper.out
