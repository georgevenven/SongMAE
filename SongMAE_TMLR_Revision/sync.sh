#!/usr/bin/env bash
# Sync paper.md with the Google Doc "SongMAE TMLR Revision" (rclone remote `gdrive`).
#   ./sync.sh pull   doc -> paper.md (undoes Google's markdown escapes) and commits paper.md if it changed
#   ./sync.sh push   paper.md -> doc; this replaces the doc body (comments lose their anchors), so it refuses
#                    if the doc changed since the last sync
set -euo pipefail
cd "$(dirname "$0")"
DOC=${DOC:-"gdrive,root_folder_id=1JAuxBNgBLSpgFsRYIr82a94izlbUAyZA:SongMAE TMLR Revision.md"}
MD=(--drive-import-formats md --drive-export-formats md)
STAMP=.gdoc_synced
modtime() { rclone lsf --format t "$DOC" "${MD[@]}" 2>/dev/null || true; }

case ${1:-} in
  pull)
    rclone copyto "$DOC" paper.md.tmp "${MD[@]}"
    { sed -E 's/\\([][\\$=_*#<>~`{}.!+|-])/\1/g' paper.md.tmp; echo; } | cat -s > paper.md
    rm paper.md.tmp
    modtime > $STAMP
    git diff --quiet -- paper.md || git commit -q -m "Paper: pull from Google Doc" -- paper.md
    ;;
  push)
    now=$(modtime)
    [[ -z $now || $now == "$(cat $STAMP 2>/dev/null)" ]] || { echo "The doc changed since the last sync; pull first." >&2; exit 1; }
    rclone copyto paper.md "$DOC" "${MD[@]}"
    modtime > $STAMP
    ;;
  *) echo "usage: $0 pull|push" >&2; exit 1 ;;
esac
