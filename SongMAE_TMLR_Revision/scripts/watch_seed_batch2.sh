#!/usr/bin/env bash
# Work desktop: wait for the remaining Table 2 seed runs (Twins) and the frequency-masking runs (here) to finish, copy them to
# the Mac, then run their kNN layer sweep there after seed batch 1.
set -euo pipefail
cd "$(dirname "$0")/../.."
TWINS_HOST=george-vengrovski@163.41.128.66
twins_runs=$(for c in p128x1_c010 p16x1_c010 p4x4_c010 p32x4_c0025 p32x4_c005 p32x4_c010; do for s in 1 2; do printf 'xcl_micro_100k_%s_seed%s ' $c $s; done; done)
local_runs="xcl_micro_100k_p32x1_frequency_seed0 xcl_micro_100k_p32x1_frequency_seed1 xcl_micro_100k_p32x1_frequency_seed2"
final=weights/model_step_099999.pth

# A run counts as finished on either machine (some Table 2 seeds train on the desktop); sync_runs_to_mac.sh copies
# from wherever it is.
twins_or_local() { for r in $twins_runs; do [[ -f runs/$r/$final ]] || ssh -o BatchMode=yes $TWINS_HOST "[ -f Documents/SongMAE-reviews/runs/$r/$final ]" || return 1; done; }
local_done() { for r in $local_runs; do [[ -f runs/$r/$final ]] || return 1; done; }
until twins_or_local && local_done; do sleep 600; done
SongMAE_TMLR_Revision/scripts/sync_runs_to_mac.sh $twins_runs $local_runs
ssh -o BatchMode=yes macstudio "cd Documents/SongMAE-reviews && git pull -q && nohup SongMAE_TMLR_Revision/scripts/mac_seed_knn.sh logs/knn_layer_sweep_seeds_batch1.log $twins_runs $local_runs > logs/knn_layer_sweep_seeds_batch2.log 2>&1 < /dev/null &"
echo "batch 2 synced and queued on the Mac"
