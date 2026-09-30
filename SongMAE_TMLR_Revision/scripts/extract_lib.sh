# Shared by probe_all.sh, kmeans_all.sh and knn_seeds.sh: one-layer extraction for any revision condition.
# Callers set PYTHON_BIN, WAV_ROOT and SELECTION (data-selection args, e.g. --num_timebins ... --balanced_events ...).
selected() { [[ -z "$2" || " $2 " == *" $1 "* ]]; }
speed_of() { local s=${1##*_speed}; [[ $1 == *_speed* ]] || s=1; s=${s/p/.}; [[ $s == .* ]] && s=0$s; echo "$s"; }

extract() {  # condition layer specs annotations bird out (uses SELECTION)
  local c=$1 layer=$2 specs=$3 ann=$4 bird=$5 out=$6 speed; speed=$(speed_of "$1")
  case $c in
    xcl_*|Xcl_*)
      "$PYTHON_BIN" -m src.core.extract_embedding --spec_dir "$specs" --run_dir "runs/$c" \
        --checkpoint "$(ls "runs/$c/weights" | sort -V | tail -1)" --out_dir "$out" --json_path "$ann" --bird "$bird" \
        --minimal --target_feature_type end_of_block --encoder_layer_idx "$layer" "${SELECTION[@]}" ;;
    birdaves_*)
      "$PYTHON_BIN" src/external_models/aves.py --speed "$speed" --chunk_timebins "$("$PYTHON_BIN" -c "print(int(1000 * $speed))")" \
        --aves_model_path files/birdaves-biox-base.torchaudio.pt --aves_config_path files/birdaves-biox-base.torchaudio.model_config.json \
        --spec_dir "$specs" --wav_dir "$WAV_ROOT" --annotation_file "$ann" --bird "$bird" --out_dir "$out" \
        --encoder_layer_idx "$layer" "${SELECTION[@]}" ;;
    hubert_*)
      "$PYTHON_BIN" src/external_models/hubert.py --model_name facebook/hubert-base-ls960 --audio_sr 16000 --chunk_timebins 1000 \
        --spec_dir "$specs" --wav_dir "$WAV_ROOT" --annotation_file "$ann" --bird "$bird" --out_dir "$out" \
        --encoder_layer_idx "$layer" "${SELECTION[@]}" ;;
    beats_*|birdmae_*)
      "$PYTHON_BIN" src/external_models/review_baselines.py --model "${c%%_*}" --speed "$speed" --chunk_timebins 1000 \
        --spec_dir "$specs" --wav_dir "$WAV_ROOT" --annotation_file "$ann" --bird "$bird" --out_dir "$out" \
        --encoder_layer_idx "$layer" "${SELECTION[@]}" ;;
    *) echo "unknown condition $c" >&2; return 1 ;;
  esac
}

birds_of() { "$PYTHON_BIN" -c "import json,sys;print(' '.join(sorted({r['recording']['bird_id'] for r in json.load(open(sys.argv[1]))['recordings']})))" "$1"; }
layers_of() { "$PYTHON_BIN" -c "import json,sys;print(*json.load(open(sys.argv[1]))[sys.argv[2]]['union'])" "$LAYERS_JSON" "$1"; }
