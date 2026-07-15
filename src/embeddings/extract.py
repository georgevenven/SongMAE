import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def add_arg(command, flag, value):
    if value is not None:
        command.extend([flag, str(value)])


def songmae_command(model, args, out_dir):
    assert args.songmae_run_dir, "--songmae_run_dir is required for SongMAE"
    command = [
        sys.executable,
        "-m",
        "src.core.extract_embedding",
        "--spec_dir", args.spec_dir,
        "--run_dir", args.songmae_run_dir,
        "--out_dir", out_dir,
        "--json_path", args.annotation_file,
        "--num_timebins", args.num_timebins,
        "--recording_mode", args.recording_mode,
    ]
    add_arg(command, "--checkpoint", args.checkpoint)
    add_arg(command, "--bird", args.bird)
    add_arg(command, "--recording_stem", args.recording_stem)
    add_arg(command, "--encoder_layer_idx", args.encoder_layer_idx)
    add_arg(command, "--target_feature_type", args.target_feature_type)
    if args.minimal:
        command.append("--minimal")
    if model == "songmae_random":
        command.append("--random_init")
    return command


def raw_command(model, args, out_dir):
    assert args.wav_dir, f"--wav_dir is required for model={model}"
    command = [
        sys.executable,
        ROOT / "src" / "external_models" / f"{model}.py",
        "--spec_dir", args.spec_dir,
        "--wav_dir", args.wav_dir,
        "--annotation_file", args.annotation_file,
        "--out_dir", out_dir,
        "--recording_mode", args.recording_mode,
        "--max_points", args.max_points,
        "--num_timebins", args.num_timebins,
        "--wav_exts", args.wav_exts,
    ]
    add_arg(command, "--bird", args.bird)
    add_arg(command, "--recording_stem", args.recording_stem)
    if model == "aves":
        command.extend(["--aves_model_path", args.aves_model_path, "--aves_config_path", args.aves_config_path])
        add_arg(command, "--encoder_layer_idx", args.encoder_layer_idx)
        add_arg(command, "--chunk_timebins", getattr(args, "chunk_timebins", None))
    if model == "bird_mae":
        command.extend(["--model_name", args.bird_mae_model_name])
    if model == "hubert":
        command.extend(["--model_name", args.hubert_model_name])
        add_arg(command, "--encoder_layer_idx", args.encoder_layer_idx)
        add_arg(command, "--chunk_timebins", getattr(args, "chunk_timebins", None))
    return command


def extract(model, args, model_dir):
    out_dir = Path(model_dir) / "embeddings"
    if args.reuse and out_dir.is_dir():
        return out_dir
    if model not in {"songmae", "songmae_random"}:
        shutil.rmtree(out_dir, ignore_errors=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    command = songmae_command(model, args, out_dir) if model.startswith("songmae") else raw_command(model, args, out_dir)
    print(" ".join(map(str, command)))
    subprocess.run([str(part) for part in command], check=True)
    return out_dir
