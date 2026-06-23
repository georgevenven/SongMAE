import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == HERE:
    sys.path.pop(0)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib import cm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.embedding_store import EmbeddingStore

RAW_MODELS = {"aves", "bird_mae", "hubert"}


def split_models(models):
    out = [model.strip() for model in models.split(",") if model.strip()]
    assert out
    for model in out:
        assert model in {"songmae", "songmae_random", *RAW_MODELS}, f"unknown model: {model}"
    return out


def add_arg(cmd, flag, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def run(cmd):
    print(" ".join(map(str, cmd)))
    subprocess.run([str(part) for part in cmd], check=True)


def songmae_command(model, args, out_path):
    assert args.songmae_run_dir, "--songmae_run_dir is required for model=songmae"
    cmd = [
        sys.executable,
        "-m",
        "src.core.extract_embedding",
        "--spec_dir",
        args.spec_dir,
        "--run_dir",
        args.songmae_run_dir,
        "--out_dir",
        out_path,
        "--json_path",
        args.annotation_file,
        "--num_timebins",
        args.num_timebins,
        "--recording_mode",
        args.recording_mode,
    ]
    add_arg(cmd, "--checkpoint", args.checkpoint)
    add_arg(cmd, "--bird", args.bird)
    add_arg(cmd, "--recording_stem", args.recording_stem)
    add_arg(cmd, "--encoder_layer_idx", args.encoder_layer_idx)
    if model == "songmae_random":
        cmd.append("--random_init")
    return cmd


def raw_command(model, args, out_dir):
    assert args.wav_dir, f"--wav_dir is required for model={model}"
    cmd = [
        sys.executable,
        ROOT / "src" / "external_models" / f"{model}.py",
        "--spec_dir",
        args.spec_dir,
        "--wav_dir",
        args.wav_dir,
        "--annotation_file",
        args.annotation_file,
        "--out_dir",
        out_dir,
        "--recording_mode",
        args.recording_mode,
        "--wav_exts",
        args.wav_exts,
    ]
    add_arg(cmd, "--bird", args.bird)
    add_arg(cmd, "--recording_stem", args.recording_stem)
    if model == "aves":
        cmd.extend(["--aves_model_path", args.aves_model_path, "--aves_config_path", args.aves_config_path])
        add_arg(cmd, "--encoder_layer_idx", args.encoder_layer_idx)
    if model == "bird_mae":
        cmd.extend(["--model_name", args.bird_mae_model_name])
    if model == "hubert":
        cmd.extend(["--model_name", args.hubert_model_name])
        add_arg(cmd, "--encoder_layer_idx", args.encoder_layer_idx)
    return cmd


def extract(model, args, model_dir):
    if model in {"songmae", "songmae_random"}:
        out_path = model_dir / "embeddings"
        if not (args.reuse and out_path.is_dir()):
            model_dir.mkdir(parents=True, exist_ok=True)
            run(songmae_command(model, args, out_path))
        return out_path

    out_path = model_dir / "embeddings"
    if not args.reuse:
        shutil.rmtree(out_path, ignore_errors=True)
    if not (args.reuse and out_path.is_dir()):
        model_dir.mkdir(parents=True, exist_ok=True)
        run(raw_command(model, args, out_path))
    return out_path


def load_arrays(path):
    store = EmbeddingStore(path)
    features = store["encoded_embeddings"].astype(np.float32, copy=False)
    labels = store["labels_downsampled"].astype(np.int64, copy=False)
    assert features.shape[0] == labels.shape[0], path
    return features, labels


def limit_points(features, labels, max_points):
    if max_points <= 0 or features.shape[0] <= max_points:
        return features, labels
    return features[:max_points], labels[:max_points]


def zscore(features):
    mean = features.mean(axis=0, keepdims=True)
    std = np.maximum(features.std(axis=0, keepdims=True), 1e-8)
    return ((features - mean) / std).astype(np.float32, copy=False)


def fit_umap(features, args):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.seed if args.deterministic else None,
        low_memory=True,
        n_jobs=-1,
    )
    return reducer.fit_transform(features).astype(np.float32, copy=False)


def build_palette(labels):
    labels = np.unique(labels[labels >= 0])
    colors = cm.tab20(np.linspace(0, 1, max(1, labels.size)))
    return {int(label): np.asarray(color[:3], dtype=np.float32) for label, color in zip(labels, colors)}


def scatter(xy, labels, out_base):
    fig = plt.figure(figsize=(5.5, 5.5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    silence = labels < 0
    if silence.any():
        ax.scatter(xy[silence, 0], xy[silence, 1], s=10, color="#404040", alpha=0.1, edgecolors="none")
    for label, color in build_palette(labels).items():
        idx = labels == label
        ax.scatter(xy[idx, 0], xy[idx, 1], s=10, color=color, alpha=0.15, edgecolors="none")
    ax.set_xlabel("UMAP 1", fontsize=20, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=20, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def run_model(model, args):
    model_dir = Path(args.out_dir) / model
    source = extract(model, args, model_dir)
    features, labels = load_arrays(source)
    features, labels = limit_points(features, labels, args.max_points)
    if args.zscore:
        features = zscore(features)
    xy = fit_umap(features, args)
    np.save(model_dir / "umap_points.npy", xy)
    np.save(model_dir / "labels.npy", labels)
    scatter(xy, labels, model_dir / "umap")
    return {"model": model, "points": int(xy.shape[0]), "dim": int(features.shape[1])}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract embeddings and make one-bird syllable UMAPs.")
    parser.add_argument("--spec_dir", required=True)
    parser.add_argument("--annotation_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--models", default="songmae")
    parser.add_argument("--wav_dir")
    parser.add_argument("--recording_mode", default="events", choices=["events", "full_recordings"])
    parser.add_argument("--recording_stem")
    parser.add_argument("--bird", required=True)
    parser.add_argument("--wav_exts", default=".wav,.flac,.ogg,.mp3")
    parser.add_argument("--max_points", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--zscore", dest="zscore", action="store_true", default=True)
    parser.add_argument("--no_zscore", dest="zscore", action="store_false")
    parser.add_argument("--umap_neighbors", type=int, default=25)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_metric", default="euclidean")
    parser.add_argument("--songmae_run_dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--num_timebins", type=int, default=12400)
    parser.add_argument("--encoder_layer_idx", type=int)
    parser.add_argument("--aves_model_path", default=str(ROOT / "files" / "aves-base-bio.torchaudio.pt"))
    parser.add_argument("--aves_config_path", default=str(ROOT / "files" / "aves-base-bio.torchaudio.model_config.json"))
    parser.add_argument("--bird_mae_model_name", default="DBD-research-group/Bird-MAE-Base")
    parser.add_argument("--hubert_model_name", default="facebook/hubert-large-ll60k")
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    summary = [run_model(model, args) for model in split_models(args.models)]
    (Path(args.out_dir) / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
