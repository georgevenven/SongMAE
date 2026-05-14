import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from individual_id.latent_trajectory_decoder.train_recurrent_sequence import RecurrentSequenceDecoder
from individual_id.latent_trajectory_decoder.train_rope_sequence import RopeSequenceDecoder, load_rows, split_rows


def load_model(path, input_dim, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    metrics = checkpoint.get("metrics", {})
    if metrics.get("model_type") == "recurrent":
        model = RecurrentSequenceDecoder(
            input_dim,
            int(args["hidden_dim"]),
            int(args["layers"]),
            str(args["cell"]),
        ).to(device)
    else:
        model = RopeSequenceDecoder(input_dim, int(args["d_model"]), int(args["heads"]), int(args["layers"])).to(device)
    model.load_state_dict(checkpoint["decoder"])
    model.eval()
    normalize_output = metrics.get("output_for_loss") == "l2_normalized_decoder_head"
    return model, normalize_output


@torch.no_grad()
def maybe_normalize(x, normalize_output):
    if not normalize_output:
        return x
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def teacher_forced(model, seq, device, normalize_output):
    x = torch.from_numpy(seq[:-1]).to(device=device, dtype=torch.float32)[None]
    valid = torch.ones((1, x.shape[1]), device=device, dtype=torch.bool)
    pred, _ = model(x, valid)
    pred = pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return maybe_normalize(pred, normalize_output).astype(np.float32), seq[1:].astype(np.float32)


@torch.no_grad()
def rollout(model, seq, prefix, max_steps, device, normalize_output):
    total = min(seq.shape[0], max_steps)
    points = torch.from_numpy(seq[:prefix]).to(device=device, dtype=torch.float32)
    while points.shape[0] < total:
        valid = torch.ones((1, points.shape[0]), device=device, dtype=torch.bool)
        pred, _ = model(points[None], valid)
        next_point = pred[0, -1:]
        if normalize_output:
            next_point = torch.nn.functional.normalize(next_point, dim=-1)
        points = torch.cat([points, next_point], dim=0)
    return points.cpu().numpy().astype(np.float32), seq[:total].astype(np.float32)


def choose_rows(rows, count, seed):
    rng = np.random.default_rng(seed)
    by_bird = {}
    for row in rows:
        by_bird.setdefault(row[1], []).append(row)
    birds = sorted(by_bird)
    keep = []
    for bird in birds:
        if len(keep) == count:
            break
        choices = [row for row in by_bird[bird] if row[2].shape[0] >= 40]
        if choices:
            keep.append(choices[int(rng.integers(len(choices)))])
    assert keep
    return keep


def fit_projection(rows, model, prefix, max_steps, device, normalize_output):
    points = []
    payload = []
    for recording, bird, seq in rows:
        pred, true = teacher_forced(model, seq[:max_steps], device, normalize_output)
        roll_pred, roll_true = rollout(model, seq, prefix, max_steps, device, normalize_output)
        payload.append((recording, bird, pred, true, roll_pred, roll_true))
        points.extend([pred, true, roll_pred, roll_true])
    pca = PCA(n_components=2, random_state=0)
    pca.fit(np.vstack(points))
    return pca, payload


def mse_by_time(pred, true):
    return ((pred - true) ** 2).mean(axis=1)


def plot_grid(payload, pca, path):
    rows = len(payload)
    fig, axes = plt.subplots(rows, 3, figsize=(13, 3.2 * rows), squeeze=False)
    for i, (recording, bird, pred, true, roll_pred, roll_true) in enumerate(payload):
        pred2 = pca.transform(pred)
        true2 = pca.transform(true)
        roll_pred2 = pca.transform(roll_pred)
        roll_true2 = pca.transform(roll_true)

        ax = axes[i, 0]
        ax.plot(true2[:, 0], true2[:, 1], color="#2c63b7", linewidth=1.5, alpha=0.9, label="true")
        ax.plot(pred2[:, 0], pred2[:, 1], color="#d64b3c", linewidth=1.2, alpha=0.8, label="one-step pred")
        ax.set_title(f"{bird} {recording}\none-step")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[i, 1]
        ax.plot(roll_true2[:, 0], roll_true2[:, 1], color="#2c63b7", linewidth=1.5, alpha=0.9)
        ax.plot(roll_pred2[:, 0], roll_pred2[:, 1], color="#d64b3c", linewidth=1.2, alpha=0.8)
        ax.scatter(roll_true2[:20, 0], roll_true2[:20, 1], s=8, color="#15366f")
        ax.set_title("free rollout")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[i, 2]
        ax.plot(mse_by_time(pred, true), color="#d64b3c", linewidth=1.3, label="one-step")
        shared = min(roll_pred.shape[0], roll_true.shape[0])
        ax.plot(mse_by_time(roll_pred[:shared], roll_true[:shared]), color="#6a3d9a", linewidth=1.3, label="rollout")
        ax.set_title("latent MSE by step")
        ax.set_xlabel("step")
        ax.set_ylabel("MSE")
        ax.legend(frameon=False, fontsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def pixels(points, lo, hi, size, pad):
    xy = (points - lo) / np.maximum(hi - lo, 1e-6)
    out = np.empty_like(xy)
    out[:, 0] = pad + xy[:, 0] * (size - 2 * pad)
    out[:, 1] = size - pad - xy[:, 1] * (size - 2 * pad)
    return [tuple(row) for row in out]


def save_gif(item, pca, path, size=720, pad=58):
    recording, bird, pred, true, roll_pred, roll_true = item
    true2 = pca.transform(true)
    pred2 = pca.transform(pred)
    roll_true2 = pca.transform(roll_true)
    roll_pred2 = pca.transform(roll_pred)
    all_xy = np.vstack([true2, pred2, roll_true2, roll_pred2])
    lo = all_xy.min(axis=0)
    hi = all_xy.max(axis=0)
    true_px = pixels(true2, lo, hi, size, pad)
    pred_px = pixels(pred2, lo, hi, size, pad)
    roll_true_px = pixels(roll_true2, lo, hi, size, pad)
    roll_pred_px = pixels(roll_pred2, lo, hi, size, pad)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    frames = []
    total = min(len(true_px), len(roll_pred_px))
    for end in range(2, total + 1):
        frame = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(frame)
        draw.rectangle((pad, pad, size - pad, size - pad), outline=(210, 210, 210))
        draw.text((pad, 18), f"{bird} {recording}", fill=(20, 20, 20), font=font)
        draw.text((pad, size - 34), "blue=true, red=one-step, purple=rollout", fill=(35, 35, 35), font=font)
        draw.line(true_px[:end], fill=(44, 99, 183), width=4)
        draw.line(pred_px[:end], fill=(214, 75, 60), width=3)
        draw.line(roll_pred_px[:end], fill=(106, 61, 154), width=3)
        draw.line(roll_true_px[:end], fill=(44, 99, 183), width=2)
        frames.append(frame)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=90, loop=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot RoPE decoder predictions against true latent trajectories.")
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--model_pt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--prefix", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=180)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, input_dim = load_rows(args.features_npz)
    _, val_rows = split_rows(rows, args.val_fraction, args.seed)
    selected = choose_rows(val_rows, args.count, args.seed)
    model, normalize_output = load_model(args.model_pt, input_dim, device)
    pca, payload = fit_projection(selected, model, args.prefix, args.max_steps, device, normalize_output)
    grid_path = out_dir / "prediction_vs_true_trajectories.png"
    gif_path = out_dir / "prediction_vs_true_trajectory.gif"
    plot_grid(payload, pca, grid_path)
    save_gif(payload[0], pca, gif_path)

    rows_out = []
    for recording, bird, pred, true, roll_pred, roll_true in payload:
        shared = min(roll_pred.shape[0], roll_true.shape[0])
        rows_out.append(
            {
                "recording": recording,
                "bird": bird,
                "steps": int(true.shape[0]),
                "one_step_mse": float(mse_by_time(pred, true).mean()),
                "rollout_mse": float(mse_by_time(roll_pred[:shared], roll_true[:shared]).mean()),
            }
        )
    summary = {
        "features_npz": args.features_npz,
        "model_pt": args.model_pt,
        "device": str(device),
        "normalize_output": bool(normalize_output),
        "projection": "PCA2 fit on selected true and predicted latent points",
        "grid_png": str(grid_path),
        "gif": str(gif_path),
        "rows": rows_out,
    }
    (out_dir / "prediction_vs_true_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
