"""Frozen BEATs/BirdMAE extraction with timestamps on the original audio clock."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.core.data_loader import balanced_event_indices
from src.external_models.aves import load_audio
from src.external_models.bird_mae import load_model as load_birdmae
from src.external_models.data_loader import (
    WavFromSpectrogramDataset, append_limited, chunked_items,
    convolution_feature_map, save_concatenated_embeddings,
)


def load_encoder(name, assets_dir):
    if name == "birdmae":
        extractor, model, frequency, _ = load_birdmae(str(assets_dir / "birdmae_base"))
        assert frequency == 8 and model.config.patch_size == 16
        assert extractor.sampling_rate == 32000 and extractor.frame_shift == 10
        return model, extractor
    assert name == "beats"
    sys.path.insert(0, str(assets_dir / "beats"))
    from BEATs import BEATs, BEATsConfig

    checkpoint = torch.load(
        assets_dir / "beats/BEATs_iter3_plus_AS2M.pt", map_location="cpu", weights_only=True,
    )
    config = BEATsConfig(checkpoint["cfg"])
    assert not config.finetuned_model and config.input_patch_size == 16
    model = BEATs(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval(), None


def extract_features(name, model, extractor, wav, layer_index, all_layers):
    blocks = model.blocks if name == "birdmae" else model.encoder.layers
    selected = range(len(blocks)) if all_layers else [layer_index]
    assert all(0 <= index < len(blocks) for index in selected)
    features = []

    def capture(_module, _inputs, output):
        # Both models flatten time then frequency; eight frequency patches share a time.
        x = output[0][0, 1:] if name == "birdmae" else output[0][:, 0]
        features.append(x.detach().reshape(-1, 8 * x.shape[-1]).cpu())

    handles = [blocks[index].register_forward_hook(capture) for index in selected]
    device = next(model.parameters()).device
    try:
        with torch.inference_mode():
            if name == "birdmae":
                assert 0 < wav.numel() <= 160000
                model(extractor(wav.numpy()).to(device))
            else:
                # Sixteen 10-ms fbank frames require 175 ms of waveform.
                wav = F.pad(wav, (0, max(0, 2800 - wav.numel())))
                model.extract_features(wav.unsqueeze(0).to(device))
    finally:
        for handle in handles:
            handle.remove()
    assert len(features) == len(selected)
    x = torch.stack(features, dim=1) if all_layers else features[0]
    return x.numpy().astype(np.float32, copy=False)


def align_features(labels, features, timebin_ms, speed):
    # Kaldi: 25-ms windows, 10-ms hop, 16 frames per patch. Include the
    # 7.5-ms center offset; never stretch padded tokens across a short event.
    geometry = (87.5 * speed / timebin_ms, 160.0 * speed / timebin_ms)
    centers = geometry[0] + np.arange(len(features)) * geometry[1]
    boundaries = (centers[:-1] + centers[1:]) / 2
    count = int(np.searchsorted(boundaries, len(labels), side="left")) + 1
    labels, edges = convolution_feature_map(labels, count, geometry)
    return features[:count], labels, edges


def save_embeddings(args):
    assert args.model == "birdmae" or args.speed == 1.0
    assert not args.out_dir.exists(), f"output already exists: {args.out_dir}"
    dataset = WavFromSpectrogramDataset(
        args.spec_dir, args.wav_dir, args.annotation_file,
        recording_mode=args.recording_mode, recording_stem=args.recording_stem,
        selected_bird=args.bird,
    )
    audio_sr = 32000 if args.model == "birdmae" else 16000
    timebin_ms = 1000 * dataset.audio_params[2] / dataset.audio_params[0]
    chunk_timebins = min(args.chunk_timebins, int(5000 * args.speed / timebin_ms))
    assert chunk_timebins > 0
    model, extractor = load_encoder(args.model, args.assets_dir)
    model = model.to(args.device).eval()
    indices = balanced_event_indices(dataset.spec_dataset, args.balanced_events, args.event_seed)
    rows, cache = [], {}
    used = 0
    for item in chunked_items(dataset, args.num_timebins, chunk_timebins, indices):
        wav = load_audio(item, audio_sr, cache)
        assert wav.numel() > 0, item["wav_path"]
        samples = round((item["end_ms"] - item["start_ms"]) * audio_sr / 1000)
        wav = F.pad(wav, (0, samples - wav.numel()))
        if args.speed != 1.0:
            # Playback slowdown lowers pitch too; this is not pitch-preserving stretching.
            wav = torchaudio.functional.resample(wav, int(audio_sr * args.speed), audio_sr)
        features = extract_features(
            args.model, model, extractor, wav, args.encoder_layer_idx, args.all_layers,
        )
        features, labels, edges = align_features(item["labels"], features, timebin_ms, args.speed)
        row = {
            "item": item, "encoded_embeddings": features,
            "labels_downsampled": labels, "token_edges": edges,
        }
        used, keep_going = append_limited(rows, row, args.max_points, used)
        if not keep_going:
            break
    save_concatenated_embeddings(
        args.out_dir, rows, model_name=args.model, audio_sr=audio_sr,
        playback_speed=args.speed, effective_stride_ms=160 * args.speed,
        feature_center_timebins=87.5 * args.speed / timebin_ms,
        feature_stride_timebins=160 * args.speed / timebin_ms,
        timestamp_clock="original_recording", frequency_reduction="concatenate",
        encoder_layer_idx=None if args.all_layers else args.encoder_layer_idx,
        all_layers=args.all_layers, target_feature_type="end_of_block",
        chunk_timebins=chunk_timebins, chunk_original_ms=chunk_timebins * timebin_ms,
        balanced_events=args.balanced_events, event_seed=args.event_seed,
        assets_manifest=str(args.assets_dir / "manifest.json"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["beats", "birdmae"], required=True)
    parser.add_argument("--assets_dir", type=Path, default=ROOT / "files/review_baselines")
    parser.add_argument("--spec_dir", required=True)
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--annotation_file", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--bird")
    parser.add_argument("--recording_stem")
    parser.add_argument("--recording_mode", choices=["events", "background", "full_recordings"], default="events")
    parser.add_argument("--speed", type=float, choices=[1.0, 0.5, 0.25, 0.125], default=1.0)
    layers = parser.add_mutually_exclusive_group()
    layers.add_argument("--encoder_layer_idx", type=int, default=11)
    layers.add_argument("--all_layers", action="store_true")
    parser.add_argument("--chunk_timebins", type=int, default=1000)
    parser.add_argument("--num_timebins", type=int, default=0)
    parser.add_argument("--max_points", type=int, default=0)
    parser.add_argument("--balanced_events", type=int, default=0)
    parser.add_argument("--event_seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    save_embeddings(parse_args())
