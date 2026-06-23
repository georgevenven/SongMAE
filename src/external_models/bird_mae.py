import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.external_models.data_loader import WavFromSpectrogramDataset, labels_for_features, limited_items, save_concatenated_embeddings


def load_model(model_name):
    from transformers import AutoFeatureExtractor, AutoModel
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.global_pool = None
    model.eval()
    freq_patches, time_patches = model.patch_embed.patch_hw
    return feature_extractor, model, int(freq_patches), int(time_patches)


def load_audio(item, audio_sr):
    wav, sr = torchaudio.load(str(item["wav_path"]))
    if wav.ndim == 2:
        wav = wav[0]
    if sr != audio_sr:
        wav = torchaudio.functional.resample(wav, sr, audio_sr)
    start = int(round(item["start_ms"] / 1000.0 * audio_sr))
    end = int(round(item["end_ms"] / 1000.0 * audio_sr))
    end = max(start, min(end, int(wav.shape[0])))
    return wav[start:end].to(torch.float32).contiguous()


def extract_features(feature_extractor, model, wav, freq_patches, time_patches, device):
    wav = wav.detach().cpu().numpy().astype(np.float32, copy=False)
    features = feature_extractor(wav)
    assert torch.is_tensor(features), f"unexpected Bird-MAE features: {type(features)}"
    with torch.no_grad():
        outputs = model(features.to(device))
    grid = outputs.last_hidden_state[:, 1:, :].detach().cpu().numpy().astype(np.float32, copy=False)[0]
    assert grid.shape[0] == freq_patches * time_patches
    grid = grid.reshape(time_patches, freq_patches, grid.shape[1])
    return grid.reshape(time_patches, -1), grid


def save_embeddings(args):
    dataset = WavFromSpectrogramDataset(
        args.spec_dir,
        args.wav_dir,
        args.annotation_file,
        recording_mode=args.recording_mode,
        recording_stem=args.recording_stem,
        selected_bird=args.bird,
        wav_exts=args.wav_exts,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor, model, freq_patches, time_patches = load_model(args.model_name)
    model = model.to(device)

    rows = []
    for item in limited_items(dataset, args.num_timebins):
        embeddings, grid = extract_features(
            feature_extractor,
            model,
            load_audio(item, args.audio_sr),
            freq_patches,
            time_patches,
            device,
        )
        if embeddings.shape[0] == 0:
            continue
        labels = labels_for_features(item["labels"], embeddings.shape[0])
        assert embeddings.shape[0] == labels.shape[0]
        rows.append(
            {
                "item": item,
                "encoded_embeddings": embeddings,
                "encoded_embeddings_grid": grid,
                "labels_downsampled": labels,
            }
        )
    save_concatenated_embeddings(
        args.out_dir,
        rows,
        model_name=args.model_name,
        audio_sr=args.audio_sr,
        num_patches_height=freq_patches,
        num_patches_time=time_patches,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Bird-MAE embeddings into an embedding folder.")
    parser.add_argument("--spec_dir", required=True)
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--annotation_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", default="DBD-research-group/Bird-MAE-Base")
    parser.add_argument("--audio_sr", type=int, default=32000)
    parser.add_argument("--recording_mode", default="events", choices=["events", "full_recordings"])
    parser.add_argument("--recording_stem")
    parser.add_argument("--bird")
    parser.add_argument("--wav_exts", default=".wav,.flac,.ogg,.mp3")
    parser.add_argument("--num_timebins", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    save_embeddings(parse_args())
