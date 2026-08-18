import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.models import wav2vec2_model

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.external_models.data_loader import (
    append_limited,
    chunked_items,
    convolution_feature_map,
    convolution_geometry,
    save_concatenated_embeddings,
    WavFromSpectrogramDataset,
)
from src.core.data_loader import balanced_event_indices


def load_model(config_path, model_path):
    config = json.loads(Path(config_path).read_text())
    model = wav2vec2_model(**config, aux_num_out=None)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def min_input_samples(model):
    length = 1
    for layer in reversed(model.feature_extractor.conv_layers):
        conv = getattr(layer, "conv", layer)
        length = (length - 1) * int(conv.stride[0]) + int(conv.kernel_size[0])
    return length


def load_audio(item, audio_sr, cache):
    path = str(item["wav_path"])
    if cache.get("path") != path:
        wav, sr = torchaudio.load(path)
        if wav.ndim == 2:
            wav = wav[0]
        if sr != audio_sr:
            wav = torchaudio.functional.resample(wav, sr, audio_sr)
        cache.update({"path": path, "wav": wav.to(torch.float32).contiguous()})
    wav = cache["wav"]
    start = int(round(item["start_ms"] / 1000.0 * audio_sr))
    end = int(round(item["end_ms"] / 1000.0 * audio_sr))
    end = max(start, min(end, int(wav.shape[0])))
    return wav[start:end]


def extract_features(model, wav, layer_idx, all_layers, min_samples, device):
    assert not (all_layers and layer_idx is not None)
    original_length = int(wav.shape[0])
    if original_length < min_samples:
        wav = F.pad(wav, (0, min_samples - original_length))
    wav = wav.unsqueeze(0).to(device)
    lengths = torch.tensor([original_length], device=device, dtype=torch.long)
    with torch.no_grad():
        features, out_lengths = model.extract_features(wav, lengths)
    if all_layers:
        features = torch.stack(features, dim=2)
    else:
        features = features[layer_idx if layer_idx is not None else -1]
    token_count = min(int(out_lengths[0].item()), int(features.shape[1]))
    return features[0, :token_count].cpu().numpy().astype(np.float32, copy=False)


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
    model = load_model(args.aves_config_path, args.aves_model_path).to(device)
    min_samples = min_input_samples(model)
    convs = [getattr(layer, "conv", layer) for layer in model.feature_extractor.conv_layers]
    samples_per_timebin = args.audio_sr * dataset.audio_params[2] / dataset.audio_params[0]
    geometry = convolution_geometry(
        [conv.kernel_size[0] for conv in convs],
        [conv.stride[0] for conv in convs],
        samples_per_timebin,
    )

    rows = []
    used = 0
    audio_cache = {}
    indices = balanced_event_indices(dataset.spec_dataset, args.balanced_events, args.event_seed)
    for item in chunked_items(dataset, args.num_timebins, args.chunk_timebins, indices):
        embeddings = extract_features(
            model,
            load_audio(item, args.audio_sr, audio_cache),
            args.encoder_layer_idx,
            args.all_layers,
            min_samples,
            device,
        )
        if embeddings.shape[0] == 0:
            continue
        labels, token_edges = convolution_feature_map(item["labels"], embeddings.shape[0], geometry)
        assert embeddings.shape[0] == labels.shape[0]
        row = {
            "item": item,
            "encoded_embeddings": embeddings,
            "labels_downsampled": labels,
            "token_edges": token_edges,
        }
        used, keep_going = append_limited(rows, row, args.max_points, used)
        if not keep_going:
            break
    save_concatenated_embeddings(
        args.out_dir,
        rows,
        model_name=args.model_name,
        audio_sr=args.audio_sr,
        encoder_layer_idx=args.encoder_layer_idx,
        all_layers=args.all_layers,
        chunk_timebins=args.chunk_timebins,
        feature_center_timebins=geometry[0],
        feature_stride_timebins=geometry[1],
        balanced_events=args.balanced_events,
        event_seed=args.event_seed,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Extract AVES embeddings into an embedding folder.")
    parser.add_argument("--spec_dir", required=True)
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--annotation_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--aves_model_path", required=True)
    parser.add_argument("--aves_config_path", required=True)
    parser.add_argument("--model_name", default="birdaves_biox_base")
    parser.add_argument("--audio_sr", type=int, default=16000)
    parser.add_argument("--recording_mode", default="events", choices=["events", "background", "full_recordings"])
    parser.add_argument("--recording_stem")
    parser.add_argument("--bird")
    parser.add_argument("--wav_exts", default=".wav,.flac,.ogg,.mp3")
    parser.add_argument("--encoder_layer_idx", type=int)
    parser.add_argument("--all_layers", action="store_true")
    parser.add_argument("--chunk_timebins", type=int, default=1000)
    parser.add_argument("--num_timebins", type=int, default=0)
    parser.add_argument("--max_points", type=int, default=0)
    parser.add_argument("--balanced_events", type=int, default=0)
    parser.add_argument("--event_seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    save_embeddings(parse_args())
