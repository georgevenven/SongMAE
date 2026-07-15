import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

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


def load_model(model_name):
    from transformers import AutoFeatureExtractor, HubertModel
    from transformers.utils import WEIGHTS_NAME
    from transformers.utils.hub import cached_file

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    state = torch.load(cached_file(model_name, WEIGHTS_NAME), map_location="cpu")
    prefix = "encoder.pos_conv_embed.conv."
    state[prefix + "parametrizations.weight.original0"] = state.pop(prefix + "weight_g")
    state[prefix + "parametrizations.weight.original1"] = state.pop(prefix + "weight_v")
    model = HubertModel.from_pretrained(model_name, state_dict=state)
    model.eval()
    return feature_extractor, model


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


def select_hidden(outputs, layer_idx, all_layers):
    assert not (all_layers and layer_idx is not None)
    if layer_idx is None:
        return torch.stack(outputs.hidden_states[1:], dim=2) if all_layers else outputs.last_hidden_state
    hidden_states = outputs.hidden_states[1:]
    idx = int(layer_idx)
    if idx < 0:
        idx = len(hidden_states) + idx
    assert 0 <= idx < len(hidden_states), f"encoder_layer_idx out of range: {layer_idx}"
    return hidden_states[idx]


def extract_features(feature_extractor, model, wav, audio_sr, layer_idx, all_layers, device):
    if int(wav.shape[0]) < 400:
        wav = F.pad(wav, (0, 400 - int(wav.shape[0])))
    inputs = feature_extractor(
        wav.detach().cpu().numpy().astype(np.float32, copy=False),
        sampling_rate=audio_sr,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=all_layers or layer_idx is not None)
    hidden = select_hidden(outputs, layer_idx, all_layers)
    return hidden[0].detach().cpu().numpy().astype(np.float32, copy=False)


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
    feature_extractor, model = load_model(args.model_name)
    model = model.to(device)
    samples_per_timebin = args.audio_sr * dataset.audio_params[2] / dataset.audio_params[0]
    geometry = convolution_geometry(model.config.conv_kernel, model.config.conv_stride, samples_per_timebin)

    rows = []
    used = 0
    audio_cache = {}
    for item in chunked_items(dataset, args.num_timebins, args.chunk_timebins):
        embeddings = extract_features(
            feature_extractor,
            model,
            load_audio(item, args.audio_sr, audio_cache),
            args.audio_sr,
            args.encoder_layer_idx,
            args.all_layers,
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
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Extract HuBERT embeddings into an embedding folder.")
    parser.add_argument("--spec_dir", required=True)
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--annotation_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", default="facebook/hubert-base-ls960")
    parser.add_argument("--audio_sr", type=int, default=16000)
    parser.add_argument("--recording_mode", default="events", choices=["events", "full_recordings"])
    parser.add_argument("--recording_stem")
    parser.add_argument("--bird")
    parser.add_argument("--wav_exts", default=".wav,.flac,.ogg,.mp3")
    parser.add_argument("--encoder_layer_idx", type=int)
    parser.add_argument("--all_layers", action="store_true")
    parser.add_argument("--chunk_timebins", type=int, default=1000)
    parser.add_argument("--num_timebins", type=int, default=0)
    parser.add_argument("--max_points", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    save_embeddings(parse_args())
