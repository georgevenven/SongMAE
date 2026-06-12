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

from src.external_models.data_loader import WavFromSpectrogramDataset
from src.core.utils import downsample_labels


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


def load_audio(item, audio_sr):
    wav, sr = torchaudio.load(str(item["wav_path"]))
    if wav.ndim == 2:
        wav = wav[0]
    if sr != audio_sr:
        wav = torchaudio.functional.resample(wav, sr, audio_sr)
    start = int(round(item["start_ms"] / 1000.0 * audio_sr))
    end = int(round(item["end_ms"] / 1000.0 * audio_sr))
    end = max(start, min(end, int(wav.shape[0])))
    wav = wav[start:end].to(torch.float32).contiguous()
    return wav


def extract_features(model, wav, layer_idx, min_samples, device):
    original_length = int(wav.shape[0])
    if original_length < min_samples:
        wav = F.pad(wav, (0, min_samples - original_length))
    wav = wav.unsqueeze(0).to(device)
    lengths = torch.tensor([original_length], device=device, dtype=torch.long)
    with torch.no_grad():
        features, out_lengths = model.extract_features(wav, lengths)
    features = features[layer_idx] if layer_idx is not None else features[-1]
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for index in range(len(dataset)):
        item = dataset[index]
        name = f"{index:06d}_{item['recording_stem']}.npz"
        embeddings = extract_features(
            model,
            load_audio(item, args.audio_sr),
            args.encoder_layer_idx,
            min_samples,
            device,
        )
        labels = downsample_labels(item["labels"], embeddings.shape[0])
        assert embeddings.shape[0] == labels.shape[0]
        np.savez(
            out_dir / name,
            encoded_embeddings=embeddings,
            labels_downsampled=labels,
            labels_original=labels,
            recording_stem=np.array(item["recording_stem"]),
            spec_path=np.array(str(item["spec_path"])),
            wav_path=np.array(str(item["wav_path"])),
            start_ms=np.array(item["start_ms"]),
            end_ms=np.array(item["end_ms"]),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Extract AVES embeddings as .npz files.")
    parser.add_argument("--spec_dir", required=True)
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--annotation_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--aves_model_path", required=True)
    parser.add_argument("--aves_config_path", required=True)
    parser.add_argument("--audio_sr", type=int, default=16000)
    parser.add_argument("--recording_mode", default="events", choices=["events", "full_recordings"])
    parser.add_argument("--recording_stem")
    parser.add_argument("--bird")
    parser.add_argument("--wav_exts", default=".wav,.flac,.ogg,.mp3")
    parser.add_argument("--encoder_layer_idx", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    save_embeddings(parse_args())
