import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.external_models.data_loader import WavFromSpectrogramDataset
from src.core.utils import downsample_labels


def load_model(model_name):
    from transformers import AutoFeatureExtractor, HubertModel

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name)
    model.eval()
    return feature_extractor, model


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


def select_hidden(outputs, layer_idx):
    if layer_idx is None:
        return outputs.last_hidden_state
    hidden_states = outputs.hidden_states
    idx = int(layer_idx)
    if idx < 0:
        idx = len(hidden_states) + idx
    assert 0 <= idx < len(hidden_states), f"encoder_layer_idx out of range: {layer_idx}"
    return hidden_states[idx]


def extract_features(feature_extractor, model, wav, audio_sr, layer_idx, device):
    if int(wav.shape[0]) < 400:
        wav = F.pad(wav, (0, 400 - int(wav.shape[0])))
    inputs = feature_extractor(
        wav.detach().cpu().numpy().astype(np.float32, copy=False),
        sampling_rate=audio_sr,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=layer_idx is not None)
    hidden = select_hidden(outputs, layer_idx)
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for index in range(len(dataset)):
        item = dataset[index]
        name = f"{index:06d}_{item['recording_stem']}.npz"
        embeddings = extract_features(
            feature_extractor,
            model,
            load_audio(item, args.audio_sr),
            args.audio_sr,
            args.encoder_layer_idx,
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
            model_name=np.array(args.model_name),
            audio_sr=np.array(args.audio_sr),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Extract HuBERT embeddings as .npz files.")
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
    return parser.parse_args()


if __name__ == "__main__":
    save_embeddings(parse_args())
