import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.external_models.data_loader import WavFromSpectrogramDataset
from src.core.utils import downsample_labels


def load_model(model_name):
    import tensorflow as tf
    from perch_hoplite.zoo import model_configs

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    return model_configs.load_model_by_name(model_name)


def load_audio(item, audio_sr):
    wav, sr = sf.read(str(item["wav_path"]), always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav[:, 0]
    if sr != audio_sr:
        wav = librosa.resample(wav, orig_sr=int(sr), target_sr=audio_sr)
    start = int(round(item["start_ms"] / 1000.0 * audio_sr))
    end = int(round(item["end_ms"] / 1000.0 * audio_sr))
    end = max(start, min(end, int(wav.shape[0])))
    return np.asarray(wav[start:end], dtype=np.float32)


def embed_window(model, wav):
    outputs = model.embed(wav)
    embedding = getattr(outputs, "embeddings", None)
    if embedding is None and isinstance(outputs, dict):
        embedding = outputs.get("embeddings")
    assert embedding is not None, "Perch embed() did not return embeddings"
    embedding = np.squeeze(np.asarray(embedding, dtype=np.float32))
    assert embedding.ndim == 1, f"unexpected Perch embedding shape: {embedding.shape}"
    return embedding


def extract_features(model, wav, window_samples):
    assert window_samples > 0
    embeddings = []
    for start in range(0, int(wav.shape[0]), window_samples):
        window = wav[start : start + window_samples]
        if window.size < window_samples:
            window = np.pad(window, (0, window_samples - window.size))
        embeddings.append(embed_window(model, window))
    assert embeddings, "empty audio segment"
    return np.stack(embeddings).astype(np.float32, copy=False)


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
    model = load_model(args.model_name)
    window_samples = int(round(args.window_seconds * args.audio_sr))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for index in range(len(dataset)):
        item = dataset[index]
        name = f"{index:06d}_{item['recording_stem']}.npz"
        embeddings = extract_features(model, load_audio(item, args.audio_sr), window_samples)
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
            window_seconds=np.array(args.window_seconds),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Perch 2.0 embeddings as .npz files.")
    parser.add_argument("--spec_dir", required=True)
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--annotation_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_name", default="perch_v2")
    parser.add_argument("--audio_sr", type=int, default=32000)
    parser.add_argument("--window_seconds", type=float, default=5.0)
    parser.add_argument("--recording_mode", default="events", choices=["events", "full_recordings"])
    parser.add_argument("--recording_stem")
    parser.add_argument("--bird")
    parser.add_argument("--wav_exts", default=".wav,.flac,.ogg,.mp3")
    return parser.parse_args()


if __name__ == "__main__":
    save_embeddings(parse_args())
