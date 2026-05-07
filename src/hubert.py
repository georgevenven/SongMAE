from pathlib import Path

import numpy as np
import torch

import aves
from individual_id.audio_augmentations import augment_audio_segment

try:
    from transformers import AutoFeatureExtractor, HubertModel
except Exception:
    AutoFeatureExtractor = None
    HubertModel = None


def _require_transformers():
    if AutoFeatureExtractor is None or HubertModel is None:
        raise RuntimeError(
            "HuBERT requires transformers. Install transformers before using encoder=HuBERT."
        )


def load_model_state_for_inference(args):
    _require_transformers()

    model_name = args.get("hubert_model_name") or "facebook/hubert-base-ls960"
    audio_sr = int(args.get("hubert_audio_sr") or 16000)
    wav_root = args.get("wav_root")
    wav_manifest = args.get("wav_manifest")
    wav_exts = aves._parse_wav_exts(args.get("wav_exts"))

    if not wav_root and not wav_manifest:
        raise ValueError("HuBERT requires wav_root or wav_manifest.")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return {
        "model": model,
        "feature_extractor": feature_extractor,
        "device": device,
        "model_name": model_name,
        "audio_sr": audio_sr,
        "wav_root": None if wav_root is None else str(Path(wav_root).resolve()),
        "wav_manifest": None if wav_manifest is None else str(Path(wav_manifest).resolve()),
        "wav_exts": wav_exts,
    }


def _select_hidden(outputs, encoder_layer_idx):
    if encoder_layer_idx is None:
        return outputs.last_hidden_state

    hidden_states = outputs.hidden_states
    idx = int(encoder_layer_idx)
    if idx < 0:
        idx = len(hidden_states) + idx
    if idx < 0 or idx >= len(hidden_states):
        raise ValueError(f"encoder_layer_idx out of range: {encoder_layer_idx} (num_layers={len(hidden_states)})")
    return hidden_states[idx]


def _embed_audio(wav, model_state, encoder_layer_idx):
    wav_np = wav.detach().cpu().numpy().astype(np.float32, copy=False)
    inputs = model_state["feature_extractor"](
        wav_np,
        sampling_rate=int(model_state["audio_sr"]),
        return_tensors="pt",
    )
    inputs = {key: value.to(model_state["device"]) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model_state["model"](
            **inputs,
            output_hidden_states=encoder_layer_idx is not None,
        )
    hidden = _select_hidden(outputs, encoder_layer_idx)
    return hidden[0].detach().cpu().numpy().astype(np.float32, copy=False)


def extract_recording_embeddings_with_state(args, model_state):
    raw = aves.load_recording_audio_segments(
        {
            "wav_root": args.get("wav_root") or model_state["wav_root"],
            "wav_manifest": args.get("wav_manifest") or model_state["wav_manifest"],
            "wav_exts": args.get("wav_exts") or model_state["wav_exts"],
            "audio_sr": args.get("audio_sr") or model_state["audio_sr"],
            "json_path": args.get("json_path"),
            "bird": args.get("bird"),
            "recording_stem": args.get("recording_stem"),
            "recording_mode": args.get("recording_mode"),
        }
    )

    segments = []
    context_seconds = args.get("audio_context_seconds")
    min_samples = 400
    for segment_index, raw_segment in enumerate(raw["segments"]):
        raw_segment = augment_audio_segment(raw_segment, args, segment_index)
        windows = aves._split_audio_segment_into_context_windows(
            raw_segment,
            audio_sr=int(raw["audio_sr"]),
            context_seconds=context_seconds,
        )
        for window in windows:
            wav = window["audio"]
            if int(wav.shape[0]) <= 0:
                continue
            if int(wav.shape[0]) < min_samples:
                wav = torch.nn.functional.pad(wav, (0, min_samples - int(wav.shape[0])))

            encoded = _embed_audio(wav, model_state, args.get("encoder_layer_idx"))
            token_len = int(encoded.shape[0])
            if token_len <= 0:
                continue

            token_labels = aves._token_labels_from_units(
                window["labels_original"],
                token_len=token_len,
                duration_ms=float(window["duration_ms"]),
            )
            segments.append(
                {
                    "recording_stem": window["recording_stem"],
                    "encoded_embeddings_before_pos_removal": encoded,
                    "encoded_embeddings_after_pos_removal": encoded,
                    "labels_original": token_labels,
                    "labels_downsampled": token_labels,
                }
            )

    if not segments:
        raise ValueError("No valid HuBERT tokens extracted for the requested recording set.")

    return {
        "segments": segments,
        "audio_sr": int(raw["audio_sr"]),
        "patch_width": 1,
        "checkpoint": args.get("checkpoint") or "",
    }
