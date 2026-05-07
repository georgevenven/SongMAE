from pathlib import Path

import numpy as np
import torch

import aves
from individual_id.audio_augmentations import augment_audio_segment

try:
    from transformers import AutoFeatureExtractor, AutoModel
    from transformers.modeling_utils import PreTrainedModel
except Exception:
    AutoFeatureExtractor = None
    AutoModel = None
    PreTrainedModel = None


def _require_transformers():
    if AutoFeatureExtractor is None or AutoModel is None or PreTrainedModel is None:
        raise RuntimeError(
            "Bird-MAE requires transformers. Install transformers before using encoder=BirdMAE."
        )


def _patch_transformers_5_custom_code_compat():
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}


def load_model_state_for_inference(args):
    _require_transformers()
    _patch_transformers_5_custom_code_compat()

    model_name = args.get("bird_mae_model_name") or "DBD-research-group/Bird-MAE-Base"
    audio_sr = int(args.get("bird_mae_audio_sr") or 32000)
    wav_root = args.get("wav_root")
    wav_manifest = args.get("wav_manifest")
    wav_exts = aves._parse_wav_exts(args.get("wav_exts"))

    if not wav_root and not wav_manifest:
        raise ValueError("Bird-MAE requires wav_root or wav_manifest.")

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.global_pool = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    freq_patches = int(model.patch_embed.patch_hw[0])
    time_patches = int(model.patch_embed.patch_hw[1])

    return {
        "model": model,
        "feature_extractor": feature_extractor,
        "device": device,
        "model_name": model_name,
        "audio_sr": audio_sr,
        "freq_patches": freq_patches,
        "time_patches": time_patches,
        "wav_root": None if wav_root is None else str(Path(wav_root).resolve()),
        "wav_manifest": None if wav_manifest is None else str(Path(wav_manifest).resolve()),
        "wav_exts": wav_exts,
    }


def _embed_audio(wav, model_state):
    wav_np = wav.detach().cpu().numpy().astype(np.float32, copy=False)
    features = model_state["feature_extractor"](wav_np)
    if not torch.is_tensor(features):
        raise ValueError(f"Unexpected Bird-MAE feature extractor output: {type(features)}")
    features = features.to(model_state["device"])
    with torch.no_grad():
        outputs = model_state["model"](features)
    hidden = outputs.last_hidden_state[:, 1:, :]
    hidden = hidden.detach().cpu().numpy().astype(np.float32, copy=False)[0]
    time_patches = int(model_state["time_patches"])
    freq_patches = int(model_state["freq_patches"])
    assert hidden.shape[0] == time_patches * freq_patches
    hidden = hidden.reshape(time_patches, freq_patches, hidden.shape[1])
    return hidden.mean(axis=1).astype(np.float32, copy=False)


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

            encoded = _embed_audio(wav, model_state)
            token_labels = aves._token_labels_from_units(
                window["labels_original"],
                token_len=int(encoded.shape[0]),
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
        raise ValueError("No valid Bird-MAE embeddings extracted for the requested recording set.")

    return {
        "segments": segments,
        "audio_sr": int(raw["audio_sr"]),
        "patch_width": 1,
        "checkpoint": args.get("checkpoint") or "",
    }
