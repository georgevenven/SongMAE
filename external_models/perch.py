from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

import aves
from individual_id.audio_augmentations import augment_audio_segment

try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    from perch_hoplite.zoo import model_configs
except Exception:
    model_configs = None


def _require_perch():
    if tf is None or model_configs is None:
        raise RuntimeError(
            "Perch requires tensorflow and perch_hoplite. "
            "Install them before using encoder=Perch."
        )


def _configure_tensorflow():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def load_model_state_for_inference(args):
    _require_perch()
    _configure_tensorflow()

    run_config = {}
    run_dir = args.get("run_dir")
    if run_dir:
        config_path = Path(run_dir) / "config.json"
        if config_path.exists():
            run_config = json_load(config_path)

    model_name = args.get("perch_model_name") or run_config.get("perch_model_name") or "perch_v2"
    audio_sr = int(args.get("perch_audio_sr") or run_config.get("perch_audio_sr") or 32000)
    window_seconds = float(
        args.get("perch_window_seconds") or run_config.get("perch_window_seconds") or 5.0
    )
    wav_root = args.get("wav_root") or run_config.get("wav_root")
    wav_manifest = args.get("wav_manifest") or run_config.get("wav_manifest")
    wav_exts = aves._parse_wav_exts(args.get("wav_exts") or run_config.get("wav_exts"))

    if not wav_root and not wav_manifest:
        raise ValueError("Perch requires wav_root or wav_manifest.")

    model = model_configs.load_model_by_name(model_name)
    return {
        "model": model,
        "model_name": model_name,
        "audio_sr": audio_sr,
        "window_seconds": window_seconds,
        "wav_root": None if wav_root is None else str(Path(wav_root).resolve()),
        "wav_manifest": None if wav_manifest is None else str(Path(wav_manifest).resolve()),
        "wav_exts": wav_exts,
    }


def json_load(path):
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _embed_window(model, waveform):
    waveform_np = waveform.detach().cpu().numpy().astype(np.float32, copy=False)
    outputs = model.embed(waveform_np)
    embedding = getattr(outputs, "embeddings", None)
    if embedding is None and isinstance(outputs, dict):
        embedding = outputs.get("embeddings")
    if embedding is None:
        raise ValueError("Perch embed() did not return embeddings.")
    embedding = np.asarray(embedding, dtype=np.float32)
    embedding = np.squeeze(embedding)
    if embedding.ndim != 1:
        raise ValueError(f"Unexpected Perch embedding shape: {embedding.shape}")
    return embedding


def _load_recording_audio_segments(args, model_state):
    wav_exts = args.get("wav_exts") or model_state["wav_exts"]
    wav_exts = aves._parse_wav_exts(wav_exts)
    wav_index = aves.build_wav_index(
        args.get("wav_root") or model_state["wav_root"],
        exts=wav_exts,
        manifest_path=args.get("wav_manifest") or model_state["wav_manifest"],
    )
    event_map = {}
    json_path = args.get("json_path")
    if json_path:
        event_map = aves._load_json_events_ms(json_path, selected_bird=args.get("bird"))

    recording_mode = args.get("recording_mode") or "events"
    recording_stem = args.get("recording_stem")
    if recording_stem is not None:
        stems = [recording_stem]
    else:
        stems = sorted(wav_index)
        if event_map:
            allowed = set(event_map)
            stems = [stem for stem in stems if stem in allowed]

    segments = []
    audio_sr = int(args.get("perch_audio_sr") or model_state["audio_sr"])
    for stem in stems:
        wav_path = wav_index[stem]
        wav_np, sr = sf.read(str(wav_path), always_2d=False)
        wav_np = np.asarray(wav_np, dtype=np.float32)
        if wav_np.ndim == 2:
            wav_np = wav_np[:, 0]
        if sr != audio_sr:
            wav_np = librosa.resample(wav_np, orig_sr=int(sr), target_sr=audio_sr)
        wav = torch.from_numpy(np.asarray(wav_np, dtype=np.float32)).contiguous()

        duration_ms = float(wav.shape[0]) / float(audio_sr) * 1000.0
        events = event_map.get(stem, [])
        if recording_mode == "full_recordings":
            units = []
            for event in events:
                units.extend(event["units"])
            selected_events = [{"onset_ms": 0.0, "offset_ms": duration_ms, "units": units}]
        else:
            selected_events = events

        if not selected_events:
            continue

        for event in selected_events:
            start_ms = max(0.0, min(float(event["onset_ms"]), duration_ms))
            end_ms = max(start_ms, min(float(event["offset_ms"]), duration_ms))
            if end_ms <= start_ms:
                continue

            start_sample = int(round(start_ms / 1000.0 * audio_sr))
            end_sample = int(round(end_ms / 1000.0 * audio_sr))
            end_sample = max(start_sample, min(end_sample, int(wav.shape[0])))
            if end_sample <= start_sample:
                continue

            units = []
            for unit in event["units"]:
                unit_start = max(start_ms, min(float(unit["onset_ms"]), end_ms))
                unit_end = max(unit_start, min(float(unit["offset_ms"]), end_ms))
                if unit_end <= unit_start:
                    continue
                units.append(
                    {
                        "onset_ms": unit_start - start_ms,
                        "offset_ms": unit_end - start_ms,
                        "id": int(unit["id"]),
                    }
                )

            segments.append(
                {
                    "recording_stem": stem,
                    "audio": wav[start_sample:end_sample].contiguous(),
                    "duration_ms": end_ms - start_ms,
                    "labels_original": units,
                }
            )

    return {"audio_sr": audio_sr, "segments": segments}


def extract_recording_embeddings_with_state(args, model_state):
    raw = _load_recording_audio_segments(args, model_state)

    target_samples = int(round(model_state["window_seconds"] * model_state["audio_sr"]))
    assert target_samples > 0

    segments = []
    for segment_index, raw_segment in enumerate(raw["segments"]):
        raw_segment = augment_audio_segment(raw_segment, args, segment_index)
        windows = aves._split_audio_segment_into_context_windows(
            raw_segment,
            audio_sr=int(raw["audio_sr"]),
            context_seconds=float(model_state["window_seconds"]),
        )
        for window in windows:
            wav = window["audio"]
            if wav.shape[0] <= 0:
                continue
            if wav.shape[0] < target_samples:
                wav = F.pad(wav, (0, target_samples - int(wav.shape[0])))
            elif wav.shape[0] > target_samples:
                wav = wav[:target_samples]
            embedding = _embed_window(model_state["model"], wav)
            segments.append(
                {
                    "recording_stem": window["recording_stem"],
                    "features": embedding.reshape(1, -1),
                }
            )

    return {
        "segments": segments,
        "audio_sr": int(raw["audio_sr"]),
        "checkpoint": args.get("checkpoint") or "",
    }
