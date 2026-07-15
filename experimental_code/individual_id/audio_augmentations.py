import hashlib

import numpy as np
import torch
import torch.nn.functional as F


def _make_audio_rng(seed, recording_stem, segment_index):
    stem_hash = int(hashlib.sha1(recording_stem.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(int(seed) + stem_hash + int(segment_index))


def sample_speed_factor(args, recording_stem, segment_index):
    max_pct = float(args.get("train_audio_speed_max_pct", 0.0))
    if max_pct <= 0.0:
        return 1.0

    min_pct = float(args.get("train_audio_speed_min_pct", 0.0))
    rng = _make_audio_rng(
        seed=args.get("seed", 0),
        recording_stem=recording_stem,
        segment_index=segment_index,
    )
    shift_pct = float(rng.uniform(min_pct, max_pct))
    direction = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
    return 1.0 + direction * shift_pct


def _resample_waveform_for_speed(wav, speed_factor):
    if abs(float(speed_factor) - 1.0) < 1e-6:
        return wav.to(torch.float32).contiguous()

    target_len = max(1, int(round(wav.shape[-1] / float(speed_factor))))
    if target_len == int(wav.shape[-1]):
        return wav.to(torch.float32).contiguous()

    return F.interpolate(
        wav.to(torch.float32).view(1, 1, -1),
        size=target_len,
        mode="linear",
        align_corners=False,
    ).view(-1).contiguous()


def augment_audio_segment(raw_segment, args, segment_index):
    speed_factor = sample_speed_factor(
        args,
        recording_stem=raw_segment["recording_stem"],
        segment_index=segment_index,
    )
    if abs(float(speed_factor) - 1.0) < 1e-6:
        return raw_segment

    scaled_units = []
    for unit in raw_segment["labels_original"]:
        scaled_units.append(
            {
                "onset_ms": float(unit["onset_ms"]) / float(speed_factor),
                "offset_ms": float(unit["offset_ms"]) / float(speed_factor),
                "id": int(unit["id"]),
            }
        )

    return {
        "recording_stem": raw_segment["recording_stem"],
        "audio": _resample_waveform_for_speed(raw_segment["audio"], speed_factor),
        "duration_ms": float(raw_segment["duration_ms"]) / float(speed_factor),
        "labels_original": scaled_units,
    }
