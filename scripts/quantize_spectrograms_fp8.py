import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


FP8_DTYPES = {
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}
STORAGE_NORMALIZATIONS = {"none", "per_file_zscore"}


def load_audio_params(spec_dir):
    path = spec_dir / "audio_params.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_audio_params(spec_dir, payload):
    path = spec_dir / "audio_params.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def quantize_array(arr, fp8_dtype, storage_normalization):
    arr = np.asarray(arr, dtype=np.float32)
    mean = None
    std = None
    if storage_normalization == "per_file_zscore":
        mean = np.float32(arr.mean())
        std = max(np.float32(arr.std()), np.float32(1e-6))
        arr = ((arr - mean) / std).astype(np.float32, copy=False)

    tensor = torch.from_numpy(arr).to(fp8_dtype)
    return tensor.view(torch.uint8).numpy(), mean, std


def decode_fp8_array(arr, storage_dtype):
    tensor = torch.from_numpy(np.array(arr, dtype=np.uint8, copy=True))
    return tensor.view(FP8_DTYPES[storage_dtype]).float().numpy()


def read_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {row["file"]: row for row in rows}


def write_manifest(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def quantize_dir(src_dir, dst_dir, storage_dtype, storage_normalization):
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_dir / "audio_params.json", dst_dir / "audio_params.json")

    rows = []
    for src_path in tqdm(sorted(src_dir.glob("*.npy")), desc="Quantizing"):
        arr = np.load(src_path, mmap_mode="r")
        codes, mean, std = quantize_array(arr, FP8_DTYPES[storage_dtype], storage_normalization)
        np.save(dst_dir / src_path.name, codes)
        if storage_normalization == "per_file_zscore":
            rows.append(
                {
                    "file": src_path.name,
                    "mean": float(mean),
                    "std": float(std),
                }
            )

    audio_params = load_audio_params(dst_dir)
    audio_params["storage_dtype"] = storage_dtype
    audio_params["storage_normalization"] = storage_normalization
    if storage_normalization == "per_file_zscore":
        manifest_name = "quantization_manifest.jsonl"
        audio_params["storage_manifest"] = manifest_name
        write_manifest(dst_dir / manifest_name, rows)
    else:
        audio_params.pop("storage_manifest", None)
    save_audio_params(dst_dir, audio_params)


def dequantize_dir(src_dir, dst_dir, output_normalization):
    dst_dir.mkdir(parents=True, exist_ok=True)
    audio_params = load_audio_params(src_dir)
    storage_dtype = audio_params.get("storage_dtype")
    storage_normalization = audio_params.get("storage_normalization", "none")
    assert storage_dtype in FP8_DTYPES
    assert storage_normalization in STORAGE_NORMALIZATIONS

    manifest = {}
    manifest_name = audio_params.get("storage_manifest")
    if manifest_name is not None:
        manifest = read_manifest(src_dir / manifest_name)

    for src_path in tqdm(sorted(src_dir.glob("*.npy")), desc="Dequantizing"):
        codes = np.load(src_path, mmap_mode="r")
        arr = decode_fp8_array(codes, storage_dtype)
        if output_normalization == "raw" and storage_normalization == "per_file_zscore":
            row = manifest[src_path.name]
            mean = np.float32(row["mean"])
            std = np.float32(row["std"])
            arr = (arr * std + mean).astype(np.float32, copy=False)
        np.save(dst_dir / src_path.name, arr.astype(np.float32, copy=False))

    output_audio_params = dict(audio_params)
    output_audio_params["storage_dtype"] = "float32"
    output_audio_params["storage_normalization"] = "none" if output_normalization == "raw" else storage_normalization
    if output_normalization == "raw":
        output_audio_params.pop("storage_manifest", None)
    elif manifest_name is not None:
        shutil.copy2(src_dir / manifest_name, dst_dir / manifest_name)
    save_audio_params(dst_dir, output_audio_params)


def main():
    parser = argparse.ArgumentParser(description="Quantize or dequantize TinyBird spectrogram directories.")
    parser.add_argument("--mode", choices=["quantize", "dequantize"], required=True)
    parser.add_argument("--src_dir", type=Path, required=True)
    parser.add_argument("--dst_dir", type=Path, required=True)
    parser.add_argument(
        "--storage_dtype",
        choices=sorted(FP8_DTYPES),
        default="float8_e4m3fn",
        help="FP8 format to use when quantizing.",
    )
    parser.add_argument(
        "--storage_normalization",
        choices=sorted(STORAGE_NORMALIZATIONS),
        default="per_file_zscore",
        help="Normalization to bake into FP8 storage.",
    )
    parser.add_argument(
        "--output_normalization",
        choices=["stored", "raw"],
        default="stored",
        help="When dequantizing, write the stored domain back out or reconstruct raw float32 spectrograms.",
    )
    args = parser.parse_args()

    if args.mode == "quantize":
        quantize_dir(
            args.src_dir.expanduser(),
            args.dst_dir.expanduser(),
            args.storage_dtype,
            args.storage_normalization,
        )
        return

    dequantize_dir(
        args.src_dir.expanduser(),
        args.dst_dir.expanduser(),
        args.output_normalization,
    )


if __name__ == "__main__":
    main()
