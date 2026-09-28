"""Download pinned baseline assets only; never extract embeddings or run probes."""

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path(__file__).resolve().parents[1] / "files/review_baselines"
BIRDMAE_REVISION = "6cc416d1a7ae2af29b6b866499b3b047a8f01304"
BEATS_REVISION = "082fb1849d55bef1ee52ee8d8910b3adc69d4bc8"
SOURCE_REVISION = "31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd"
BEATS_SHA256 = "d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34"
BIRDMAE_SHA256 = "402103ecc81787ca6317ad78377fb38a912b3548f23a18f424d7a0dc31f754c5"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    birdmae = ROOT / "birdmae_base"
    snapshot_download(
        "DBD-research-group/Bird-MAE-Base", revision=BIRDMAE_REVISION,
        local_dir=birdmae, allow_patterns=["*.json", "*.py", "*.safetensors", "README.md"],
    )
    assert sha256(birdmae / "model.safetensors") == BIRDMAE_SHA256

    beats = ROOT / "beats"
    checkpoint = Path(hf_hub_download(
        "Bencr/beats-checkpoints", "BEATs_iter3_plus_AS2M.pt", repo_type="dataset",
        revision=BEATS_REVISION, local_dir=beats,
    ))
    assert sha256(checkpoint) == BEATS_SHA256
    source_url = f"https://raw.githubusercontent.com/microsoft/unilm/{SOURCE_REVISION}"
    for name in ["BEATs.py", "backbone.py", "modules.py", "README.md"]:
        path = beats / name
        with urllib.request.urlopen(f"{source_url}/beats/{name}", timeout=60) as source:
            with path.with_suffix(path.suffix + ".partial").open("wb") as destination:
                shutil.copyfileobj(source, destination)
        path.with_suffix(path.suffix + ".partial").replace(path)
    with urllib.request.urlopen(f"{source_url}/LICENSE", timeout=60) as source:
        (beats / "LICENSE").write_bytes(source.read())

    manifest = {
        "birdmae": {
            "repository": "DBD-research-group/Bird-MAE-Base", "revision": BIRDMAE_REVISION,
            "directory": str(birdmae), "sampling_rate": 32000,
        },
        "beats": {
            "checkpoint": str(checkpoint), "variant": "BEATs_iter3_plus_AS2M",
            "mirror": "https://huggingface.co/datasets/Bencr/beats-checkpoints",
            "mirror_revision": BEATS_REVISION, "sampling_rate": 16000,
            "source": "https://github.com/microsoft/unilm/tree/master/beats",
            "source_revision": SOURCE_REVISION,
            "mirror_reason": "Microsoft README OneDrive link returned HTTP 403 during preparation.",
        },
        "birdmae_playback_speeds": [1.0, 0.5, 0.25, 0.125],
        "sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for directory in [birdmae, beats] for path in sorted(directory.iterdir())
            if path.is_file()
        },
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Assets verified: {ROOT / 'manifest.json'}")


if __name__ == "__main__":
    main()
