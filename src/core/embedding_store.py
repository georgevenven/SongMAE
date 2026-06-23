import json
import shutil
from pathlib import Path

import numpy as np


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _tmp_dir(out_dir):
    out_dir = Path(out_dir)
    return out_dir.with_name(f"{out_dir.name}.tmp")


def _replace_dir(tmp_dir, out_dir):
    out_dir = Path(out_dir)
    if out_dir.exists():
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        else:
            out_dir.unlink()
    tmp_dir.replace(out_dir)


class EmbeddingStore:
    def __init__(self, path):
        self.path = Path(path)
        assert self.path.is_dir(), f"embedding folder not found: {self.path}"
        metadata_path = self.path / "metadata.json"
        self.metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    def __contains__(self, key):
        return (self.path / f"{key}.npy").exists()

    def __getitem__(self, key):
        assert key in self, f"{key} missing from {self.path}"
        return np.load(self.path / f"{key}.npy", mmap_mode="r")


def save_embedding_arrays(out_dir, arrays, metadata=None):
    tmp_dir = _tmp_dir(out_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)

    for key, value in arrays.items():
        np.save(tmp_dir / f"{key}.npy", np.asarray(value))

    metadata = metadata or {}
    text = json.dumps({key: _jsonable(value) for key, value in metadata.items()}, indent=2) + "\n"
    (tmp_dir / "metadata.json").write_text(text)
    _replace_dir(tmp_dir, out_dir)
    print(f"embeddings saved to {out_dir}")


class EmbeddingFolderWriter:
    def __init__(self, out_dir, shapes, dtypes, metadata=None):
        self.out_dir = Path(out_dir)
        self.tmp_dir = _tmp_dir(self.out_dir)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        self.tmp_dir.mkdir(parents=True)
        self.arrays = {
            key: np.lib.format.open_memmap(self.tmp_dir / f"{key}.npy", mode="w+", dtype=dtypes[key], shape=shape)
            for key, shape in shapes.items()
        }
        metadata = metadata or {}
        text = json.dumps({key: _jsonable(value) for key, value in metadata.items()}, indent=2) + "\n"
        (self.tmp_dir / "metadata.json").write_text(text)

    def close(self):
        for array in self.arrays.values():
            array.flush()
        _replace_dir(self.tmp_dir, self.out_dir)
        print(f"embeddings saved to {self.out_dir}")
