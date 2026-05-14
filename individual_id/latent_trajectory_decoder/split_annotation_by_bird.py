#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def bird_id(recording):
    return str(recording["recording"]["bird_id"]).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--birds_per_chunk", type=int, required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.annotation_json).read_text(encoding="utf-8"))
    birds = sorted({bird_id(recording) for recording in data["recordings"]})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for chunk_index, start in enumerate(range(0, len(birds), args.birds_per_chunk)):
        chunk_birds = birds[start : start + args.birds_per_chunk]
        chunk_set = set(chunk_birds)
        chunk = {
            "metadata": data.get("metadata", {}),
            "recordings": [
                recording for recording in data["recordings"] if bird_id(recording) in chunk_set
            ],
        }
        path = out_dir / f"chunk_{chunk_index:03d}.json"
        path.write_text(json.dumps(chunk, indent=2), encoding="utf-8")
        rows.append((path, chunk_birds, len(chunk["recordings"])))

    manifest = [
        {
            "path": str(path),
            "birds": birds,
            "recordings": recording_count,
        }
        for path, birds, recording_count in rows
    ]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for path, birds, recording_count in rows:
        print(f"{path}\t{len(birds)} birds\t{recording_count} recordings\t{','.join(birds)}")


if __name__ == "__main__":
    main()
