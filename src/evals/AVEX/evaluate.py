import argparse
import os
from pathlib import Path

from alp_data.io import filesystem_from_path
from avex.run_evaluate import main as evaluate
from avex.utils import experiment_tracking

from .birdset import keep_test_clean, use_local_manifests
from .songmae import SongMAEAVEX


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=Path, required=True)
    parser.add_argument("--patch", "-p", action="append", default=[])
    args = parser.parse_args()

    if metadata_dir := os.getenv("AVEX_EXPERIMENT_DIR"):
        Path(metadata_dir).mkdir(parents=True, exist_ok=True)
        experiment_tracking._GLOBAL_EXPERIMENT_DIR = metadata_dir
        experiment_tracking._fs = filesystem_from_path(metadata_dir)
    if birdset_root := os.getenv("AVEX_BIRDSET_ROOT"):
        use_local_manifests(birdset_root)
        keep_test_clean()
    assert SongMAEAVEX.name == "songmae"
    evaluate(config_path=args.config, patches=tuple(args.patch))


if __name__ == "__main__":
    main()
