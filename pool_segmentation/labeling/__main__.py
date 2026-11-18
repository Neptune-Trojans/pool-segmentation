"""Make training data for the pool segmentation model (docs/labeling.md). Run in the folder that holds the data:

    python -m pool_segmentation.labeling pick data/input/*.MOV   # 1. pick frames (one every 2 s, at most 25 per video)
    python -m pool_segmentation.labeling auto                     # 2. first labels (water) from the newest trained model
    python -m pool_segmentation.app                               # 3. check and fix them on the Labels page
    python -m pool_segmentation.labeling masks                    # 4. training masks from the checked labels
    python -m pool_segmentation.labeling status                   # how many frames are checked
"""

import argparse
from pathlib import Path

from pool_segmentation.labeling.commands import auto_label, pick, write_masks
from pool_segmentation.labeling.dataset import Dataset

DEFAULT_DATASET = "data/labels/pool"


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m pool_segmentation.labeling", description="Make training data for the pool model.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help=f"Dataset folder (default: {DEFAULT_DATASET})")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("pick", help="Pick frames from videos")
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--every", type=float, default=2.0, help="Seconds between picked frames (default 2)")
    p.add_argument("--max-per-video", type=int, default=25, help="At most this many frames per video (default 25)")

    a = commands.add_parser("auto", help="First labels (water) from a trained pool model for frames that have none")
    a.add_argument("--redo", action="store_true", help="Label every frame again (checked labels are kept)")
    a.add_argument("--model", default="latest", help="Training run folder name, or latest (default: the newest)")
    a.add_argument("--models", default="data/models/pool", help="Folder of the training runs (default: data/models/pool)")

    m = commands.add_parser("masks", help="Training masks (PNG) from the checked labels")
    m.add_argument("--all", action="store_true", help="Also frames not checked yet")

    commands.add_parser("status", help="How many frames are checked")
    args = parser.parse_args()

    dataset = Dataset(args.dataset)
    if args.command == "pick":
        pick(dataset, args.videos, args.every, args.max_per_video)
    elif args.command == "auto":
        auto_label(dataset, redo=args.redo, model=args.model, models_dir=args.models)
    elif args.command == "masks":
        write_masks(dataset, include_unreviewed=args.all)
    else:
        counts = dataset.status_counts()
        print(f"{dataset.root}: {sum(counts.values())} frames: " + ", ".join(f"{k} {v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
