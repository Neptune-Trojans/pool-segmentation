"""Train the pool segmentation network, and publish a trained model (docs/model.md). Run in the folder that holds the
data:

    python -m pool_segmentation.model train      # on the checked frames of data/labels/pool; a run folder in data/models/pool/
    python -m pool_segmentation.model publish    # the newest run to the Hugging Face Hub (HF_TOKEN in the environment or .env)
"""

import argparse
import os
from pathlib import Path

from pool_segmentation.model.hub import HUB_REPO
from pool_segmentation.model.runs import MODELS_DIR, list_runs


def size(text: str) -> tuple[int, int]:
    width, _, height = text.lower().partition("x")
    try:
        return int(width), int(height)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, e.g. 1024x576, got '{text}'") from None


def env_file_token(path: Path = Path(".env")) -> str | None:
    """HF_TOKEN from a .env file (KEY=value lines), or None."""
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        key, _, value = line.strip().removeprefix("export ").partition("=")
        if key.strip() == "HF_TOKEN" and value.strip():
            return value.strip().strip("'\"")
    return None


def publish(args, parser) -> None:
    from pool_segmentation.model.hub import publish as publish_run  # imports torch (reads model.pt)

    runs = list_runs(args.models)
    run = args.run or (runs[0] if runs else None)
    if run is None or run not in runs:
        parser.exit(1, f"No training run {run or ''} with a model.pt in {args.models}\n")
    # The token: HF_TOKEN in the environment or in .env, else the one saved by `hf auth login`
    token = os.environ.get("HF_TOKEN") or env_file_token()
    print(f"Publishing {Path(args.models) / run} to {args.repo} ...")
    url = publish_run(Path(args.models) / run, args.repo, token)
    print(f"Published: {url} (tag {run}). The package now downloads it where there is no trained run of its own.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m pool_segmentation.model",
                                     description="Train the pool segmentation network, publish a trained model.")
    commands = parser.add_subparsers(dest="command", required=True)
    t = commands.add_parser("train", help="Train on the checked frames (water / other)")
    t.add_argument("--dataset", default="data/labels/pool", help="Labeling dataset (default: data/labels/pool)")
    t.add_argument("--out", default="data/models/pool", help="Folder for run folders (default: data/models/pool)")
    t.add_argument("--epochs", type=int, default=40, help="Passes over the training frames (default 40)")
    t.add_argument("--batch", type=int, default=8, help="Frames per training step (default 8)")
    t.add_argument("--size", type=size, default=(1024, 576), help="Network input, sides divisible by 32 (default 1024x576)")
    t.add_argument("--lr", type=float, default=1e-3, help="Learning rate of the decoder; the encoder's is 10× lower (default 1e-3)")
    t.add_argument("--encoder", default="resnet18", choices=["resnet18", "resnet34"], help="Pretrained encoder (default resnet18)")
    t.add_argument("--val-every", type=int, default=6, help="Every n-th frame of each video validates (default 6)")
    t.add_argument("--device", default="auto", help="mps, cuda, cpu or auto (default: the fastest here)")
    t.add_argument("--workers", type=int, default=4, help="Processes preparing training frames (default 4)")
    t.add_argument("--seed", type=int, default=0)
    p = commands.add_parser("publish", help="Publish a trained model to the Hugging Face Hub: the package's default")
    p.add_argument("run", nargs="?", help="Training run folder name (default: the newest)")
    p.add_argument("--models", default=str(MODELS_DIR), help=f"Folder of the training runs (default: {MODELS_DIR})")
    p.add_argument("--repo", default=HUB_REPO, help=f"Hub model repo (default: {HUB_REPO})")
    args = parser.parse_args()

    if args.command == "publish":
        publish(args, parser)
        return
    from pool_segmentation.model.train import train  # imports torch

    try:
        run = train(args.dataset, args.out, epochs=args.epochs, batch=args.batch, size=args.size, lr=args.lr,
                    encoder=args.encoder, val_every=args.val_every, device=args.device, workers=args.workers, seed=args.seed)
    except ValueError as e:
        parser.exit(1, f"{e}\n")
    import json

    summary = json.loads((run / "summary.json").read_text())
    print(f"\nSaved {run}/model.pt ({summary['seconds']} s)")
    if summary["val_iou"]:
        print(f"Validation water IoU: mean {summary['val_iou_mean']:.3f}, worst {summary['val_iou_min']:.3f}")
        print("Worst frames: " + ", ".join(f"{name} {iou:.3f}" for name, iou in list(summary["val_iou"].items())[:3]))
        print(f"Pictures: {run}/val/")


if __name__ == "__main__":
    main()
