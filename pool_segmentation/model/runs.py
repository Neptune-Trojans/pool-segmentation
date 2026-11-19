"""Training runs: `python -m pool_segmentation.model train` saves each in its own folder, <runs>/<YYYYmmdd-HHMMSS>/
(model.pt, summary.json, log.csv, val/ pictures)."""

from pathlib import Path

MODELS_DIR = Path("data/models/pool")


def list_runs(root: str | Path = MODELS_DIR) -> list[str]:
    """The names of the training runs in `root` that finished (have a model.pt), newest first."""
    # Folder names are date-times: sorted = in time
    return sorted((path.parent.name for path in Path(root).glob("*/model.pt")), reverse=True)


def find_model(name: str = "latest", root: str | Path = MODELS_DIR) -> Path:
    """The model.pt of training run `name` in `root`, or of the newest run that finished (has a model.pt) for
    "latest". Raises FileNotFoundError if there is none."""
    root = Path(root)
    if name == "latest":
        runs = list_runs(root)
        if not runs:
            raise FileNotFoundError(f"No trained pool model in {root}: train one with python -m pool_segmentation.model train")
        return root / runs[0] / "model.pt"
    path = root / name / "model.pt"
    if not path.exists():
        raise FileNotFoundError(f"No pool model {path} (give the name of a run folder in {root}, or latest)")
    return path
