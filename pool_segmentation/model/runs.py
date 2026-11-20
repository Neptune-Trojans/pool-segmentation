"""Training runs: `python -m pool_segmentation.model train` saves each in its own folder, <runs>/<YYYYmmdd-HHMMSS>/
(model.pt, summary.json, log.csv, val/ pictures). Without one, the published model (hub.py) is used."""

import json
from pathlib import Path

MODELS_DIR = Path("data/models/pool")
HUB = "hub"  # find_model(): the published model ("hub:<run>": that version)


def list_runs(root: str | Path = MODELS_DIR) -> list[str]:
    """The names of the training runs in `root` that finished (have a model.pt), newest first."""
    # Folder names are date-times: sorted = in time
    return sorted((path.parent.name for path in Path(root).glob("*/model.pt")), reverse=True)


def find_model(name: str = "latest", root: str | Path = MODELS_DIR) -> Path:
    """The weights file of a pool model:

    - "latest" (default): the newest training run in `root` that finished (has a model.pt); without one, the
      published model (downloaded once from the Hugging Face Hub, then cached)
    - a training run's folder name in `root`: its model.pt
    - "hub": the newest published model; "hub:<run>": the published version of that training run

    Raises FileNotFoundError if there is none."""
    from pool_segmentation.model import hub  # huggingface_hub only when used

    root = Path(root)
    if name == HUB or name.startswith(f"{HUB}:"):
        return hub.download(name.partition(":")[2] or None)
    if name == "latest":
        runs = list_runs(root)
        if runs:
            return root / runs[0] / "model.pt"
        try:
            return hub.download()
        except FileNotFoundError as error:
            raise FileNotFoundError(f"No trained pool model in {root}: train one with python -m pool_segmentation.model "
                                    f"train. ({error})") from error
    path = root / name / "model.pt"
    if not path.exists():
        raise FileNotFoundError(f"No pool model {path} (give the name of a run folder in {root}, latest or hub)")
    return path


def model_name(path: str | Path) -> str:
    """The training run a weights file comes from: its run folder (model.pt), or the run in config.json beside a
    published model.safetensors."""
    path = Path(path)
    config = path.parent / "config.json"
    if path.suffix == ".safetensors" and config.exists():
        return json.loads(config.read_text()).get("run") or path.parent.name
    return path.parent.name
