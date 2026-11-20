"""The published pool model, on the Hugging Face Hub (docs/model.md): `publish` uploads a training run's weights,
`download` gets the newest published model (or a given run's), cached in ~/.cache/huggingface.

Published form: model.safetensors (the network's weights only: no pickle), config.json (classes, encoder, input size,
the run and its validation IoU) and README.md (the model card). Each upload is tagged with its training run's name,
so a version can be pinned: find_model("hub:<run>"). The run's val/ pictures are never uploaded: they show real
people, like every frame of the dataset.
"""

import json
import tempfile
from pathlib import Path

HUB_REPO = "nlateralus/pool-segmentation"
WEIGHTS = "model.safetensors"
CONFIG = "config.json"
CODE_URL = "https://github.com/Neptune-Trojans/pool-segmentation"


def download(revision: str | None = None, repo_id: str = HUB_REPO) -> Path:
    """The published model's weights (model.safetensors, config.json beside it): the newest, or the training run
    tagged `revision`. Downloaded once, then from the cache; when online, a newer published model replaces it.
    Raises FileNotFoundError if it can't be had (offline and never downloaded, no such run)."""
    from huggingface_hub import snapshot_download

    try:
        # One call for both files: they come from the same version
        folder = Path(snapshot_download(repo_id, revision=revision, allow_patterns=[WEIGHTS, CONFIG]))
    except Exception as error:  # offline without a copy, no such repo or run (huggingface_hub has many kinds)
        where = f"{repo_id}@{revision}" if revision else repo_id
        raise FileNotFoundError(f"Can't get the published pool model {where}: {error}") from error
    if not (folder / WEIGHTS).exists() or not (folder / CONFIG).exists():
        raise FileNotFoundError(f"The published pool model {repo_id} has no {WEIGHTS} and {CONFIG}")
    return folder / WEIGHTS


def export(run: Path, folder: Path) -> dict:
    """Write training run `run` in the published form to `folder`: model.safetensors, config.json and README.md.
    Returns the config."""
    import torch  # only here: publishing needs the model, downloading doesn't
    from safetensors.torch import save_file

    saved = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    names = summary.get("train_frames", []) + summary.get("val_frames", [])
    config = {
        "run": run.name, "classes": saved["classes"], "encoder": saved["encoder"], "size": saved["size"],
        "created": summary.get("created"), "epochs": summary.get("epochs"),
        "train_frames": len(summary.get("train_frames", [])), "val_frames": len(summary.get("val_frames", [])),
        "videos": len({name.rsplit("_", 1)[0] for name in names}),
        "val_iou_mean": summary.get("val_iou_mean"), "val_iou_min": summary.get("val_iou_min"),
    }
    folder.mkdir(parents=True, exist_ok=True)
    save_file({name: tensor.contiguous() for name, tensor in saved["network"].items()}, folder / WEIGHTS)
    (folder / CONFIG).write_text(json.dumps(config, indent=1))
    (folder / "README.md").write_text(model_card(config))
    return config


def publish(run: Path, repo_id: str = HUB_REPO, token: str | None = None) -> str:
    """Upload training run `run` to the Hub repo `repo_id` (made public if new) as its newest model, tagged with the
    run's name. `token`: a Hugging Face token with write access (None: the one saved by `hf auth login`). Returns
    the model's page."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        config = export(run, Path(folder))
        iou = f", validation water IoU {config['val_iou_mean']:.3f}" if config["val_iou_mean"] else ""
        commit = api.upload_folder(repo_id=repo_id, folder_path=folder, commit_message=f"Model {run.name}{iou}")
    api.create_tag(repo_id, tag=run.name, revision=commit.oid, exist_ok=True)
    return f"https://huggingface.co/{repo_id}"


def model_card(config: dict) -> str:
    """The Hub page of the model (README.md): what it does, how to use it, this version's numbers."""
    width, height = config["size"]
    iou = ""
    if config["val_iou_mean"] is not None:
        iou = f"water IoU mean {config['val_iou_mean']:.3f}, worst {config['val_iou_min']:.3f}"
    return f"""---
license: mit
pipeline_tag: image-segmentation
tags:
- pytorch
- u-net
- swimming-pool
- semantic-segmentation
---

# pool-segmentation: a swimming pool's water

A small U-Net ({config['encoder']} encoder) that finds a swimming pool's **water** in video frames: a class for every
pixel, 0 other, 1 water (swimmers in the water, splashes and lane ropes count as water). About 25 ms per frame on a
MacBook. Made for swimming video analysis (dropping people outside the pool; later, finding the lanes).

Code, training and labeling tools: [{CODE_URL.removeprefix('https://')}]({CODE_URL}) (`pip install pool-segmentation`).

## Use

```python
from pool_segmentation import PoolModel, find_model

model = PoolModel.load(find_model())   # your newest training run in data/models/pool/, else this model
classes = model.predict(frame)         # BGR frame -> class per pixel: 0 other, 1 water
water = model.water(frame)             # the water's outline: one polygon [(x, y), ...] per piece of water
```

The package downloads this model the first time it is needed and keeps it in `~/.cache/huggingface`; when online, it
gets a newer one when there is one. A given version: `find_model("hub:{config['run']}")` (every version is tagged
with its training run).

## This version

| | |
|---|---|
| Training run | `{config['run']}` |
| Classes | {', '.join(f'{k} {name}' for k, name in enumerate(config['classes']))} |
| Input | frames scaled to {width}×{height} (the long side; shape kept) |
| Training | {config['train_frames']} hand-checked frames from {config['videos']} videos, {config['epochs']} epochs |
| Validation | {config['val_frames']} frames (every 6th of each video): {iou or 'not measured'} |

## Limits

Trained on frames filmed at a few pools, from the deck. Pools that look very different (another camera angle, light,
water color) may need some frames of their own: the package's labeling tools pick frames from your videos, let this
model label them, and you check them in the browser before training again. The training frames are not published:
they show real people.
"""
