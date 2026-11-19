"""The labeling steps run by `python -m pool_segmentation.labeling` (docs/labeling.md): pick frames, first labels from a
trained pool model, training masks."""

import time
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from pool_segmentation.labeling.dataset import Dataset
from pool_segmentation.labeling.frames import add_video_frames
from pool_segmentation.labeling.labels import REVIEWED, FrameLabels, Point, rasterize
from pool_segmentation.model.runs import MODELS_DIR, find_model


class Labeler(Protocol):
    name: str

    def label(self, frame: np.ndarray) -> tuple[list[list[Point]], list[list[Point]]]: ...  # water, rope polygons


def create_labeler(model: str = "latest", models_dir: str | Path = MODELS_DIR) -> Labeler:
    """A trained pool model as labeler: `model` is a training run's folder name in `models_dir`, or "latest"."""
    # Imported here: loading the model pulls in torch
    from pool_segmentation.labeling.model_labeler import ModelLabeler

    return ModelLabeler(find_model(model, models_dir))


def pick(dataset: Dataset, videos: list[Path], every_seconds: float, max_per_video: int) -> list[str]:
    added = []
    for video in videos:
        names = add_video_frames(dataset, video, every_seconds, max_per_video)
        print(f"  {Path(video).name}: {len(names)} new frames")
        added += names
    print(f"{len(added)} frames added, {len(dataset.frames())} in {dataset.root}")
    return added


def auto_label(dataset: Dataset, redo: bool = False, labeler: Labeler | None = None, model: str = "latest",
               models_dir: str | Path = MODELS_DIR) -> int:
    """Label every frame that has no labels yet (with `redo`: also those not checked yet, again), with `labeler` or else
    the trained pool model `model` (create_labeler). Checked labels are never replaced. Returns how many were labeled."""

    def to_label(frame) -> bool:
        labels = dataset.labels(frame.name)
        return labels is None or (redo and labels.status not in REVIEWED)

    todo = [f for f in dataset.frames() if to_label(f)]
    if not todo:
        print("Every frame already has labels (use --redo to label the ones not checked yet again)")
        return 0
    labeler = labeler or create_labeler(model, models_dir)
    print(f"Labeling {len(todo)} frames with {labeler.name}")
    start, done = time.time(), 0
    for k, frame in enumerate(todo, 1):
        image = cv2.imread(str(dataset.image_path(frame.name)))
        if image is None:
            raise OSError(f"Cannot read {dataset.image_path(frame.name)}")
        water, ropes = labeler.label(image)
        if not to_label(frame):  # checked on the Labels page meanwhile
            continue
        dataset.save(FrameLabels(video=frame.video, frame_idx=frame.frame_idx, image=frame.image, width=frame.width,
                                 height=frame.height, status="auto", water=water, ropes=ropes, source=labeler.name))
        done += 1
        left = (len(todo) - k) * (time.time() - start) / k
        print(f"  {k}/{len(todo)} {frame.name}: {len(water)} water polygon(s), {len(ropes)} rope(s)"
              f"  (about {left / 60:.0f} min left)", flush=True)
    return done


def write_masks(dataset: Dataset, include_unreviewed: bool = False) -> int:
    """Training masks (PNG, one class number per pixel) for the reviewed frames (all frames with labels if
    `include_unreviewed`). Masks of frames no longer included are removed, so masks/ is exactly the training set."""
    dataset.masks.mkdir(parents=True, exist_ok=True)
    keep = set()
    for frame in dataset.frames():
        labels = dataset.labels(frame.name)
        if labels is None or (labels.status not in REVIEWED and not include_unreviewed):
            continue
        cv2.imwrite(str(dataset.masks / f"{frame.name}.png"), rasterize(labels))
        keep.add(f"{frame.name}.png")
    for old in dataset.masks.glob("*.png"):
        if old.name not in keep:
            old.unlink()
    print(f"{len(keep)} masks in {dataset.masks}")
    return len(keep)
