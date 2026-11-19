"""A labeling dataset: one folder (default data/labels/pool/, git-ignored), see docs/labeling.md.

  frames.json             the picked frames: name, video, frame index, size
  images/<name>.jpg       the frames
  labels/<name>.json      the labels: the model's first labels (python -m pool_segmentation.labeling auto), checked
                          and fixed on the Labels page, or drawn there by hand; `status` says which
  masks/<name>.png        training masks drawn from labels/ (one class number per pixel)
  removed.json            names of frames removed on the Labels page: picking frames again skips them

A frame's name is <video name>_<frame index, 6 digits>, e.g. IMG_4884_000261.
"""

import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from pool_segmentation.labeling.labels import FrameLabels


@dataclass(frozen=True)
class Frame:
    name: str
    video: str
    frame_idx: int
    width: int
    height: int

    @property
    def image(self) -> str:
        return f"{self.name}.jpg"


def frame_name(video: str | Path, frame_idx: int) -> str:
    return f"{Path(video).stem}_{frame_idx:06d}"


class Dataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.images = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.masks = self.root / "masks"
        # frames.json and removed.json are read, changed and written by one thread at a time (the Labels page removes
        # frames while the Add videos page adds some)
        self._lock = threading.RLock()

    # -- frames --------------------------------------------------------------------------------------------------

    def frames(self) -> list[Frame]:
        """The picked frames, sorted by name (video, then frame index)."""
        path = self.root / "frames.json"
        if not path.exists():
            return []
        return sorted((Frame(**f) for f in json.loads(path.read_text())), key=lambda f: f.name)

    def frame(self, name: str) -> Frame | None:
        return next((f for f in self.frames() if f.name == name), None)

    def add_frames(self, frames: list[Frame]) -> None:
        with self._lock:
            known = {f.name: f for f in self.frames()}
            known.update({f.name: f for f in frames})
            self._write_json("frames.json", [asdict(f) for f in sorted(known.values(), key=lambda f: f.name)])

    def removed(self) -> set[str]:
        """Names of the frames removed from the dataset: picking frames again skips them, so they don't come back."""
        path = self.root / "removed.json"
        return set(json.loads(path.read_text())) if path.exists() else set()

    def remove_frame(self, name: str) -> bool:
        """Remove a frame: its entry, image, labels and mask. Its name goes to removed.json, so picking frames again
        skips it. False if there is no such frame."""
        with self._lock:
            frames = self.frames()
            if not any(f.name == name for f in frames):
                return False
            self._write_json("removed.json", sorted(self.removed() | {name}))
            self._write_json("frames.json", [asdict(f) for f in frames if f.name != name])
        for path in (self.image_path(name), self.labels_dir / f"{name}.json", self.masks / f"{name}.png"):
            path.unlink(missing_ok=True)
        return True

    def _write_json(self, filename: str, data) -> None:
        # To a temporary file first, then renamed: whoever reads the file meanwhile never gets half of it
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / filename
        temporary = path.with_name(f".{filename}.tmp")
        temporary.write_text(json.dumps(data, indent=1))
        os.replace(temporary, path)

    def image_path(self, name: str) -> Path:
        return self.images / f"{name}.jpg"

    # -- labels --------------------------------------------------------------------------------------------------

    def labels(self, name: str) -> FrameLabels | None:
        path = self.labels_dir / f"{name}.json"
        return FrameLabels.from_json(path) if path.exists() else None

    def save(self, labels: FrameLabels) -> None:
        labels.to_json(self.labels_dir / f"{labels.name}.json")

    def empty_labels(self, frame: Frame) -> FrameLabels:
        """Labels of a frame the model has not labeled yet: no shapes."""
        return FrameLabels(video=frame.video, frame_idx=frame.frame_idx, image=frame.image, width=frame.width,
                           height=frame.height, status="auto")

    def status_counts(self) -> dict[str, int]:
        counts = {"no labels": 0, "auto": 0, "fixed": 0, "accepted": 0}
        for f in self.frames():
            labels = self.labels(f.name)
            counts["no labels" if labels is None else labels.status] += 1
        return counts
