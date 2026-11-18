"""A labeling dataset: one folder (default data/labels/pool/, git-ignored), see docs/labeling.md.

  frames.json             the picked frames: name, video, frame index, size
  images/<name>.jpg       the frames
  auto/<name>.json        the model's labels (python -m pool_segmentation.labeling auto), never edited
  labels/<name>.json      the labels to check and fix (start as a copy of auto/, or drawn by hand)
  masks/<name>.png        training masks drawn from labels/ (one class number per pixel)

A frame's name is <video name>_<frame index, 6 digits>, e.g. IMG_4884_000261.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pool_segmentation.labeling.labels import REVIEWED, FrameLabels


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
        self.auto_dir = self.root / "auto"
        self.labels_dir = self.root / "labels"
        self.masks = self.root / "masks"

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
        known = {f.name: f for f in self.frames()}
        known.update({f.name: f for f in frames})
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "frames.json").write_text(json.dumps([asdict(f) for f in sorted(known.values(), key=lambda f: f.name)], indent=1))

    def image_path(self, name: str) -> Path:
        return self.images / f"{name}.jpg"

    # -- labels --------------------------------------------------------------------------------------------------

    def labels(self, name: str) -> FrameLabels | None:
        path = self.labels_dir / f"{name}.json"
        return FrameLabels.from_json(path) if path.exists() else None

    def auto(self, name: str) -> FrameLabels | None:
        """The model's labels of a frame, as `auto` made them (None if it hasn't labeled the frame)."""
        path = self.auto_dir / f"{name}.json"
        return FrameLabels.from_json(path) if path.exists() else None

    def save(self, labels: FrameLabels) -> None:
        labels.to_json(self.labels_dir / f"{labels.name}.json")

    def save_auto(self, labels: FrameLabels) -> None:
        """Save the model's labels; they also become the labels to check, unless those were already reviewed."""
        labels.to_json(self.auto_dir / f"{labels.name}.json")
        current = self.labels(labels.name)
        if current is None or current.status not in REVIEWED:
            self.save(labels)

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
