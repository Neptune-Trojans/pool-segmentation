"""Pick frames from videos for labeling: evenly spread over each video, a capped number per video."""

from collections.abc import Callable
from pathlib import Path

import cv2

from pool_segmentation.labeling.dataset import Dataset, Frame, frame_name
from pool_segmentation.video import get_video_info, read_frames

JPEG_QUALITY = 95
EVERY_SECONDS = 2.0  # defaults of `pick` and of the Add videos page: one frame every 2 s ...
MAX_PER_VIDEO = 25  # ... at most 25 per video


def pick_indices(frame_count: int, fps: float, every_seconds: float, max_frames: int) -> list[int]:
    """Frame indices one every `every_seconds`, at most `max_frames`, evenly spread (half a step from each end)."""
    if frame_count <= 0:
        return []
    step = max(every_seconds * fps, 1)
    count = max(1, min(max_frames, int(frame_count / step)))
    spacing = frame_count / count
    return sorted({min(int(spacing * (k + 0.5)), frame_count - 1) for k in range(count)})


def new_indices(dataset: Dataset, video: str | Path, frame_count: int, fps: float, every_seconds: float,
                max_frames: int) -> list[int]:
    """The picked frame indices of a video that are not in the dataset yet (nor were removed from it)."""
    skip = {f.name for f in dataset.frames()} | dataset.removed()
    return [i for i in pick_indices(frame_count, fps, every_seconds, max_frames) if frame_name(video, i) not in skip]


def add_video_frames(dataset: Dataset, video: str | Path, every_seconds: float, max_frames: int,
                     progress: Callable[[int, int], None] | None = None) -> list[str]:
    """Save the picked frames of a video to the dataset, skipping frames already there or removed from it; returns
    the new names. `progress(frames read, frames to read)` is called as the video is read; if it raises, picking
    stops and the frames saved so far stay in the dataset."""
    info = get_video_info(video)
    wanted = set(new_indices(dataset, video, info.frame_count, info.fps, every_seconds, max_frames))
    if not wanted:
        return []
    dataset.images.mkdir(parents=True, exist_ok=True)
    added = []
    last = max(wanted)
    try:
        for i, frame in enumerate(read_frames(video)):
            if i in wanted:
                name = frame_name(video, i)
                cv2.imwrite(str(dataset.image_path(name)), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                added.append(Frame(name=name, video=str(video), frame_idx=i, width=frame.shape[1], height=frame.shape[0]))
            if progress:
                progress(i + 1, last + 1)
            if i >= last:
                break
    finally:
        dataset.add_frames(added)
    return [f.name for f in added]
