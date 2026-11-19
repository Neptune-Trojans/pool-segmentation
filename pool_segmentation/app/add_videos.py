"""The Add videos page (docs/labeling.md): the videos in a folder with how many of their frames are in the dataset
and how many new ones a sample rate picks, and adding videos in the background: pick their frames, then the first
labels from a trained model. One job at a time; it runs on when the page is closed.

The model is loaded only when a job labels frames: the Labels page starts without torch.
"""

import threading
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2

from pool_segmentation.labeling import commands
from pool_segmentation.labeling.dataset import Dataset
from pool_segmentation.labeling.frames import add_video_frames, new_indices
from pool_segmentation.video import get_video_info, read_frames

VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
THUMBNAIL_WIDTH = 240


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES and path.is_file() and not path.name.startswith(".")


def video_of(frame_name: str) -> str:
    """The video name in a frame's name: IMG_4884_000261 -> IMG_4884."""
    return frame_name.rsplit("_", 1)[0]


def stored_path(video: Path) -> Path:
    """The video path to record in frames.json: relative to the current folder when inside it, as `pick data/input/...`
    records it (training groups frames by it), else absolute."""
    try:
        return video.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return video


# ---------------------------------------------------------------------------------------------------------------
# Videos in a folder

def list_folder(folder: Path, dataset: Dataset, every_seconds: float, max_per_video: int) -> dict:
    """Subfolders and videos of `folder` (absolute). Each video: size, length, how many of its frames are in the
    dataset (by name: <video name>_<frame index>), and how many new frames picking it would add."""
    entries = sorted(folder.iterdir(), key=lambda p: p.name.lower())
    in_dataset = Counter(video_of(f.name) for f in dataset.frames())
    videos = []
    for path in entries:
        if not is_video(path):
            continue
        stat = path.stat()
        try:
            info = video_info(path)
        except OSError:
            info = None  # OpenCV can't read it
        new = len(new_indices(dataset, path, info["frame_count"], info["fps"], every_seconds, max_per_video)) if info else 0
        videos.append({"file": path.name, "path": str(path), "name": path.stem, "size": stat.st_size,
                       "modified": stat.st_mtime, "info": info, "in_dataset": in_dataset[path.stem], "new_frames": new})
    return {
        "folder": str(folder),
        "parent": str(folder.parent) if folder.parent != folder else None,
        "folders": [{"name": p.name, "path": str(p)} for p in entries if p.is_dir() and not p.name.startswith(".")],
        "videos": videos,
    }


def video_info(path: Path) -> dict:
    """fps, width, height and frame count, cached until the file changes."""
    stat = path.stat()
    return _video_info(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4096)
def _video_info(path: str, mtime_ns: int, size: int) -> dict:
    info = asdict(get_video_info(path))
    del info["path"]
    return info


def thumbnail(path: Path) -> bytes:
    """The first frame as a small JPEG, cached until the file changes."""
    stat = path.stat()
    return _thumbnail(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=1024)
def _thumbnail(path: str, mtime_ns: int, size: int) -> bytes:
    frames = read_frames(path)
    try:
        frame = next(frames, None)
    finally:
        frames.close()
    if frame is None:
        raise OSError(f"No frame in {path}")
    height, width = frame.shape[:2]
    small = cv2.resize(frame, (THUMBNAIL_WIDTH, round(height * THUMBNAIL_WIDTH / width)), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise OSError(f"Cannot make a thumbnail of {path}")
    return jpeg.tobytes()


# ---------------------------------------------------------------------------------------------------------------
# Adding videos in the background

class Cancelled(Exception):
    """Raised in a progress callback to stop the job."""


@dataclass
class AddJob:
    videos: list[Path]  # absolute
    every_seconds: float
    max_per_video: int
    model: str | None  # the training run that makes the first labels; None: none, draw them by hand
    state: str = "running"  # running, done, failed, cancelled
    step: str = "picking"  # picking (frames from a video), loading (the model), labeling
    video: str | None = None  # the video being read
    videos_done: int = 0
    done: int = 0  # frames read of the video (picking), frames labeled (labeling)
    total: int = 0  # ... of how many
    added: list[str] = field(default_factory=list)  # names of the new frames
    labeled: int = 0
    error: str | None = None
    cancel_requested: bool = False
    started: float = field(default_factory=time.time)
    finished: float | None = None
    step_start: tuple[float, int] | None = None  # (time, done) at the step's start: its speed so far

    def to_dict(self, now: float) -> dict:
        seconds_left = None
        if self.state == "running" and self.step_start and self.done > self.step_start[1] and self.total:
            t0, d0 = self.step_start
            if now - t0 > 1:
                seconds_left = round((self.total - self.done) * (now - t0) / (self.done - d0), 1)
        return {
            "state": self.state, "step": self.step, "video": self.video, "videos_done": self.videos_done,
            "videos_total": len(self.videos), "done": self.done, "total": self.total, "added": len(self.added),
            "first_added": self.added[0] if self.added else None, "labeled": self.labeled, "model": self.model,
            "error": self.error, "cancel_requested": self.cancel_requested, "seconds_left": seconds_left,
            "seconds": round((self.finished or now) - self.started, 1),
        }


class VideoAdder:
    """Adds videos to the dataset in a background thread, one job at a time: picks their frames, then (with a model)
    labels the frames of these videos that have no labels yet."""

    def __init__(self, dataset: Dataset, models_dir: Path):
        self.dataset = dataset
        self.models_dir = models_dir
        self._job: AddJob | None = None
        self._lock = threading.Lock()  # guards _job and its fields

    def status(self) -> dict | None:
        with self._lock:
            return self._job.to_dict(time.time()) if self._job else None

    def start(self, videos: list[Path], every_seconds: float, max_per_video: int, model: str | None) -> dict:
        """Start adding videos. Raises RuntimeError if a job is running."""
        with self._lock:
            if self._job and self._job.state == "running":
                raise RuntimeError("Videos are being added already: wait for them, or cancel")
            self._job = job = AddJob(videos, every_seconds, max_per_video, model)
            threading.Thread(target=self._run, args=(job,), name="add-videos", daemon=True).start()
            return job.to_dict(time.time())

    def cancel(self) -> dict | None:
        """Stop the running job at its next frame; the frames added and labeled so far stay."""
        with self._lock:
            if self._job and self._job.state == "running":
                self._job.cancel_requested = True
            return self._job.to_dict(time.time()) if self._job else None

    def _set(self, job: AddJob, **changes) -> None:
        with self._lock:
            if "step" in changes or "video" in changes:
                changes.setdefault("done", 0)
                changes.setdefault("total", 0)
                changes["step_start"] = None
            for key, value in changes.items():
                setattr(job, key, value)

    def _progress(self, job: AddJob, done: int, total: int) -> None:
        """Called at every frame: raises Cancelled if the job should stop."""
        with self._lock:
            if job.cancel_requested:
                raise Cancelled
            if job.step_start is None:
                job.step_start = (time.time(), done)
            job.done, job.total = done, total

    def _run(self, job: AddJob) -> None:
        names = {video.stem for video in job.videos}
        before = {f.name for f in self.dataset.frames()}

        def new_frames() -> list[str]:
            return [f.name for f in self.dataset.frames() if f.name not in before and video_of(f.name) in names]

        end = {"state": "failed", "error": "stopped"}
        try:
            for k, video in enumerate(job.videos):
                self._set(job, step="picking", video=video.stem, videos_done=k)
                add_video_frames(self.dataset, stored_path(video), job.every_seconds, job.max_per_video,
                                 progress=lambda done, total: self._progress(job, done, total))
                self._set(job, videos_done=k + 1, added=new_frames())
            if job.model:
                # The frames of these videos without labels: the new ones, and any left without by an earlier job
                todo = {f.name for f in self.dataset.frames()
                        if video_of(f.name) in names and self.dataset.labels(f.name) is None}
                if todo:
                    self._set(job, step="loading", video=None)
                    labeler = commands.create_labeler(job.model, self.models_dir)  # seconds: imports torch
                    self._set(job, step="labeling", total=len(todo))
                    self._progress(job, 0, len(todo))  # cancelled while the model was loading: stop here
                    commands.auto_label(self.dataset, labeler=labeler, only=todo,
                                        progress=lambda done, total: self._progress(job, done, total))
            end = {"state": "done"}
        except Cancelled:
            end = {"state": "cancelled"}
        except Exception as error:  # shown on the page; the frames added so far stay
            traceback.print_exc()
            end = {"state": "failed", "error": str(error) or type(error).__name__}
        finally:
            # The end state together with what was added: the page never sees a finished job with old numbers
            added = new_frames()
            labeled = sum(self.dataset.labels(name) is not None for name in added)
            self._set(job, **end, added=added, labeled=labeled, finished=time.time())
