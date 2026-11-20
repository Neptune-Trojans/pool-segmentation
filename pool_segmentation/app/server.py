"""The Labels page: check and fix the labels of the frames picked for training, and the Add videos page: pick frames
from new videos and let the newest trained model label them (docs/labeling.md).

Start it with `python -m pool_segmentation.app`. Another app can show the pages too, under a path of its own:

    app.mount("/labels", create_app("data/labels/pool", home=("← Videos", "/")))

The pages use relative links only, so they work at the root and under such a path.
"""

import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from pool_segmentation.app.add_videos import VIDEO_SUFFIXES, VideoAdder, is_video, list_folder, thumbnail
from pool_segmentation.labeling.dataset import Dataset
from pool_segmentation.labeling.frames import EVERY_SECONDS, MAX_PER_VIDEO
from pool_segmentation.labeling.labels import CLASSES, FrameLabels
from pool_segmentation.labeling.shapes import add_water
from pool_segmentation.model.hub import HUB_REPO
from pool_segmentation.model.runs import HUB, MODELS_DIR, list_runs

STATIC_DIR = Path(__file__).parent / "static"
VIDEOS_DIR = Path("data/input")
MAX_EVERY_SECONDS = 3600.0
MAX_FRAMES_PER_VIDEO = 1000


class SaveLabelsRequest(BaseModel):
    water: list[list[list[float]]] = []  # polygons: [[x, y], ...]
    ropes: list[list[list[float]]] = []  # polygons: each rope's outline


class EditWaterRequest(BaseModel):
    water: list[list[list[float]]] = []  # the frame's water polygons as they are on the page
    add: list[list[float]]  # the polygon to merge into the water


def is_video_name(name: str) -> bool:
    """A plain file name (no folder, not hidden) with a video's extension."""
    plain = name == Path(name).name and "\\" not in name and not name.startswith(".")
    return plain and Path(name).suffix.lower() in VIDEO_SUFFIXES


class AddVideosRequest(BaseModel):
    videos: list[str]  # absolute paths, as the Add videos page lists them
    every_seconds: float = Field(EVERY_SECONDS, gt=0, le=MAX_EVERY_SECONDS)
    max_per_video: int = Field(MAX_PER_VIDEO, ge=1, le=MAX_FRAMES_PER_VIDEO)
    model: str | None = "latest"  # the training run that makes the first labels, "latest", "hub" (the published
    # model), or None: none


def create_app(labels_dir: str | Path = "data/labels/pool", home: tuple[str, str] | None = None,
               models_dir: str | Path = MODELS_DIR, videos_dir: str | Path = VIDEOS_DIR) -> FastAPI:
    """The Labels and Add videos pages for the labeling dataset in `labels_dir`. `home`: (text, url) of a link back to
    the app that shows the pages, e.g. ("← Videos", "/"); none on its own. `models_dir`: the training runs, for the
    first labels of new frames. `videos_dir`: the folder the Add videos page opens first, and where video files chosen
    or dropped on it are copied (made on the first copy)."""
    dataset = Dataset(labels_dir)
    models_dir, videos_dir = Path(models_dir), Path(videos_dir)
    adder = VideoAdder(dataset, models_dir)
    app = FastAPI(title="pool-segmentation")
    app.state.adder = adder
    # Only requests addressed to this computer: a web page on another site can't reach the server by pointing its
    # own domain name at 127.0.0.1 (DNS rebinding)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    def labeled_frame(name: str):
        # Only names of picked frames are accepted, so a request can't reach any other file
        frame = dataset.frame(name)
        if frame is None:
            raise HTTPException(404, f"No frame '{name}' in {dataset.root}")
        return frame

    @app.get("/api/labels")
    def list_labels() -> dict:
        frames = []
        for frame in dataset.frames():
            labels = dataset.labels(frame.name)
            frames.append({"name": frame.name, "video": frame.video, "frame_idx": frame.frame_idx,
                           "status": labels.status if labels else "no labels",
                           "water": len(labels.water) if labels else 0, "ropes": len(labels.ropes) if labels else 0})
        return {"dataset": str(dataset.root), "classes": CLASSES, "frames": frames, "counts": dataset.status_counts(),
                "home": {"text": home[0], "url": home[1]} if home else None}

    @app.get("/api/labels/{name}")
    def frame_labels(name: str) -> dict:
        """The frame and its labels (empty ones if the model has not labeled it yet)."""
        frame = labeled_frame(name)
        labels = dataset.labels(name) or dataset.empty_labels(frame)
        return {"frame": asdict(frame), "labels": labels.to_dict()}

    @app.get("/api/labels/{name}/image")
    def frame_image(name: str) -> FileResponse:
        labeled_frame(name)
        path = dataset.image_path(name)
        if not path.is_file():
            raise HTTPException(404, f"Image not found: {path}")
        return FileResponse(path, media_type="image/jpeg")

    @app.put("/api/labels/{name}")
    def save_labels(name: str, request: SaveLabelsRequest) -> dict:
        """Save the frame's checked shapes. The status comes from them: "accepted" if they are still the model's labels,
        "fixed" if not (a fixed frame stays fixed). The frame's video, index and size come from the dataset, not the page."""
        frame = labeled_frame(name)
        previous = dataset.labels(name)
        try:
            labels = FrameLabels.from_dict({
                "video": frame.video, "frame_idx": frame.frame_idx, "image": frame.image, "width": frame.width,
                "height": frame.height, "status": "fixed", "water": request.water, "ropes": request.ropes,
                "source": previous.source if previous else "",
            })
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        if previous and previous.status in ("auto", "accepted") and labels.same_shapes(previous):
            labels.status = "accepted"
        dataset.save(labels)
        return {"labels": labels.to_dict(), "counts": dataset.status_counts()}

    @app.post("/api/labels/{name}/water")
    def edit_frame_water(name: str, request: EditWaterRequest) -> dict:
        """The page's + Water: the water polygons with an area merged in, simplified again (nothing is saved)."""
        frame = labeled_frame(name)
        if any(len(polygon) < 3 for polygon in request.water + [request.add]):
            raise HTTPException(400, "Every polygon needs at least 3 corners")
        water = add_water([[tuple(pt) for pt in polygon] for polygon in request.water],
                          [tuple(pt) for pt in request.add], frame.width, frame.height)
        return {"water": [[[round(x, 1), round(y, 1)] for x, y in polygon] for polygon in water]}

    @app.delete("/api/labels/{name}")
    def remove_frame(name: str) -> dict:
        """Remove a frame from the dataset (image, labels, mask); picking frames again skips it."""
        labeled_frame(name)
        dataset.remove_frame(name)
        return {"counts": dataset.status_counts()}

    # -----------------------------------------------------------------------------------------------------------
    # Add videos page: videos in a folder, adding them in the background (add_videos.py)

    def start_folder() -> Path:
        return (videos_dir if videos_dir.is_dir() else Path.cwd()).resolve()

    @app.get("/api/add")
    def add_options() -> dict:
        """What the Add videos page starts with: the folder to open, the trained models (newest first) and the
        published one, the default sample rate, the job running or last run."""
        return {"folder": str(start_folder()), "upload_folder": str(videos_dir.resolve()),
                "models": list_runs(models_dir), "models_dir": str(models_dir), "hub": HUB_REPO,
                "dataset": str(dataset.root), "defaults": {"every_seconds": EVERY_SECONDS, "max_per_video": MAX_PER_VIDEO},
                "home": {"text": home[0], "url": home[1]} if home else None, "job": adder.status()}

    @app.get("/api/add/folder")
    def add_folder(path: str | None = None, every_seconds: float = EVERY_SECONDS, max_per_video: int = MAX_PER_VIDEO) -> dict:
        """Subfolders and videos of `path` (default: the videos folder), each video with its frames in the dataset
        and the new frames picking it would add at this sample rate. `path` can also be a video: its folder, with
        the video as `selected`."""
        if not (0 < every_seconds <= MAX_EVERY_SECONDS and 1 <= max_per_video <= MAX_FRAMES_PER_VIDEO):
            raise HTTPException(400, "every_seconds or max_per_video is out of range")
        folder = (Path(path).expanduser() if path else start_folder()).resolve()
        selected = None
        if folder.is_file():
            if not is_video(folder):
                raise HTTPException(404, f"Not a folder or a video ({', '.join(sorted(VIDEO_SUFFIXES))}): {folder}")
            selected, folder = str(folder), folder.parent
        if not folder.is_dir():
            raise HTTPException(404, f"Folder or video not found: {folder}")
        try:
            return {**list_folder(folder, dataset, every_seconds, max_per_video), "selected": selected}
        except PermissionError as e:
            raise HTTPException(403, f"Can't read {folder}: {e.strerror}") from e

    @app.put("/api/add/upload")
    async def upload_video(request: Request, name: str) -> dict:
        """Copy a video file chosen or dropped on the page (the request's body) to the videos folder, as `name`.
        Never replaces a file: 409 if the folder has one with that name."""
        if not is_video_name(name):
            raise HTTPException(400, f"Not a video file name ({', '.join(sorted(VIDEO_SUFFIXES))}): {name}")
        folder = videos_dir.resolve()
        target = folder / name
        if target.exists():
            raise HTTPException(409, f"{target} exists already: open that folder and tick it there")
        folder.mkdir(parents=True, exist_ok=True)
        # To a hidden partial file first, then renamed: the folder never lists half a video
        partial = folder / f".{name}.part"
        try:
            size = 0
            with partial.open("wb") as out:
                async for chunk in request.stream():
                    out.write(chunk)
                    size += len(chunk)
            if not size:
                raise HTTPException(400, f"{name} is empty")
            if target.exists():  # copied meanwhile by another request
                raise HTTPException(409, f"{target} exists already: open that folder and tick it there")
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)
        return {"path": str(target), "size": size}

    @app.get("/api/add/thumbnail")
    def video_thumbnail(path: str) -> Response:
        video = Path(path)
        if not video.is_absolute() or not is_video(video):
            raise HTTPException(404, f"Not a video: {path}")
        try:
            return Response(thumbnail(video), media_type="image/jpeg")
        except OSError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/api/add")
    def add_videos(request: AddVideosRequest) -> dict:
        """Start adding videos: pick their frames, then the model's first labels. 409 while a job is running."""
        videos = [Path(v) for v in request.videos]
        if not videos:
            raise HTTPException(400, "No videos selected")
        for video in videos:
            if not video.is_absolute() or not is_video(video):
                raise HTTPException(400, f"Not a video: {video}")
        names = [v.stem for v in videos]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise HTTPException(400, f"Several videos named {', '.join(repeated)}: their frames would get the same "
                                     f"names. Add them one at a time")
        runs = list_runs(models_dir)
        model = (runs[0] if runs else HUB) if request.model == "latest" else request.model  # latest: else the published
        if model is not None and model not in runs and model != HUB:
            raise HTTPException(400, f"No trained model '{request.model}' in {models_dir}")
        try:
            job = adder.start(videos, request.every_seconds, request.max_per_video, model)
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e
        return {"job": job}

    @app.get("/api/add/job")
    def add_job() -> dict:
        return {"job": adder.status()}

    @app.post("/api/add/job/cancel")
    def cancel_add_job() -> dict:
        return {"job": adder.cancel()}

    # -----------------------------------------------------------------------------------------------------------
    # Pages

    @app.get("/")
    def labels_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "labels.html")

    @app.get("/add")
    def add_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "add.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
