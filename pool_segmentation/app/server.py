"""The Labels page: check and fix the labels of the frames picked for training (docs/labeling.md).

Start it with `python -m pool_segmentation.app`. Another app can show the page too, under a path of its own:

    app.mount("/labels", create_app("data/labels/pool", home=("← Videos", "/")))

The page uses relative links only, so it works at the root and under such a path.
"""

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from pool_segmentation.labeling.dataset import Dataset
from pool_segmentation.labeling.labels import CLASSES, FrameLabels
from pool_segmentation.labeling.shapes import add_water

STATIC_DIR = Path(__file__).parent / "static"


class SaveLabelsRequest(BaseModel):
    status: str
    water: list[list[list[float]]] = []  # polygons: [[x, y], ...]
    ropes: list[list[list[float]]] = []  # polygons: each rope's outline


class EditWaterRequest(BaseModel):
    water: list[list[list[float]]] = []  # the frame's water polygons as they are on the page
    add: list[list[float]]  # the polygon to merge into the water


def create_app(labels_dir: str | Path = "data/labels/pool", home: tuple[str, str] | None = None) -> FastAPI:
    """The Labels page for the labeling dataset in `labels_dir`. `home`: (text, url) of a link back to the app that
    shows the page, e.g. ("← Videos", "/"); none on its own."""
    dataset = Dataset(labels_dir)
    app = FastAPI(title="pool-segmentation")
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
        """The frame, its labels (empty ones if the model has not labeled it yet) and the model's original labels."""
        frame = labeled_frame(name)
        labels = dataset.labels(name) or dataset.empty_labels(frame)
        sam3 = dataset.sam3(name)
        return {"frame": asdict(frame), "labels": labels.to_dict(), "sam3": sam3.to_dict() if sam3 else None}

    @app.get("/api/labels/{name}/image")
    def frame_image(name: str) -> FileResponse:
        labeled_frame(name)
        path = dataset.image_path(name)
        if not path.is_file():
            raise HTTPException(404, f"Image not found: {path}")
        return FileResponse(path, media_type="image/jpeg")

    @app.put("/api/labels/{name}")
    def save_labels(name: str, request: SaveLabelsRequest) -> dict:
        """Save the frame's shapes and status; the frame's video, index and size come from the dataset, not the page."""
        frame = labeled_frame(name)
        previous = dataset.labels(name)
        try:
            labels = FrameLabels.from_dict({
                "video": frame.video, "frame_idx": frame.frame_idx, "image": frame.image, "width": frame.width,
                "height": frame.height, "status": request.status, "water": request.water, "ropes": request.ropes,
                "source": previous.source if previous else "",
            })
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
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

    @app.get("/")
    def labels_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "labels.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
