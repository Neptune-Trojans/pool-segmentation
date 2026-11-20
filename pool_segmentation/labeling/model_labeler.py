"""First labels of a frame from a trained pool model (docs/labeling.md): its water as polygons with few corners, to
check and fix on the Labels page. The model has no rope class yet: ropes are drawn by hand."""

from pathlib import Path

import numpy as np

from pool_segmentation.labeling import shapes
from pool_segmentation.labeling.labels import Point
from pool_segmentation.model.runs import model_name


class ModelLabeler:
    def __init__(self, model_path: str | Path, device: str | None = None):
        from pool_segmentation.model.predict import PoolModel  # imports torch

        self.model = PoolModel.load(model_path, device)
        self.name = f"pool-unet {model_name(model_path)}"  # a run's name, also for the published model

    def label(self, frame: np.ndarray) -> tuple[list[list[Point]], list[list[Point]]]:
        """(water polygons, rope polygons) of a BGR frame."""
        return shapes.mask_polygons(self.model.predict(frame) == 1), []
