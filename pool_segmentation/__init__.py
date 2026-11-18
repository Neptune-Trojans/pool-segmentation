"""Find a swimming pool's water in video frames (docs/model.md), and make the training data for it (docs/labeling.md).

    from pool_segmentation import PoolModel, find_model

    model = PoolModel.load(find_model())   # the newest training run in data/models/pool/
    classes = model.predict(frame)         # class per pixel of a BGR frame: 0 other, 1 water
    water = model.water(frame)             # the water's outline: one polygon [(x, y), ...] per piece of water

Parts:
    model/      the network: training (python -m pool_segmentation.model train) and prediction
    labeling/   training data: pick frames, first labels from a trained model (python -m pool_segmentation.labeling)
    app/        the Labels page: check and fix the labels in the browser (python -m pool_segmentation.app)
"""

from pool_segmentation.model.runs import find_model
from pool_segmentation.polygons import water_polygons

__all__ = ["PoolModel", "find_model", "water_polygons"]


def __getattr__(name: str):
    # PoolModel imports torch: only when it is used, so the labeling tools and the Labels page start fast
    if name == "PoolModel":
        from pool_segmentation.model.predict import PoolModel

        return PoolModel
    raise AttributeError(f"module 'pool_segmentation' has no attribute '{name}'")
