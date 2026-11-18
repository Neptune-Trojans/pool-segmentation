import numpy as np
import torch

from pool_segmentation import PoolModel
from pool_segmentation.model.network import UNet


def save_untrained_model(path):
    """A model.pt as `python -m pool_segmentation.model train` saves it (untrained here)."""
    path.parent.mkdir(parents=True)
    torch.save({"network": UNet(2, pretrained=False).state_dict(), "classes": ["other", "water"], "encoder": "resnet18",
                "size": [128, 64]}, path)
    return path


def test_a_saved_model_predicts_a_class_per_pixel_and_outlines_its_water(tmp_path):
    model = PoolModel.load(save_untrained_model(tmp_path / "20261117-090000" / "model.pt"), device="cpu")
    frame = np.full((90, 160, 3), 120, np.uint8)

    classes = model.predict(frame)
    water = model.water(frame)

    assert (model.classes, model.size) == (("other", "water"), (128, 64))
    assert classes.shape == (90, 160) and classes.dtype == np.uint8 and set(np.unique(classes)) <= {0, 1}
    assert all(len(polygon) >= 3 and all(0 <= x < 160 and 0 <= y < 90 for x, y in polygon) for polygon in water)
