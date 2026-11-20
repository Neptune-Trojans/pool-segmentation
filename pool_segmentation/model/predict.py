"""A trained pool model: the class of every pixel of a frame (0 other, 1 water)."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pool_segmentation.model.data import input_size, prepare, resize
from pool_segmentation.model.network import UNet
from pool_segmentation.model.runs import model_name
from pool_segmentation.polygons import Polygon, water_polygons


def pick_device(name: str | None = None) -> torch.device:
    """The given device, or the fastest one here: the Mac's GPU (mps), an NVIDIA GPU (cuda), else the CPU."""
    if name and name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_classes(net: UNet, frame: np.ndarray, size: tuple[int, int], device: torch.device) -> np.ndarray:
    """Class number per pixel of a BGR frame, at the frame's own size (uint8). The frame is scaled like the training
    frames (its long side to the long side of `size`, the training view; shape kept, `input_size`), and the scores
    scaled back up before picking the class."""
    net.eval()
    x = torch.from_numpy(prepare(resize(frame, input_size(frame.shape, max(size)))))[None].to(device)
    scores = F.interpolate(net(x), size=frame.shape[:2], mode="bilinear", align_corners=False)
    return scores.argmax(1)[0].to(torch.uint8).cpu().numpy()


class PoolModel:
    """A trained pool model: saved by `python -m pool_segmentation.model train` (a run's model.pt), or the published
    one (model.safetensors with its config.json, from the Hugging Face Hub: find_model())."""

    def __init__(self, net: UNet, classes: tuple[str, ...], size: tuple[int, int], device: torch.device, name: str = ""):
        self.net = net.to(device).eval()
        self.classes = classes
        self.size = size
        self.device = device
        self.name = name  # the training run it comes from

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "PoolModel":
        """Load a run's model.pt, or a published model.safetensors (config.json beside it)."""
        path = Path(path)
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            saved = json.loads((path.parent / "config.json").read_text())
            saved["network"] = load_file(path)
        else:
            saved = torch.load(path, map_location="cpu", weights_only=True)
        net = UNet(len(saved["classes"]), saved["encoder"], pretrained=False)
        net.load_state_dict(saved["network"])
        return cls(net, tuple(saved["classes"]), tuple(saved["size"]), pick_device(device), model_name(path))

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """Class number per pixel of a BGR frame (uint8, the frame's size): 0 other, 1 water."""
        return predict_classes(self.net, frame, self.size, self.device)

    def water(self, frame: np.ndarray) -> list[Polygon]:
        """The water's outline in a BGR frame: one polygon per piece of water, holes filled (water_polygons)."""
        return water_polygons(self.predict(frame) == 1)
