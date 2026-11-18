"""Shared by all tests: a small test video."""

import cv2
import numpy as np
import pytest


@pytest.fixture
def make_video():
    """make_video(path, frames=10): write a small video (64×48, 30 fps) whose frames get brighter."""

    def make(path, frames=10, width=64, height=48):
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
        assert writer.isOpened(), f"cannot write {path}"
        for i in range(frames):
            writer.write(np.full((height, width, 3), i * 20 % 256, dtype=np.uint8))
        writer.release()
        return path

    return make
