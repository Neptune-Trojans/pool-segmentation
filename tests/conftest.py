"""Shared by all tests: a small test video. No test reaches the Hugging Face Hub (the published model)."""

import cv2
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def no_hub(monkeypatch):
    """Every test: downloading or publishing on the Hugging Face Hub fails, as offline. Tests of the published model
    put a stand-in in place of these."""

    def offline(*args, **kwargs):
        raise ConnectionError("no Hugging Face Hub in tests")

    monkeypatch.setattr("huggingface_hub.snapshot_download", offline)
    monkeypatch.setattr("huggingface_hub.HfApi", offline)


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
