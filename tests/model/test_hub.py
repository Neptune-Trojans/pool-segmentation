"""The published model: export a training run (safetensors, config, model card), publish it (a stand-in for the Hub),
and find_model() / PoolModel.load() with it. No test reaches the Hub (tests/conftest.py)."""

import json

import numpy as np
import pytest
import torch

from pool_segmentation import PoolModel, find_model
from pool_segmentation.model.hub import CONFIG, HUB_REPO, WEIGHTS, export, publish
from pool_segmentation.model.network import UNet

RUN = "20261119-231345"


@pytest.fixture
def run(tmp_path):
    """A training run folder as `python -m pool_segmentation.model train` saves it (an untrained model)."""
    folder = tmp_path / "models" / RUN
    (folder / "val").mkdir(parents=True)
    torch.manual_seed(0)
    torch.save({"network": UNet(2, pretrained=False).state_dict(), "classes": ["other", "water"], "encoder": "resnet18",
                "size": [128, 64]}, folder / "model.pt")
    (folder / "summary.json").write_text(json.dumps({
        "created": "2026-11-19T23:19:56", "epochs": 40, "val_iou_mean": 0.9956, "val_iou_min": 0.9901,
        "train_frames": ["IMG_1_000030", "IMG_1_000091", "IMG_2_000030"], "val_frames": ["IMG_1_000213"]}))
    (folder / "val" / "IMG_1_000213.jpg").write_bytes(b"a picture of real people")
    return folder


@pytest.fixture
def published(run, tmp_path):
    """The run in its published form, in a folder like the Hub's download cache."""
    export(run, tmp_path / "snapshot")
    return tmp_path / "snapshot"


def test_export_writes_the_published_form_and_it_predicts_the_same(run, published):
    config = json.loads((published / CONFIG).read_text())

    assert sorted(p.name for p in published.iterdir()) == ["README.md", CONFIG, WEIGHTS]  # no val/ pictures
    assert config | {"created": None} == {
        "run": RUN, "classes": ["other", "water"], "encoder": "resnet18", "size": [128, 64], "created": None,
        "epochs": 40, "train_frames": 3, "val_frames": 1, "videos": 2, "val_iou_mean": 0.9956, "val_iou_min": 0.9901}
    card = (published / "README.md").read_text()
    assert card.startswith("---\nlicense: mit\n") and RUN in card and "water IoU mean 0.996, worst 0.990" in card

    frame = np.random.default_rng(0).integers(0, 255, (90, 160, 3), dtype=np.uint8)
    original, loaded = PoolModel.load(run / "model.pt", "cpu"), PoolModel.load(published / WEIGHTS, "cpu")
    assert np.array_equal(original.predict(frame), loaded.predict(frame))
    assert (loaded.classes, loaded.size, loaded.name) == (("other", "water"), (128, 64), RUN)
    assert original.name == RUN


def test_without_a_trained_run_find_model_downloads_the_published_model(published, tmp_path, monkeypatch):
    calls = []

    def snapshot_download(repo_id, revision=None, allow_patterns=None):
        calls.append((repo_id, revision, allow_patterns))
        return str(published)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)

    assert find_model("latest", tmp_path / "no-runs") == published / WEIGHTS
    assert find_model("hub") == published / WEIGHTS
    assert find_model(f"hub:{RUN}") == published / WEIGHTS  # a given version: the run's tag
    assert calls == [(HUB_REPO, None, [WEIGHTS, CONFIG]), (HUB_REPO, None, [WEIGHTS, CONFIG]),
                     (HUB_REPO, RUN, [WEIGHTS, CONFIG])]


def test_a_trained_run_here_comes_before_the_published_model(run, monkeypatch):
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: pytest.fail("the Hub was asked"))

    assert find_model("latest", run.parent) == run / "model.pt"


def test_without_a_run_or_the_hub_find_model_says_to_train_one(tmp_path):
    # Offline (tests/conftest.py) and nothing downloaded
    with pytest.raises(FileNotFoundError, match="train one.*Can't get the published pool model"):
        find_model("latest", tmp_path / "no-runs")
    with pytest.raises(FileNotFoundError, match=f"Can't get the published pool model {HUB_REPO}@{RUN}"):
        find_model(f"hub:{RUN}")


def test_a_hub_download_without_the_model_files_is_refused(tmp_path, monkeypatch):
    (tmp_path / "empty").mkdir()
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: str(tmp_path / "empty"))

    with pytest.raises(FileNotFoundError, match="has no model.safetensors"):
        find_model("hub")


def test_publish_uploads_the_run_public_and_tagged_with_its_name(run, monkeypatch):
    calls = []

    class FakeApi:  # stands in for huggingface_hub.HfApi
        def __init__(self, token=None):
            calls.append(("token", token))

        def create_repo(self, repo_id, **options):
            calls.append(("create_repo", repo_id, options))

        def upload_folder(self, repo_id, folder_path, commit_message):
            calls.append(("upload", repo_id, sorted(p.name for p in __import__("pathlib").Path(folder_path).rglob("*")),
                          commit_message))
            return type("CommitInfo", (), {"oid": "abc123"})()

        def create_tag(self, repo_id, **options):
            calls.append(("tag", repo_id, options))

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)

    assert publish(run, token="hf_secret") == f"https://huggingface.co/{HUB_REPO}"

    assert calls == [
        ("token", "hf_secret"),
        ("create_repo", HUB_REPO, {"repo_type": "model", "private": False, "exist_ok": True}),
        ("upload", HUB_REPO, ["README.md", CONFIG, WEIGHTS], f"Model {RUN}, validation water IoU 0.996"),  # no val/
        ("tag", HUB_REPO, {"tag": RUN, "revision": "abc123", "exist_ok": True}),
    ]
