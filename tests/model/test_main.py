import argparse
import json
import sys
from pathlib import Path

import pytest

from pool_segmentation.model.__main__ import main, size


@pytest.fixture
def fake_train(tmp_path, monkeypatch):
    """Stands in for train() (no training): records its arguments and writes a run folder with a summary."""
    calls = []

    def train(dataset, out, **options):
        calls.append((dataset, out, options))
        if dataset == "empty":
            raise ValueError("0 checked frames in empty")
        run = tmp_path / "20261118-120000"
        run.mkdir()
        (run / "summary.json").write_text(json.dumps({
            "seconds": 12.5, "val_iou": {"IMG_1_000010": 0.95, "IMG_2_000020": 0.99},
            "val_iou_mean": 0.97, "val_iou_min": 0.95}))
        return run

    monkeypatch.setattr("pool_segmentation.model.train.train", train)
    return calls


def test_train_passes_its_options_and_prints_the_result(fake_train, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pool_segmentation.model", "train", "--dataset", "labels", "--out", "models",
                                      "--epochs", "3", "--size", "512x288", "--device", "cpu", "--workers", "0"])

    main()

    assert fake_train == [("labels", "models", {"epochs": 3, "batch": 8, "size": (512, 288), "lr": 1e-3,
                                                "encoder": "resnet18", "val_every": 6, "device": "cpu", "workers": 0,
                                                "seed": 0})]
    out = capsys.readouterr().out
    assert "Validation water IoU: mean 0.970, worst 0.950" in out
    assert "IMG_1_000010 0.950" in out


def test_train_without_checked_frames_stops_with_the_reason(fake_train, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pool_segmentation.model", "train", "--dataset", "empty"])

    with pytest.raises(SystemExit) as stop:
        main()

    assert stop.value.code == 1
    assert "0 checked frames" in capsys.readouterr().err


@pytest.fixture
def fake_publish(tmp_path, monkeypatch):
    """Stands in for hub.publish() (no upload): records its arguments. Two training runs in tmp_path/models; the
    command runs in tmp_path (its .env)."""
    calls = []
    monkeypatch.setattr("pool_segmentation.model.hub.publish",
                        lambda run, repo, token: calls.append((run, repo, token)) or f"https://huggingface.co/{repo}")
    for name in ("20261116-224516", "20261119-231345"):
        (tmp_path / "models" / name).mkdir(parents=True)
        (tmp_path / "models" / name / "model.pt").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return calls


def test_publish_sends_the_newest_run_with_the_token_from_env_file(fake_publish, tmp_path, monkeypatch, capsys):
    (tmp_path / ".env").write_text("OTHER=1\nHF_TOKEN='hf_from_file'\n")
    monkeypatch.setattr(sys, "argv", ["pool_segmentation.model", "publish", "--models", "models"])

    main()

    assert fake_publish == [(Path("models/20261119-231345"), "nlateralus/pool-segmentation", "hf_from_file")]
    assert "Published: https://huggingface.co/nlateralus/pool-segmentation (tag 20261119-231345)" in capsys.readouterr().out


def test_publish_a_given_run_with_the_token_from_the_environment(fake_publish, tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("HF_TOKEN=hf_from_file\n")
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    monkeypatch.setattr(sys, "argv", ["pool_segmentation.model", "publish", "20261116-224516", "--models", "models",
                                      "--repo", "someone/pool"])

    main()

    assert fake_publish == [(Path("models/20261116-224516"), "someone/pool", "hf_from_env")]


def test_publish_without_a_token_leaves_it_to_the_saved_login(fake_publish, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pool_segmentation.model", "publish", "--models", "models"])

    main()

    assert fake_publish[0][2] is None  # huggingface_hub then uses the token saved by `hf auth login`


def test_publish_an_unknown_run_stops(fake_publish, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pool_segmentation.model", "publish", "20990101-000000", "--models", "models"])

    with pytest.raises(SystemExit) as stop:
        main()

    assert stop.value.code == 1 and "No training run 20990101-000000" in capsys.readouterr().err
    assert fake_publish == []


def test_size_is_width_x_height():
    assert size("1024x576") == size("1024X576") == (1024, 576)
    with pytest.raises(argparse.ArgumentTypeError, match="WIDTHxHEIGHT"):
        size("1024")
