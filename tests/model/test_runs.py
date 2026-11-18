import pytest

from pool_segmentation import find_model


def test_latest_model_is_the_newest_training_run(tmp_path):
    for run in ("20261116-223804", "20261116-224516", "20261117-090000"):
        (tmp_path / run).mkdir()
    for run in ("20261116-223804", "20261116-224516"):  # the newest folder has no model.pt (stopped run)
        (tmp_path / run / "model.pt").write_bytes(b"")

    assert find_model("latest", tmp_path) == tmp_path / "20261116-224516" / "model.pt"
    assert find_model(root=tmp_path) == tmp_path / "20261116-224516" / "model.pt"  # latest is the default
    assert find_model("20261116-223804", tmp_path) == tmp_path / "20261116-223804" / "model.pt"
    with pytest.raises(FileNotFoundError, match="No pool model"):
        find_model("20261117-090000", tmp_path)
    with pytest.raises(FileNotFoundError, match="train one"):
        find_model("latest", tmp_path / "empty")
