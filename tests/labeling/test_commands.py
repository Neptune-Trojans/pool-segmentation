import cv2
import numpy as np
import pytest

from pool_segmentation.labeling.commands import auto_label, pick, write_masks
from pool_segmentation.labeling.dataset import Dataset
from pool_segmentation.labeling.frames import add_video_frames, pick_indices
from pool_segmentation.labeling.labels import FrameLabels


class FakeLabeler:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def label(self, frame):
        self.calls += 1
        h, w = frame.shape[:2]
        water = [[(0, h / 2), (w - 1, h / 2), (w - 1, h - 1), (0, h - 1)]]
        return water, [[(0, h * 0.7 - 1.5), (w - 1, h * 0.7 - 1.5), (w - 1, h * 0.7 + 1.5), (0, h * 0.7 + 1.5)]]


def test_pick_indices_spreads_frames_and_caps_them():
    assert pick_indices(300, 30.0, 2.0, 25) == [30, 90, 150, 210, 270]  # 10 s: one frame every 2 s, centered
    assert len(pick_indices(10_000, 30.0, 2.0, 25)) == 25
    assert pick_indices(20, 30.0, 2.0, 25) == [10]  # shorter than one step: one frame from the middle


def test_pick_saves_frames_once(tmp_path, make_video):
    video = make_video(tmp_path / "input" / "IMG_1.mp4", frames=120)  # 4 s at 30 fps
    dataset = Dataset(tmp_path / "labels")

    assert pick(dataset, [video], every_seconds=1.0, max_per_video=25) == [
        "IMG_1_000015", "IMG_1_000045", "IMG_1_000075", "IMG_1_000105"]
    assert pick(dataset, [video], every_seconds=1.0, max_per_video=25) == []
    frame = dataset.frame("IMG_1_000045")
    assert (frame.video, frame.frame_idx, frame.width, frame.height) == (str(video), 45, 64, 48)
    assert cv2.imread(str(dataset.image_path("IMG_1_000045"))).shape == (48, 64, 3)


def test_pick_skips_removed_frames(tmp_path, make_video):
    video = make_video(tmp_path / "input" / "IMG_1.mp4", frames=120)
    dataset = Dataset(tmp_path / "labels")
    pick(dataset, [video], every_seconds=1.0, max_per_video=25)

    dataset.remove_frame("IMG_1_000045")

    assert pick(dataset, [video], every_seconds=1.0, max_per_video=25) == []  # not picked again
    assert pick(dataset, [video], every_seconds=0.5, max_per_video=25) == [  # a shorter step: the frames between
        "IMG_1_000007", "IMG_1_000022", "IMG_1_000037", "IMG_1_000052", "IMG_1_000067", "IMG_1_000082", "IMG_1_000097",
        "IMG_1_000112"]
    assert "IMG_1_000045" not in {f.name for f in dataset.frames()}


def test_picking_stopped_halfway_keeps_the_frames_saved_so_far(tmp_path, make_video):
    video = make_video(tmp_path / "input" / "IMG_1.mp4", frames=120)
    dataset = Dataset(tmp_path / "labels")
    seen = []

    def stop_at_frame_50(done, total):
        seen.append((done, total))
        if done == 50:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        add_video_frames(dataset, video, 1.0, 25, progress=stop_at_frame_50)

    assert seen[0] == (1, 106) and seen[-1] == (50, 106)  # frames read, of those up to the last one picked
    assert [f.name for f in dataset.frames()] == ["IMG_1_000015", "IMG_1_000045"]
    assert sorted(p.name for p in dataset.images.iterdir()) == ["IMG_1_000015.jpg", "IMG_1_000045.jpg"]
    assert add_video_frames(dataset, video, 1.0, 25) == ["IMG_1_000075", "IMG_1_000105"]  # the rest, next time


def test_auto_labels_new_frames_and_never_replaces_checked_labels(tmp_path, make_video):
    dataset = Dataset(tmp_path / "labels")
    pick(dataset, [make_video(tmp_path / "IMG_1.mp4", frames=60)], every_seconds=1.0, max_per_video=25)
    labeler = FakeLabeler()

    assert auto_label(dataset, labeler=labeler) == 2
    first = dataset.labels("IMG_1_000015")
    assert (first.status, first.source, len(first.water), len(first.ropes)) == ("auto", "fake", 1, 1)
    assert auto_label(dataset, labeler=labeler) == 0  # nothing new
    assert labeler.calls == 2

    first.status, first.ropes = "fixed", []
    dataset.save(first)
    assert auto_label(dataset, redo=True, labeler=labeler) == 1  # only the frame not checked yet
    assert dataset.labels("IMG_1_000015") == first  # checked by hand: kept
    assert dataset.labels("IMG_1_000045").status == "auto"
    assert dataset.status_counts() == {"no labels": 0, "auto": 1, "fixed": 1, "accepted": 0}


def test_auto_keeps_labels_checked_while_it_runs(tmp_path, make_video):
    dataset = Dataset(tmp_path / "labels")
    pick(dataset, [make_video(tmp_path / "IMG_1.mp4", frames=60)], every_seconds=1.0, max_per_video=25)
    frame = dataset.frame("IMG_1_000045")
    by_hand = FrameLabels(video=frame.video, frame_idx=frame.frame_idx, image=frame.image, width=frame.width,
                          height=frame.height, status="fixed", water=[[(0, 0), (63, 0), (63, 47)]])

    class CheckedMeanwhile(FakeLabeler):  # the frame is saved on the Labels page while the model labels it
        def label(self, image):
            if self.calls == 1:
                dataset.save(by_hand)
            return super().label(image)

    assert auto_label(dataset, labeler=CheckedMeanwhile()) == 1
    assert dataset.labels("IMG_1_000045") == by_hand


def test_auto_labels_only_the_frames_asked_for_and_reports_progress(tmp_path, make_video):
    dataset = Dataset(tmp_path / "labels")
    pick(dataset, [make_video(tmp_path / "IMG_1.mp4", frames=60), make_video(tmp_path / "IMG_2.mp4", frames=60)],
         every_seconds=1.0, max_per_video=25)
    steps = []

    assert auto_label(dataset, labeler=FakeLabeler(), only={"IMG_2_000015", "IMG_2_000045"},
                      progress=lambda done, total: steps.append((done, total))) == 2

    assert steps == [(1, 2), (2, 2)]
    assert dataset.status_counts() == {"no labels": 2, "auto": 2, "fixed": 0, "accepted": 0}
    assert dataset.labels("IMG_1_000015") is None


def test_auto_skips_frames_removed_while_it_runs(tmp_path, make_video):
    dataset = Dataset(tmp_path / "labels")
    pick(dataset, [make_video(tmp_path / "IMG_1.mp4", frames=90)], every_seconds=1.0, max_per_video=25)

    class RemovedMeanwhile(FakeLabeler):  # frames removed on the Labels page while the model labels the first
        def label(self, image):
            if self.calls == 0:
                dataset.remove_frame("IMG_1_000015")  # the one being labeled
                dataset.remove_frame("IMG_1_000075")  # one still to come
            return super().label(image)

    assert auto_label(dataset, labeler=RemovedMeanwhile()) == 1
    assert [f.name for f in dataset.frames()] == ["IMG_1_000045"]
    assert sorted(p.name for p in dataset.labels_dir.iterdir()) == ["IMG_1_000045.json"]  # no labels of removed frames


def test_masks_only_for_checked_frames(tmp_path, make_video):
    dataset = Dataset(tmp_path / "labels")
    pick(dataset, [make_video(tmp_path / "IMG_1.mp4", frames=60)], every_seconds=1.0, max_per_video=25)
    auto_label(dataset, labeler=FakeLabeler())
    checked = dataset.labels("IMG_1_000045")
    checked.status = "accepted"
    dataset.save(checked)

    assert write_masks(dataset) == 1
    mask = cv2.imread(str(dataset.masks / "IMG_1_000045.png"), cv2.IMREAD_UNCHANGED)
    assert mask.shape == (48, 64) and set(mask.flatten()) == {0, 1, 2}
    assert write_masks(dataset, include_unreviewed=True) == 2

    checked.status = "auto"
    dataset.save(checked)
    assert write_masks(dataset) == 0 and not list(dataset.masks.glob("*.png"))  # old mask removed


class FakePoolModel:
    """Stands in for a trained pool model: the lower half of every frame is water."""

    def predict(self, frame):
        classes = np.zeros(frame.shape[:2], np.uint8)
        classes[frame.shape[0] // 2:] = 1
        return classes


def test_auto_labels_with_the_newest_trained_model(tmp_path, make_video, monkeypatch):
    runs = tmp_path / "models"
    for run in ("20261116-223804", "20261116-224516"):
        (runs / run).mkdir(parents=True)
        (runs / run / "model.pt").write_bytes(b"")
    loaded = []
    monkeypatch.setattr("pool_segmentation.model.predict.PoolModel.load",
                        lambda path, device=None: loaded.append(path) or FakePoolModel())
    dataset = Dataset(tmp_path / "labels")
    pick(dataset, [make_video(tmp_path / "IMG_1.mp4", frames=60)], every_seconds=1.0, max_per_video=25)

    assert auto_label(dataset, models_dir=runs) == 2

    assert loaded == [runs / "20261116-224516" / "model.pt"]  # the newest run
    labels = dataset.labels("IMG_1_000015")
    assert (labels.status, labels.source) == ("auto", "pool-unet 20261116-224516")
    assert labels.water == [[(0.0, 24.0), (0.0, 47.0), (63.0, 47.0), (63.0, 24.0)]]  # the lower half, on the frame's edges
    assert labels.ropes == []  # the model has no rope class: ropes are drawn by hand


def test_auto_without_a_trained_model_says_to_train_one(tmp_path, make_video):
    dataset = Dataset(tmp_path / "labels")
    pick(dataset, [make_video(tmp_path / "IMG_1.mp4", frames=30)], every_seconds=1.0, max_per_video=25)

    with pytest.raises(FileNotFoundError, match="train one"):
        auto_label(dataset, models_dir=tmp_path / "no-models")
