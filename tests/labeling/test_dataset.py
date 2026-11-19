import cv2
import numpy as np

from pool_segmentation.labeling.dataset import Dataset, Frame
from pool_segmentation.labeling.labels import FrameLabels


def make_dataset(root) -> Dataset:
    """Two frames of IMG_1, both with an image, labels and a mask."""
    dataset = Dataset(root)
    frames = [Frame(f"IMG_1_{i:06d}", "data/input/IMG_1.MOV", i, 64, 48) for i in (15, 45)]
    dataset.add_frames(frames)
    dataset.images.mkdir(parents=True)
    dataset.masks.mkdir(parents=True)
    for f in frames:
        cv2.imwrite(str(dataset.image_path(f.name)), np.zeros((48, 64, 3), np.uint8))
        cv2.imwrite(str(dataset.masks / f"{f.name}.png"), np.zeros((48, 64), np.uint8))
        dataset.save(FrameLabels(video=f.video, frame_idx=f.frame_idx, image=f.image, width=64, height=48,
                                 status="accepted", water=[[(0, 24), (63, 24), (63, 47)]]))
    return dataset


def test_remove_a_frame_deletes_its_files_and_remembers_its_name(tmp_path):
    dataset = make_dataset(tmp_path / "labels")

    assert dataset.remove_frame("IMG_1_000015")

    assert [f.name for f in dataset.frames()] == ["IMG_1_000045"]
    assert not dataset.image_path("IMG_1_000015").exists()
    assert dataset.labels("IMG_1_000015") is None
    assert not (dataset.masks / "IMG_1_000015.png").exists()
    assert dataset.removed() == {"IMG_1_000015"}
    assert dataset.labels("IMG_1_000045").status == "accepted"  # the other frame is untouched
    assert dataset.status_counts() == {"no labels": 0, "auto": 0, "fixed": 0, "accepted": 1}


def test_removing_an_unknown_frame_changes_nothing(tmp_path):
    dataset = make_dataset(tmp_path / "labels")

    assert not dataset.remove_frame("IMG_9_000001")
    assert len(dataset.frames()) == 2 and dataset.removed() == set()


def test_frames_json_is_replaced_whole(tmp_path):
    dataset = make_dataset(tmp_path / "labels")
    dataset.add_frames([Frame("IMG_2_000010", "data/input/IMG_2.MOV", 10, 64, 48)])

    assert [f.name for f in dataset.frames()] == ["IMG_1_000015", "IMG_1_000045", "IMG_2_000010"]
    assert sorted(p.name for p in dataset.root.iterdir() if p.is_file()) == ["frames.json"]  # no temporary file left
