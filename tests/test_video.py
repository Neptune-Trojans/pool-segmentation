import pytest

from pool_segmentation.video import get_video_info, read_frames


def test_video_info_and_frames(tmp_path, make_video):
    video = make_video(tmp_path / "clip.mp4", frames=5)

    info = get_video_info(video)
    frames = list(read_frames(video))

    assert (info.fps, info.width, info.height, info.frame_count) == (30.0, 64, 48, 5)
    assert len(frames) == 5 and frames[0].shape == (48, 64, 3)


def test_a_missing_video_raises(tmp_path):
    with pytest.raises(OSError, match="Cannot open video"):
        get_video_info(tmp_path / "none.mp4")
    with pytest.raises(OSError, match="Cannot open video"):
        next(read_frames(tmp_path / "none.mp4"))
