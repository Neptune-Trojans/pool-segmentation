"""Add videos page API: the videos in a folder with their frames in the dataset (or a video's folder), video files
copied to the videos folder, adding them in the background (pick frames, then the first labels from a trained model;
a stand-in model here), cancel. Also removing a frame."""

import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pool_segmentation.app.server import create_app
from pool_segmentation.labeling.dataset import Dataset

RUNS = ("20261116-223804", "20261116-224516")


class FakeLabeler:
    """Stands in for a trained model: the lower half of every frame is water. `gate`: wait for it at every frame."""

    def __init__(self, name="pool-unet test", gate: threading.Event | None = None):
        self.name = name
        self.gate = gate
        self.labeling = threading.Event()  # set once it labels a frame (waiting at the gate)
        self.calls = 0

    def label(self, frame):
        self.labeling.set()
        if self.gate:
            assert self.gate.wait(10), "the test didn't open the gate"
        self.calls += 1
        h, w = frame.shape[:2]
        return [[(0, h / 2), (w - 1, h / 2), (w - 1, h - 1), (0, h - 1)]], []


@pytest.fixture
def setup(tmp_path, make_video, monkeypatch):
    """Videos IMG_1 and IMG_2 (2 s each) in videos/, two trained runs, an empty dataset. The model is a stand-in."""
    videos = tmp_path / "videos"
    make_video(videos / "IMG_1.mp4", frames=60)
    make_video(videos / "IMG_2.mp4", frames=60)
    (videos / "notes.txt").write_text("not a video")
    (videos / "pool B").mkdir()
    models = tmp_path / "models"
    for run in RUNS:
        (models / run).mkdir(parents=True)
        (models / run / "model.pt").write_bytes(b"")
    labelers = []

    def create_labeler(model, models_dir):
        assert models_dir == models
        labelers.append(FakeLabeler(f"pool-unet {model}"))
        return labelers[-1]

    monkeypatch.setattr("pool_segmentation.labeling.commands.create_labeler", create_labeler)
    dataset = Dataset(tmp_path / "labels")
    client = TestClient(create_app(dataset.root, models_dir=models, videos_dir=videos), base_url="http://127.0.0.1")
    return client, dataset, videos, labelers


def wait_for_job(client) -> dict:
    deadline = time.time() + 15
    while (job := client.get("/api/add/job").json()["job"])["state"] == "running":
        assert time.time() < deadline, "the job did not finish"
        time.sleep(0.02)
    return job


def test_options_name_the_start_folder_and_the_models_newest_first(setup):
    client, dataset, videos, _ = setup

    options = client.get("/api/add").json()

    assert options["folder"] == str(videos.resolve())
    assert options["models"] == ["20261116-224516", "20261116-223804"]
    assert options["defaults"] == {"every_seconds": 2.0, "max_per_video": 25}
    assert options["dataset"] == str(dataset.root) and options["job"] is None and options["home"] is None


def test_folder_lists_videos_with_their_frames_in_the_dataset_and_the_new_ones(setup):
    client, _, videos, _ = setup
    client.post("/api/add", json={"videos": [str(videos / "IMG_1.mp4")], "every_seconds": 1, "model": None})
    wait_for_job(client)

    listing = client.get("/api/add/folder", params={"every_seconds": 0.5, "max_per_video": 25}).json()

    assert listing["folder"] == str(videos.resolve()) and listing["parent"] == str(videos.resolve().parent)
    assert listing["folders"] == [{"name": "pool B", "path": str(videos.resolve() / "pool B")}]
    assert [(v["file"], v["in_dataset"], v["new_frames"]) for v in listing["videos"]] == [
        ("IMG_1.mp4", 2, 4),  # 4 frames at 0.5 s, none of them the 2 picked at 1 s
        ("IMG_2.mp4", 0, 4)]
    assert listing["videos"][0]["info"] == {"fps": 30.0, "width": 64, "height": 48, "frame_count": 60}
    at_one_second = client.get("/api/add/folder", params={"path": str(videos), "every_seconds": 1}).json()
    assert [v["new_frames"] for v in at_one_second["videos"]] == [0, 2]


def test_a_video_path_opens_its_folder_with_the_video_selected(setup):
    client, _, videos, _ = setup

    listing = client.get("/api/add/folder", params={"path": str(videos / "IMG_2.mp4")}).json()

    assert listing["folder"] == str(videos.resolve())
    assert listing["selected"] == str((videos / "IMG_2.mp4").resolve())
    assert [v["file"] for v in listing["videos"]] == ["IMG_1.mp4", "IMG_2.mp4"]
    assert client.get("/api/add/folder").json()["selected"] is None  # a folder: nothing selected
    assert client.get("/api/add/folder", params={"path": str(videos / "notes.txt")}).status_code == 404


def test_a_video_file_is_copied_to_the_videos_folder(tmp_path, make_video):
    source = make_video(tmp_path / "phone" / "IMG_9.mp4", frames=60)
    videos = tmp_path / "data" / "input"  # made on the first copy
    client = TestClient(create_app(tmp_path / "labels", models_dir=tmp_path / "none", videos_dir=videos),
                        base_url="http://127.0.0.1")
    assert client.get("/api/add").json()["upload_folder"] == str(videos.resolve())

    copied = client.put("/api/add/upload", params={"name": "IMG_9.mp4"}, content=source.read_bytes())

    assert copied.status_code == 200
    assert copied.json() == {"path": str(videos.resolve() / "IMG_9.mp4"), "size": source.stat().st_size}
    assert (videos / "IMG_9.mp4").read_bytes() == source.read_bytes()
    assert [p.name for p in videos.iterdir()] == ["IMG_9.mp4"]  # no partial file left
    listing = client.get("/api/add/folder", params={"path": copied.json()["path"], "every_seconds": 1}).json()
    assert [(v["file"], v["new_frames"]) for v in listing["videos"]] == [("IMG_9.mp4", 2)]


def test_a_copy_never_replaces_a_file(setup):
    client, _, videos, _ = setup
    before = (videos / "IMG_1.mp4").read_bytes()

    response = client.put("/api/add/upload", params={"name": "IMG_1.mp4"}, content=b"another video")

    assert response.status_code == 409 and "exists already" in response.json()["detail"]
    assert (videos / "IMG_1.mp4").read_bytes() == before


@pytest.mark.parametrize("name, content", [
    ("../IMG_9.mp4", b"video"),
    ("pool B/IMG_9.mp4", b"video"),
    ("..\\IMG_9.mp4", b"video"),
    (".IMG_9.mp4", b"video"),
    ("notes.txt", b"video"),
    ("IMG_9", b"video"),
    ("IMG_9.mp4", b""),  # empty
])
def test_a_copy_needs_a_plain_video_file_name_and_content(setup, name, content):
    client, _, videos, _ = setup
    before = sorted(p.relative_to(videos.parent) for p in videos.parent.rglob("*"))

    assert client.put("/api/add/upload", params={"name": name}, content=content).status_code == 400
    assert sorted(p.relative_to(videos.parent) for p in videos.parent.rglob("*")) == before  # nothing written


@pytest.mark.parametrize("params, status", [
    ({"path": "/no/such/folder"}, 404),
    ({"every_seconds": 0}, 400),
    ({"max_per_video": 0}, 400),
])
def test_folder_refuses_bad_requests(setup, params, status):
    client, *_ = setup
    assert client.get("/api/add/folder", params=params).status_code == status


def test_thumbnail_of_a_video(setup):
    client, _, videos, _ = setup

    thumbnail = client.get("/api/add/thumbnail", params={"path": str(videos / "IMG_1.mp4")})

    assert thumbnail.status_code == 200 and thumbnail.headers["content-type"] == "image/jpeg"
    assert client.get("/api/add/thumbnail", params={"path": str(videos / "notes.txt")}).status_code == 404
    assert client.get("/api/add/thumbnail", params={"path": "IMG_1.mp4"}).status_code == 404  # absolute paths only


def test_add_picks_frames_and_the_newest_model_labels_them(setup):
    client, dataset, videos, labelers = setup

    started = client.post("/api/add", json={"videos": [str(videos / "IMG_1.mp4"), str(videos / "IMG_2.mp4")],
                                            "every_seconds": 1, "max_per_video": 25})
    assert started.status_code == 200 and started.json()["job"]["state"] == "running"
    job = wait_for_job(client)

    assert (job["state"], job["added"], job["labeled"], job["first_added"]) == ("done", 4, 4, "IMG_1_000015")
    assert (job["model"], job["videos_done"], job["videos_total"]) == ("20261116-224516", 2, 2)  # "latest": the newest
    assert [f.name for f in dataset.frames()] == ["IMG_1_000015", "IMG_1_000045", "IMG_2_000015", "IMG_2_000045"]
    assert dataset.frame("IMG_2_000045").video == str(videos / "IMG_2.mp4")  # outside the current folder: absolute
    labels = dataset.labels("IMG_2_000045")
    assert (labels.status, labels.source, labels.water) == ("auto", "pool-unet 20261116-224516",
                                                           [[(0, 24), (63, 24), (63, 47), (0, 47)]])
    assert [labeler.calls for labeler in labelers] == [4]
    assert client.get("/api/add").json()["job"]["state"] == "done"  # the page finds the last job when it opens


def test_the_video_path_is_recorded_relative_to_the_current_folder_when_inside_it(setup, monkeypatch):
    client, dataset, videos, _ = setup
    monkeypatch.chdir(videos.parent)  # e.g. swim-analytics' root, its videos in data/input/

    client.post("/api/add", json={"videos": [str(videos / "IMG_1.mp4")], "every_seconds": 1, "model": None})
    wait_for_job(client)

    assert {f.video for f in dataset.frames()} == {"videos/IMG_1.mp4"}  # as `pick videos/IMG_1.mp4` records it


def test_add_with_another_model_or_none(setup):
    client, dataset, videos, labelers = setup

    client.post("/api/add", json={"videos": [str(videos / "IMG_1.mp4")], "every_seconds": 1, "model": RUNS[0]})
    assert wait_for_job(client)["model"] == RUNS[0]
    assert dataset.labels("IMG_1_000015").source == f"pool-unet {RUNS[0]}"

    client.post("/api/add", json={"videos": [str(videos / "IMG_2.mp4")], "every_seconds": 1, "model": None})
    job = wait_for_job(client)
    assert (job["state"], job["added"], job["labeled"]) == ("done", 2, 0)
    assert dataset.status_counts() == {"no labels": 2, "auto": 2, "fixed": 0, "accepted": 0}
    assert len(labelers) == 1  # no model loaded for None


def test_adding_a_video_again_adds_only_new_frames_and_labels_those_left_without(setup):
    client, dataset, videos, _ = setup
    client.post("/api/add", json={"videos": [str(videos / "IMG_1.mp4")], "every_seconds": 1, "model": None})
    wait_for_job(client)

    client.post("/api/add", json={"videos": [str(videos / "IMG_1.mp4")], "every_seconds": 0.5})
    job = wait_for_job(client)

    assert (job["added"], job["labeled"]) == (4, 4)
    assert dataset.status_counts()["no labels"] == 0  # the 2 frames added without labels before got labels too


@pytest.mark.parametrize("body, message", [
    ({"videos": []}, "No videos"),
    ({"videos": ["IMG_1.mp4"]}, "Not a video"),  # absolute paths only
    ({"videos": ["{videos}/notes.txt"]}, "Not a video"),
    ({"videos": ["{videos}/IMG_1.mp4", "{videos}/pool B/IMG_1.mp4"]}, "Several videos named IMG_1"),
    ({"videos": ["{videos}/IMG_1.mp4"], "model": "20990101-000000"}, "No trained model"),
])
def test_add_refuses_bad_requests(setup, make_video, body, message):
    client, dataset, videos, _ = setup
    make_video(videos / "pool B" / "IMG_1.mp4", frames=30)
    body["videos"] = [v.format(videos=videos) for v in body["videos"]]

    response = client.post("/api/add", json=body)

    assert response.status_code == 400 and message in response.json()["detail"]
    assert dataset.frames() == [] and client.get("/api/add/job").json()["job"] is None


@pytest.mark.parametrize("body", [{"every_seconds": 0}, {"max_per_video": 0}, {"max_per_video": 5000}])
def test_add_refuses_sample_rates_out_of_range(setup, body):
    client, _, videos, _ = setup
    assert client.post("/api/add", json={"videos": [str(videos / "IMG_1.mp4")], **body}).status_code == 422


def test_one_job_at_a_time_and_cancel_keeps_what_was_done(setup, monkeypatch):
    client, dataset, videos, _ = setup
    gate = threading.Event()
    slow = FakeLabeler(gate=gate)
    monkeypatch.setattr("pool_segmentation.labeling.commands.create_labeler", lambda model, models_dir: slow)
    client.post("/api/add", json={"videos": [str(videos / "IMG_1.mp4")], "every_seconds": 1})
    assert slow.labeling.wait(10)  # the job waits at the first frame's gate
    assert client.get("/api/add/job").json()["job"]["step"] == "labeling"

    busy = client.post("/api/add", json={"videos": [str(videos / "IMG_2.mp4")]})
    assert busy.status_code == 409
    assert client.post("/api/add/job/cancel").json()["job"]["cancel_requested"]
    gate.set()  # the first frame is labeled, then the job stops
    job = wait_for_job(client)

    assert (job["state"], job["added"], job["labeled"]) == ("cancelled", 2, 1)
    assert dataset.status_counts() == {"no labels": 1, "auto": 1, "fixed": 0, "accepted": 0}
    assert client.post("/api/add", json={"videos": [str(videos / "IMG_2.mp4")], "model": None}).status_code == 200
    assert wait_for_job(client)["state"] == "done"


def test_a_failing_job_says_why_and_keeps_the_frames(setup, monkeypatch):
    client, dataset, videos, _ = setup

    def broken(model, models_dir):
        raise RuntimeError("model.pt is damaged")

    monkeypatch.setattr("pool_segmentation.labeling.commands.create_labeler", broken)
    client.post("/api/add", json={"videos": [str(videos / "IMG_1.mp4")], "every_seconds": 1})
    job = wait_for_job(client)

    assert (job["state"], job["error"], job["added"], job["labeled"]) == ("failed", "model.pt is damaged", 2, 0)
    assert len(dataset.frames()) == 2


def test_remove_a_frame(setup):
    client, dataset, videos, _ = setup
    client.post("/api/add", json={"videos": [str(videos / "IMG_1.mp4")], "every_seconds": 1})
    wait_for_job(client)

    removed = client.delete("/api/labels/IMG_1_000015")

    assert removed.status_code == 200
    assert removed.json()["counts"] == {"no labels": 0, "auto": 1, "fixed": 0, "accepted": 0}
    assert [f["name"] for f in client.get("/api/labels").json()["frames"]] == ["IMG_1_000045"]
    assert not dataset.image_path("IMG_1_000015").exists() and dataset.labels("IMG_1_000015") is None
    assert client.delete("/api/labels/IMG_1_000015").status_code == 404
    listing = client.get("/api/add/folder", params={"every_seconds": 1}).json()
    assert [(v["in_dataset"], v["new_frames"]) for v in listing["videos"]] == [(1, 0), (0, 2)]  # not picked again


def test_add_page_uses_relative_links_and_works_under_another_app(setup):
    client, dataset, videos, _ = setup

    page = client.get("/add")
    assert page.status_code == 200
    assert 'href="static/labels.css"' in page.text and 'src="static/add.js"' in page.text
    assert 'href="/' not in page.text and 'src="/' not in page.text
    assert client.get("/static/add.js").status_code == 200
    assert 'href="add"' in client.get("/").text  # the Labels page links to it

    parent = FastAPI()
    parent.mount("/labels", create_app(dataset.root, home=("← Videos", "/"), videos_dir=videos))
    mounted = TestClient(parent, base_url="http://127.0.0.1")
    assert mounted.get("/labels/add").status_code == 200
    options = mounted.get("/labels/api/add").json()
    assert options["home"] == {"text": "← Videos", "url": "/"} and options["folder"] == str(videos.resolve())


def test_without_a_videos_folder_the_page_opens_the_current_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app(tmp_path / "labels", models_dir=tmp_path / "none", videos_dir=tmp_path / "missing"),
                        base_url="http://127.0.0.1")

    options = client.get("/api/add").json()

    assert options["folder"] == str(tmp_path.resolve()) and options["models"] == []
