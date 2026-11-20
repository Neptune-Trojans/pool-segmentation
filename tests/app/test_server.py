"""Labels page API: list the picked frames, their labels and image, save checked labels. Also the page itself, on its
own and inside another app."""

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pool_segmentation.app.server import create_app
from pool_segmentation.labeling.dataset import Dataset, Frame
from pool_segmentation.labeling.labels import FrameLabels

WATER = [[[0, 20], [99, 20], [99, 59], [0, 59]]]
ROPES = [[[0, 38], [99, 34], [99, 36], [0, 42]]]


@pytest.fixture
def setup(tmp_path):
    """Dataset with 2 frames: IMG_1_000010 labeled by the model, IMG_1_000040 not labeled yet."""
    dataset = Dataset(tmp_path / "labels")
    frames = [Frame(f"IMG_1_{i:06d}", "data/input/IMG_1.MOV", i, 100, 60) for i in (10, 40)]
    dataset.add_frames(frames)
    dataset.images.mkdir(parents=True)
    for f in frames:
        cv2.imwrite(str(dataset.image_path(f.name)), np.zeros((60, 100, 3), np.uint8))
    dataset.save(FrameLabels(video=frames[0].video, frame_idx=10, image=frames[0].image, width=100, height=60,
                             water=[[(0, 20), (99, 20), (99, 59), (0, 59)]], ropes=[[(0, 38), (99, 34), (99, 36), (0, 42)]],
                             source="pool-unet 20261116-224516"))
    client = TestClient(create_app(dataset.root), base_url="http://127.0.0.1")
    return client, dataset


def test_lists_frames_with_their_status(setup):
    client, _ = setup

    listing = client.get("/api/labels").json()

    assert [(f["name"], f["status"], f["water"], f["ropes"]) for f in listing["frames"]] == [
        ("IMG_1_000010", "auto", 1, 1), ("IMG_1_000040", "no labels", 0, 0)]
    assert listing["counts"] == {"no labels": 1, "auto": 1, "fixed": 0, "accepted": 0}
    assert listing["classes"] == {"0": "other", "1": "water", "2": "lane rope"}


def test_frame_labels_and_image(setup):
    client, _ = setup

    data = client.get("/api/labels/IMG_1_000010").json()

    assert data["frame"] == {"name": "IMG_1_000010", "video": "data/input/IMG_1.MOV", "frame_idx": 10, "width": 100, "height": 60}
    assert data["labels"]["water"] == WATER and data["labels"]["ropes"] == ROPES
    assert data["labels"]["status"] == "auto"
    unlabeled = client.get("/api/labels/IMG_1_000040").json()
    assert unlabeled["labels"]["water"] == [] and unlabeled["labels"]["source"] == ""
    image = client.get("/api/labels/IMG_1_000010/image")
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"


def test_save_fixed_labels(setup):
    client, dataset = setup
    ropes = [[[0, 39], [99, 32], [99, 34], [0, 43], [0, 41]]]

    response = client.put("/api/labels/IMG_1_000010", json={"water": WATER, "ropes": ropes})

    assert response.status_code == 200
    assert response.json()["counts"]["fixed"] == 1
    saved = dataset.labels("IMG_1_000010")
    assert (saved.status, saved.ropes, saved.source) == ("fixed", [[(0, 39), (99, 32), (99, 34), (0, 43), (0, 41)]],
                                                         "pool-unet 20261116-224516")


def test_the_status_comes_from_the_shapes(setup):
    client, dataset = setup
    status = lambda: dataset.labels("IMG_1_000010").status
    save = lambda ropes: client.put("/api/labels/IMG_1_000010", json={"water": WATER, "ropes": ropes}).json()

    assert save(ROPES)["labels"]["status"] == "accepted"  # the model's labels, unchanged: the model was right
    assert save([[[0.04, 38], [99, 34], [99, 36], [0, 42]]])["labels"]["status"] == "accepted"  # same to 0.1 pixel
    assert save([[[0, 40], [99, 34], [99, 36], [0, 42]]])["labels"]["status"] == "fixed"
    assert save(ROPES)["labels"]["status"] == "fixed" and status() == "fixed"  # once fixed, it stays fixed


def test_a_frame_without_model_labels_can_be_labeled_by_hand(setup):
    client, dataset = setup

    assert client.put("/api/labels/IMG_1_000040", json={"water": WATER, "ropes": []}).status_code == 200
    labels = dataset.labels("IMG_1_000040")
    assert (labels.status, labels.water) == ("fixed", [[(0, 20), (99, 20), (99, 59), (0, 59)]])


@pytest.mark.parametrize("body", [
    {"water": [[[0, 0], [5, 5]]], "ropes": []},
    {"water": WATER, "ropes": [[[0, 0], [9, 9]]]},
    {"water": WATER, "ropes": [{"points": [[0, 0], [9, 9]], "widths": [3, 3]}]},  # old format: files only
    {"water": "everywhere"},
])
def test_bad_labels_are_refused_and_nothing_changes(setup, body):
    client, dataset = setup
    before = dataset.labels("IMG_1_000010")

    assert client.put("/api/labels/IMG_1_000010", json=body).status_code in (400, 422)  # 422: not even the request's form
    assert dataset.labels("IMG_1_000010") == before


@pytest.mark.parametrize("request_", [("get", "/api/labels/nope"), ("get", "/api/labels/nope/image"),
                                      ("get", "/api/labels/..%2Fframes/image")])
def test_unknown_frames(setup, request_):
    client, _ = setup
    method, url = request_
    assert getattr(client, method)(url).status_code == 404


def test_empty_dataset_lists_nothing(tmp_path):
    client = TestClient(create_app(tmp_path / "none"), base_url="http://127.0.0.1")
    assert client.get("/api/labels").json()["frames"] == []
    assert "labels.js" in client.get("/").text


def test_water_edit_merges_without_saving(setup):
    client, dataset = setup
    notch = [[0, 20], [99, 20], [99, 59], [0, 59], [0, 45], [20, 45], [20, 35], [0, 35]]

    merged = client.post("/api/labels/IMG_1_000010/water", json={"water": [notch], "add": [[0, 30], [30, 30], [30, 50], [0, 50]]})
    assert merged.status_code == 200
    assert sorted(map(tuple, merged.json()["water"][0])) == [(0, 20), (0, 59), (99, 20), (99, 59)]

    assert dataset.labels("IMG_1_000010").water == [[(0, 20), (99, 20), (99, 59), (0, 59)]]  # nothing saved


def test_water_edit_refuses_bad_polygons(setup):
    client, _ = setup
    assert client.post("/api/labels/IMG_1_000010/water", json={"water": WATER, "add": [[0, 0], [5, 5]]}).status_code == 400
    assert client.post("/api/labels/nope/water", json={"water": WATER, "add": WATER[0]}).status_code == 404


def test_page_and_its_files_use_relative_links(setup):
    client, _ = setup

    page = client.get("/")

    assert page.status_code == 200
    assert 'href="static/labels.css"' in page.text and 'src="static/labels.js"' in page.text
    assert 'href="/' not in page.text and 'src="/' not in page.text  # absolute links would break under another app
    assert client.get("/static/labels.js").status_code == 200
    assert client.get("/static/labels.css").status_code == 200
    assert client.get("/api/labels").json()["home"] is None  # on its own: no link back


def test_the_browser_checks_for_newer_page_files(setup):
    client, _ = setup

    for url in ("/", "/static/labels.js", "/api/labels", "/api/labels/IMG_1_000010/image"):
        assert client.get(url).headers["cache-control"] == "no-cache", url
    assert client.get("/static/labels.js", headers={"if-none-match": client.get("/static/labels.js").headers["etag"]}
                      ).status_code == 304  # unchanged: not sent again


def test_another_app_can_show_the_page_under_its_own_path(setup):
    _, dataset = setup
    parent = FastAPI()
    parent.mount("/labels", create_app(dataset.root, home=("← Videos", "/")))
    client = TestClient(parent, base_url="http://127.0.0.1")

    assert client.get("/labels/").status_code == 200
    assert client.get("/labels/static/labels.js").status_code == 200
    listing = client.get("/labels/api/labels").json()
    assert listing["home"] == {"text": "← Videos", "url": "/"}
    assert [f["name"] for f in listing["frames"]] == ["IMG_1_000010", "IMG_1_000040"]
    redirect = client.get("/labels?frame=IMG_1_000040", follow_redirects=False)  # links made before it moved
    assert redirect.headers["location"].endswith("/labels/?frame=IMG_1_000040")
