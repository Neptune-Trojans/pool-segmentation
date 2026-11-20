"""Browser tests of the Labels page: drag a rope corner, save, add a rope and water by clicking dots. Also: the page
inside another app, under a path of its own, and the Add videos page: add a video, check its new frames, remove one;
choose a video file (copied to the videos folder) or open one by its path.

Drives headless Chrome (DevTools protocol) against the page. Skipped without Chrome (or set CHROME_PATH). Leave it
out with: pytest -m "not browser"
"""

import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import FastAPI

from pool_segmentation.app.server import create_app
from pool_segmentation.labeling.dataset import Dataset, Frame
from pool_segmentation.labeling.labels import FrameLabels

websocket = pytest.importorskip("websocket", reason="needs websocket-client")
uvicorn = pytest.importorskip("uvicorn")

pytestmark = pytest.mark.browser

WIDTH, HEIGHT = 640, 360
LABEL_A, LABEL_B = "test_000010", "test_000020"


def write_test_labels(dataset: Dataset) -> None:
    """Two frames to label: LABEL_A with the model's labels (one water area, one rope), LABEL_B without."""
    frames = [Frame(name, "data/input/test.MOV", i, WIDTH, HEIGHT) for i, name in ((10, LABEL_A), (20, LABEL_B))]
    dataset.add_frames(frames)
    dataset.images.mkdir(parents=True)
    for f in frames:
        cv2.imwrite(str(dataset.image_path(f.name)), np.full((HEIGHT, WIDTH, 3), 180, np.uint8))
    dataset.save(FrameLabels(video=frames[0].video, frame_idx=10, image=frames[0].image, width=WIDTH, height=HEIGHT,
                             water=[[(20, 100), (620, 100), (620, 340), (20, 340)]],
                             ropes=[[(40, 196), (600, 178), (600, 182), (40, 204)]], source="test"))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def find_chrome() -> str | None:
    candidates = [
        os.environ.get("CHROME_PATH"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ]
    return next((c for c in candidates if c and Path(c).exists()), None)


def serve(app) -> tuple[str, uvicorn.Server, threading.Thread]:
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}", server, thread


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """The Labels page on its own, and inside another app at /labels: (its url, the other app's url, labels root)."""
    root = tmp_path_factory.mktemp("labels")
    write_test_labels(Dataset(root / "labels"))
    write_test_labels(Dataset(root / "mounted"))
    parent = FastAPI()
    parent.mount("/labels", create_app(root / "mounted", home=("← Videos", "/")))

    servers = [serve(create_app(root / "labels")), serve(parent)]
    yield servers[0][0], servers[1][0], root
    for _, app_server, thread in servers:
        app_server.should_exit = True
        thread.join(timeout=5)


def write_video(path: Path, frames: int = 120) -> Path:
    """A 4 s video (30 fps) whose frames get brighter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (WIDTH, HEIGHT))
    for i in range(frames):
        writer.write(np.full((HEIGHT, WIDTH, 3), 60 + i, np.uint8))
    writer.release()
    return path


@pytest.fixture(scope="module")
def add_server(tmp_path_factory):
    """The pages for an empty dataset, with a 4 s video (clip.mp4) to add and no trained model: (url, labels root)."""
    root = tmp_path_factory.mktemp("adding")
    write_video(root / "videos" / "clip.mp4")
    url, app_server, thread = serve(create_app(root / "labels", models_dir=root / "models", videos_dir=root / "videos"))
    yield url, root / "labels"
    app_server.should_exit = True
    thread.join(timeout=5)


class Chrome:
    """Minimal DevTools protocol client: send commands, keep the page's exceptions."""

    def __init__(self, ws):
        self.ws = ws
        self.last_id = 0
        self.exceptions = []

    def cdp(self, method: str, **params):
        self.last_id += 1
        self.ws.send(json.dumps({"id": self.last_id, "method": method, "params": params}))
        while True:
            message = json.loads(self.ws.recv())
            if message.get("method") == "Runtime.exceptionThrown":
                details = message["params"]["exceptionDetails"]
                self.exceptions.append(details.get("exception", {}).get("description") or details.get("text"))
            if message.get("id") == self.last_id:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message["result"]

    def js(self, expression: str):
        result = self.cdp("Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result["result"].get("value")

    def wait_for(self, expression: str, done, timeout: float = 15):
        """Evaluate `expression` until `done(value)`; return the last value."""
        deadline = time.time() + timeout
        value = self.js(expression)
        while not done(value) and time.time() < deadline:
            time.sleep(0.05)
            value = self.js(expression)
        return value

    def key(self, key: str, code: int, modifiers: int = 0) -> None:
        """Press a key; `modifiers`: 8 = Shift (DevTools protocol bit flags)."""
        for event in ("rawKeyDown", "keyUp"):
            self.cdp("Input.dispatchKeyEvent", type=event, key=key, code=key, windowsVirtualKeyCode=code,
                     modifiers=modifiers)


@pytest.fixture(scope="module")
def chrome_session(tmp_path_factory):
    path = find_chrome()
    if path is None:
        pytest.skip("Chrome not found (set CHROME_PATH)")
    profile = tmp_path_factory.mktemp("chrome-profile")
    process = subprocess.Popen(
        [path, "--headless=new", "--remote-debugging-port=0", f"--user-data-dir={profile}", "--window-size=1440,900",
         "--mute-audio", "--hide-scrollbars", "--no-first-run", "--no-default-browser-check", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        # With port 0 Chrome picks a free port and writes it to DevToolsActivePort
        port_file = profile / "DevToolsActivePort"
        deadline = time.time() + 20
        while not (port_file.exists() and port_file.read_text().strip()):
            assert time.time() < deadline, "Chrome did not start"
            time.sleep(0.1)
        port = port_file.read_text().split()[0]
        targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
        page = next(t for t in targets if t["type"] == "page")
        ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=30, suppress_origin=True)
        browser = Chrome(ws)
        browser.cdp("Runtime.enable")
        browser.cdp("Page.enable")
        browser.cdp("Emulation.setDeviceMetricsOverride", width=1440, height=900, deviceScaleFactor=1, mobile=False)
        yield browser
        ws.close()
    finally:
        process.terminate()
        process.wait(timeout=10)


@pytest.fixture
def chrome(chrome_session):
    chrome_session.exceptions.clear()
    yield chrome_session
    assert chrome_session.exceptions == [], "exceptions in the page"


def open_frame(chrome: Chrome, url: str, name: str) -> None:
    chrome.cdp("Page.navigate", url=url)
    assert chrome.wait_for("window.__labels?.frame() ?? null", lambda shown: shown == name) == name


def test_drag_save_and_add_a_rope(chrome, server):
    js = chrome.js
    base_url, _, root = server
    dataset = Dataset(root / "labels")
    open_frame(chrome, f"{base_url}/?frame={LABEL_A}", LABEL_A)
    assert js("document.getElementById('home').hidden")  # on its own: no link back

    def screen(x, y):  # image point -> viewport point
        return js(f"""(() => {{ const r = document.getElementById('canvas').getBoundingClientRect();
          const [x, y] = __labels.toScreen({x}, {y}); return [r.left + x, r.top + y]; }})()""")

    def mouse(kind, point):
        chrome.cdp("Input.dispatchMouseEvent", type=kind, x=point[0], y=point[1], button="left",
                   buttons=1 if kind != "mouseReleased" else 0, clickCount=1)

    def click(point):
        mouse("mousePressed", point)
        mouse("mouseReleased", point)

    def hover(point):  # move the mouse without a button; the tooltip's text, or None if hidden
        chrome.cdp("Input.dispatchMouseEvent", type="mouseMoved", x=point[0], y=point[1], buttons=0)
        return js("(() => { const t = document.getElementById('class-tip'); return t.hidden ? null : t.textContent; })()")

    # Class tooltip: the class under the mouse, as in the training mask (rope over water over other)
    assert hover(screen(300, 300)) == "Water"
    assert hover(screen(320, 190)) == "Lane rope 1"  # inside the rope's outline
    assert hover(screen(10, 50)) == "Other"  # outside the pool
    assert hover(screen(40, 196)) == "Lane rope 1 · drag to move this corner"

    # Drag the rope's first corner 40 px down (in frame pixels)
    start, target = screen(40, 196), screen(40, 236)
    mouse("mousePressed", start)
    for k in range(1, 6):
        mouse("mouseMoved", [start[0] + (target[0] - start[0]) * k / 5, start[1] + (target[1] - start[1]) * k / 5])
    mouse("mouseReleased", target)
    x, y = js("__labels.shapes().ropes[0][0]")
    assert abs(x - 40) < 1.5 and abs(y - 236) < 1.5
    assert js("!document.getElementById('unsaved').hidden")

    chrome.key("s", 83)
    assert chrome.wait_for("document.getElementById('message').textContent", lambda t: t.startswith("Saved")) == "Saved as fixed."
    saved = dataset.labels(LABEL_A)
    assert saved.status == "fixed" and abs(saved.ropes[0][0][1] - 236) < 1.5

    # + Rope: click dots around it; each dot connects to the last; a click on the first dot closes it
    chrome.key("r", 82)
    assert js("document.getElementById('add-rope').classList.contains('active')")  # the button shows the tool is on
    dots = [(100, 290), (500, 280), (500, 300), (100, 310)]
    click(screen(*dots[0]))
    click(screen(*dots[0]))  # the same dot twice (e.g. a double-click's first click): one dot
    click(screen(300, 330))
    chrome.key("Backspace", 8)  # removes the last dot
    origin = screen(0, 0)  # a drag moves the frame, and adds no dot
    start = screen(300, 200)
    mouse("mousePressed", start)
    for k in range(1, 6):
        mouse("mouseMoved", [start[0] + 12 * k, start[1] + 6 * k])
    mouse("mouseReleased", [start[0] + 60, start[1] + 30])
    assert np.allclose(screen(0, 0), [origin[0] + 60, origin[1] + 30], atol=0.5)
    for dot in dots[1:]:
        click(screen(*dot))
    assert len(js("__labels.shapes().ropes")) == 1  # not added before it is closed
    click(screen(*dots[0]))
    ropes = js("__labels.shapes().ropes")
    assert len(ropes) == 2 and not js("document.getElementById('add-rope').classList.contains('active')")
    assert np.allclose(ropes[1], dots, atol=1.5)
    assert hover(screen(300, 295)) == "Lane rope 2"

    # Next frame (no labels yet): a rope with dots beyond the frame's edge (they land on it), Enter to finish
    chrome.key("ArrowRight", 39)  # saves frame A's changes first
    assert chrome.wait_for("window.__labels.frame()", lambda name: name == LABEL_B) == LABEL_B
    assert len(dataset.labels(LABEL_A).ropes) == 2
    chrome.key("r", 82)
    for dot in ((50, 150), (643, 155), (643, 165), (50, 160)):
        click(screen(*dot))
    chrome.key("Enter", 13)
    assert js("__labels.shapes().ropes.length") == 1
    chrome.key("a", 65)
    chrome.wait_for("document.getElementById('message').textContent", lambda t: t.startswith("Saved"))
    [rope] = dataset.labels(LABEL_B).ropes
    assert np.allclose(rope, [(50, 150), (640, 155), (640, 165), (50, 160)], atol=1.5)
    assert rope[1][0] == rope[2][0] == 640

    # + Water on frame B (no water yet): W, click 4 corners, then save without finishing the area: what is on screen
    # is saved (the area is finished first, and the save waits for the server's merge)
    open_frame(chrome, f"{base_url}/?frame={LABEL_B}", LABEL_B)
    chrome.key("w", 87)
    for point in (screen(100, 120), screen(500, 120), screen(500, 300), screen(100, 300)):
        click(point)
    chrome.key("s", 83)
    chrome.wait_for("document.getElementById('message').textContent", lambda t: t.startswith("Saved"))
    [water] = dataset.labels(LABEL_B).water
    xs, ys = zip(*water)
    assert len(water) == 4 and abs(min(xs) - 100) < 2 and abs(max(ys) - 300) < 2
    assert js("__labels.shapes().water.length") == 1 and not js("document.getElementById('add-water').classList.contains('active')")


def test_inside_another_app_the_page_works_and_links_back(chrome, server):
    js = chrome.js
    _, other_url, root = server

    open_frame(chrome, f"{other_url}/labels/?frame={LABEL_B}", LABEL_B)

    home = js("(() => { const a = document.getElementById('home'); return [a.hidden, a.textContent, a.href]; })()")
    assert home == [False, "← Videos", f"{other_url}/"]
    assert js("document.querySelectorAll('#frame-list .frame-item').length") == 2
    chrome.key("s", 83)  # saving goes to the other app's /labels/api/...
    assert chrome.wait_for("document.getElementById('message').textContent", lambda t: t.startswith("Saved")).startswith("Saved")
    assert Dataset(root / "mounted").labels(LABEL_B).status == "fixed"


def test_add_a_video_then_check_and_remove_its_frames(chrome, add_server):
    js = chrome.js
    url, labels_root = add_server
    dataset = Dataset(labels_root)
    text = lambda id_: js(f"document.getElementById('{id_}').textContent")

    # The empty Labels page points to the Add videos page
    chrome.cdp("Page.navigate", url=f"{url}/")
    assert chrome.wait_for("!document.getElementById('no-dataset').hidden", bool)
    assert js("document.querySelector('#no-dataset a[href=\"add\"]') !== null")

    chrome.cdp("Page.navigate", url=f"{url}/add")
    assert chrome.wait_for("window.__add?.listing()?.videos.length ?? 0", lambda n: n == 1) == 1
    assert js("document.querySelector('.video-file').textContent") == "clip.mp4"
    assert js("document.querySelector('.new-frames').textContent") == "+2 new frames"  # one every 2 s: 2 of 4 s
    assert js("[...document.getElementById('model').options].map((o) => o.textContent)") == [
        "Published model (Hugging Face)", "None: draw them by hand"]  # no trained run here: the published model first
    assert "No trained model" in text("model-note") and "the published model labels the frames" in text("model-note")
    js("document.getElementById('model').value = ''")  # None: no download in the test
    assert js("document.getElementById('add').disabled")  # nothing selected

    # One frame every second: the new frames are counted again
    js("(() => { const e = document.getElementById('every'); e.value = '1'; e.dispatchEvent(new Event('input')); })()")
    assert chrome.wait_for("document.querySelector('.new-frames').textContent", lambda t: t == "+4 new frames") == "+4 new frames"
    js("document.querySelector('.video-label').click()")  # a click anywhere on the row selects the video
    assert text("add-summary") == "4 new frames from 1 video" and not js("document.getElementById('add').disabled")

    js("document.getElementById('add').click()")
    job = chrome.wait_for("window.__add.job()", lambda j: j is not None and j["state"] != "running")
    assert job["state"] == "done" and job["added"] == 4
    assert text("job-text") == "Done: 4 frames added, without labels: draw them on the Labels page."
    assert chrome.wait_for("document.querySelector('.new-frames').textContent", lambda t: t == "nothing new") == "nothing new"
    assert js("document.querySelector('.pill').textContent") == "4 frames in the dataset"
    assert [f.name for f in dataset.frames()] == ["clip_000015", "clip_000045", "clip_000075", "clip_000105"]

    # Check the new frames: the Labels page opens the first, showing the frames not checked
    js("document.getElementById('check').click()")
    assert chrome.wait_for("window.__labels?.frame() ?? null", lambda name: name == "clip_000015") == "clip_000015"
    assert js("document.getElementById('filter').value") == "todo"
    assert js("location.search") == "?frame=clip_000015&filter=todo"

    # Remove it: the next frame opens, and the frame is gone from the dataset
    js("window.confirm = () => true")
    js("document.getElementById('remove').click()")
    assert chrome.wait_for("window.__labels.frame()", lambda name: name == "clip_000045") == "clip_000045"
    assert text("message") == "Removed clip_000015."
    assert js("[...document.querySelectorAll('.frame-item-name')].map((e) => e.textContent)") == [
        "clip_000045", "clip_000075", "clip_000105"]
    assert [f.name for f in dataset.frames()] == ["clip_000045", "clip_000075", "clip_000105"]
    assert dataset.removed() == {"clip_000015"}

    # Shift+Backspace (Shift+⌫ on a Mac) removes the frame too, after asking; answered "no", nothing happens
    js("window.confirm = () => false")
    chrome.key("Backspace", 8, modifiers=8)
    assert js("window.__labels.frame()") == "clip_000045" and len(dataset.frames()) == 3
    js("window.confirm = () => true")
    chrome.key("Backspace", 8, modifiers=8)
    assert chrome.wait_for("window.__labels.frame()", lambda name: name == "clip_000075") == "clip_000075"
    assert [f.name for f in dataset.frames()] == ["clip_000075", "clip_000105"]
    assert dataset.removed() == {"clip_000015", "clip_000045"}


def test_choose_a_video_file_or_open_one_by_its_path(chrome, add_server, tmp_path):
    js = chrome.js
    url, labels_root = add_server
    videos = labels_root.parent / "videos"
    source = write_video(tmp_path / "phone" / "swim.mp4")
    ticked = lambda: js("[...document.querySelectorAll('.video-row.selected .video-file')].map((e) => e.textContent)")

    chrome.cdp("Page.navigate", url=f"{url}/add")
    chrome.wait_for("window.__add?.listing() ?? null", lambda listing: listing is not None)
    assert js("document.getElementById('upload-folder').textContent") == str(videos.resolve())

    # Choose files…: the file is copied to the videos folder, which opens with it ticked
    document = chrome.cdp("DOM.getDocument")
    node = chrome.cdp("DOM.querySelector", nodeId=document["root"]["nodeId"], selector="#file-input")
    chrome.cdp("DOM.setFileInputFiles", files=[str(source)], nodeId=node["nodeId"])
    chrome.wait_for("window.__add.listing().videos.map((v) => v.file)", lambda files: "swim.mp4" in files)
    assert (videos / "swim.mp4").read_bytes() == source.read_bytes()
    assert chrome.wait_for("document.getElementById('upload').hidden", bool)
    assert ticked() == ["swim.mp4"]
    assert "new frames from 1 video" in js("document.getElementById('add-summary').textContent")

    # The same file again (unticked meanwhile): not copied again (same name and size), ticked again
    before = (videos / "swim.mp4").stat().st_mtime_ns
    js("document.querySelector('.video-row.selected .video-label').click()")
    assert ticked() == []
    chrome.cdp("DOM.setFileInputFiles", files=[str(source)], nodeId=node["nodeId"])
    assert chrome.wait_for("[...document.querySelectorAll('.video-row.selected .video-file')].map((e) => e.textContent)",
                           lambda files: files == ["swim.mp4"]) == ["swim.mp4"]
    assert (videos / "swim.mp4").stat().st_mtime_ns == before
    assert js("document.getElementById('error').hidden")

    # A video by its path in the path field: its folder opens with it ticked too
    js(f"""(() => {{ document.getElementById('folder-path').value = {json.dumps(str(videos / 'clip.mp4'))};
      document.getElementById('folder-form').requestSubmit(); }})()""")
    assert chrome.wait_for("[...document.querySelectorAll('.video-row.selected .video-file')].map((e) => e.textContent)",
                           lambda files: files == ["clip.mp4", "swim.mp4"]) == ["clip.mp4", "swim.mp4"]
    assert js("document.getElementById('folder-path').value") == str(videos.resolve())
