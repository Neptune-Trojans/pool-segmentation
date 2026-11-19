"""Browser tests of the Labels page: drag a rope corner, save, add a rope and water by clicking dots. Also: the page
inside another app, under a path of its own.

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

    def key(self, key: str, code: int) -> None:
        for event in ("rawKeyDown", "keyUp"):
            self.cdp("Input.dispatchKeyEvent", type=event, key=key, code=key, windowsVirtualKeyCode=code)


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
