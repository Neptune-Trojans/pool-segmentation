"""scripts/fetch_videos.py: which search results are downloaded, and the record of where each came from. The search
and the download are stand-ins: no test reaches YouTube, or needs yt-dlp."""

import json
import sys
import types

import pytest

from scripts import fetch_videos
from scripts.fetch_videos import EXCLUDE, Result, fetch, interleave, skip_reason


def result(id_, title="Swim meet, 50 m freestyle", duration=60, url=None, query="swim meet"):
    return Result(id_, url or f"https://www.youtube.com/watch?v={id_}", title, duration, "A channel", query)


@pytest.fixture
def youtube(monkeypatch):
    """Stand-ins for search() and download(): `results` per search term; download() writes a small file, returns
    None for the ids in `upright`, and fails for those in `failing`."""
    state = {"results": {}, "upright": set(), "failing": set(), "downloaded": [], "searched": []}

    def search(query, count):
        state["searched"].append((query, count))
        return state["results"].get(query, [])[:count]

    def download(found, out):
        if found.id in state["failing"]:
            raise RuntimeError("ERROR: [youtube] Video unavailable\nThis video is private")
        if found.id in state["upright"]:
            return None
        (out / f"yt-{found.id}.mp4").write_bytes(b"video")
        state["downloaded"].append(found.id)
        return {"file": f"yt-{found.id}.mp4", "id": found.id, "url": found.url, "title": found.title,
                "channel": found.channel, "license": None, "duration": found.duration, "width": 1920, "height": 1080,
                "query": found.query, "source": "youtube", "downloaded": "2026-11-20"}

    monkeypatch.setattr(fetch_videos, "search", search)
    monkeypatch.setattr(fetch_videos, "download", download)
    return state


def test_skip_reasons():
    skip = lambda r: skip_reason(r, {"have"}, 10, 900, EXCLUDE)

    assert skip(result("new")) is None
    assert skip(result("new", duration=None)) is None  # length unknown: tried
    assert skip(result("have")) == "downloaded already"
    assert skip(result("s1", url="https://www.youtube.com/shorts/s1")) == "a Short (filmed upright)"
    assert skip(result("new", duration=8)) == "too short (8 s)"
    assert skip(result("new", duration=1800)) == "too long (30 min)"
    assert skip(result("new", title="UNDERWATER swimming pool")) == "title has 'underwater'"


def test_every_search_term_gets_its_turn():
    a, b = [result(f"a{k}") for k in range(3)], [result("b0")]
    assert interleave([a, b]) == [a[0], b[0], a[1], a[2]]


def test_downloads_up_to_max_new_videos_taking_turns_between_search_terms(youtube, tmp_path):
    youtube["results"] = {"pool": [result("a", query="pool"), result("b", query="pool"), result("c", query="pool")],
                          "race": [result("d", query="race"), result("e", query="race")]}

    added = fetch(["pool", "race"], tmp_path / "web", 3, log=lambda line: None)

    assert youtube["downloaded"] == ["a", "d", "b"]
    assert [entry["file"] for entry in added] == ["yt-a.mp4", "yt-d.mp4", "yt-b.mp4"]
    sources = json.loads((tmp_path / "web" / "sources.json").read_text())
    assert sorted(sources) == ["yt-a.mp4", "yt-b.mp4", "yt-d.mp4"]
    assert (sources["yt-d.mp4"]["query"], sources["yt-d.mp4"]["url"]) == ("race", "https://www.youtube.com/watch?v=d")
    assert all(count >= 10 for _, count in youtube["searched"])  # more results than needed: some are skipped


def test_skips_known_shorts_excluded_failing_and_upright_videos(youtube, tmp_path):
    out = tmp_path / "web"
    out.mkdir()
    (out / "yt-old.mp4").write_bytes(b"video")  # downloaded before, e.g. without sources.json
    youtube["results"] = {"pool": [
        result("old"), result("s1", url="https://www.youtube.com/shorts/s1"), result("uw", title="Underwater pool tour"),
        result("private"), result("upright"), result("good"), result("good"),  # the same video twice: once
    ]}
    youtube["failing"], youtube["upright"] = {"private"}, {"upright"}
    lines = []

    added = fetch(["pool"], out, 5, log=lines.append)

    assert [entry["id"] for entry in added] == ["good"]
    assert sorted(p.name for p in out.iterdir()) == ["sources.json", "yt-good.mp4", "yt-old.mp4"]
    log = "\n".join(lines)
    for reason in ("downloaded already", "a Short", "title has 'underwater'", "Video unavailable", "not filmed landscape"):
        assert reason in log


def test_stops_after_five_failures_in_a_row(youtube, tmp_path):
    youtube["results"] = {"pool": [result(f"v{k}") for k in range(12)]}
    youtube["failing"] = {"v0", "v1", "v2", "v3", "v5", "v6", "v7", "v8", "v9"}  # v4 works: the count starts again
    lines = []

    added = fetch(["pool"], tmp_path / "web", 10, log=lines.append)

    assert [entry["id"] for entry in added] == ["v4"]
    assert "5 downloads failed in a row" in lines[-1] and youtube["downloaded"] == ["v4"]  # v10, v11 not tried


def test_a_second_run_downloads_only_new_videos_and_keeps_the_record(youtube, tmp_path):
    youtube["results"] = {"pool": [result(id_) for id_ in "abcd"]}
    fetch(["pool"], tmp_path / "web", 2, log=lambda line: None)

    added = fetch(["pool"], tmp_path / "web", 2, log=lambda line: None)

    assert [entry["id"] for entry in added] == ["c", "d"] and youtube["downloaded"] == list("abcd")
    assert sorted(json.loads((tmp_path / "web" / "sources.json").read_text())) == [f"yt-{x}.mp4" for x in "abcd"]


def test_list_only_downloads_nothing(youtube, tmp_path):
    youtube["results"] = {"pool": [result("a", duration=75), result("b")]}
    lines = []

    shown = fetch(["pool"], tmp_path / "web", 1, list_only=True, log=lines.append)

    assert [entry["id"] for entry in shown] == ["a"] and youtube["downloaded"] == []
    assert not (tmp_path / "web").exists()
    assert any("1:15" in line and "https://www.youtube.com/watch?v=a" in line for line in lines)


def test_main_passes_its_options(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setitem(sys.modules, "yt_dlp", types.ModuleType("yt_dlp"))  # installed (a stand-in)
    monkeypatch.setattr(fetch_videos, "fetch", lambda *args: calls.append(args) or [])

    fetch_videos.main(["swim meet", "lap swimming", "--max", "3", "--out", str(tmp_path), "--exclude", "kids", "--list"])
    fetch_videos.main([])

    assert calls[0] == (["swim meet", "lap swimming"], tmp_path, 3, 10, 900, EXCLUDE + ["kids"], True)
    assert calls[1][0] == fetch_videos.QUERIES and calls[1][1] == fetch_videos.OUT_DIR  # the defaults


def test_main_without_yt_dlp_says_how_to_install_it(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)  # import fails

    with pytest.raises(SystemExit) as stop:
        fetch_videos.main(["swim meet"])

    assert stop.value.code == 1 and 'pip install "yt-dlp[default]"' in capsys.readouterr().err


def test_max_must_be_at_least_one():
    with pytest.raises(SystemExit) as stop:
        fetch_videos.main(["--max", "0"])
    assert stop.value.code == 2
