"""Search for swimming pool videos and download them, for the labeling dataset (docs/labeling.md, "Videos from the
web"). Run in the repo folder:

    .venv/bin/pip install "yt-dlp[default]"      # once: the downloader (YouTube also needs Node.js or Deno installed)
    .venv/bin/python scripts/fetch_videos.py "swimming competition" "lap swimming pool" --max 10
    .venv/bin/python scripts/fetch_videos.py --list "swim meet 50m"     # only show what it would download

Searches YouTube for each term (no API key needed), keeps videos of a usable length, filmed landscape, and downloads
up to --max new ones to data/input/web/: the video only (no sound, so no ffmpeg needed), at most 1080p, H.264. Each
video's origin (link, title, channel, the search term) goes to sources.json in the same folder. Then open that folder
on the Add videos page, tick the videos filmed like your camera (from the deck, the water in view), and add them.

The videos belong to their owners, and YouTube's terms don't allow downloading: whether you may use them for training
depends on where you are and what you do with the model. They stay local (data/ is git-ignored).
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from itertools import zip_longest
from pathlib import Path

OUT_DIR = Path("data/input/web")
QUERIES = ["swimming competition pool", "swim meet 50m freestyle", "lap swimming pool", "swimming training lanes"]
# Titles with these words are skipped: views that don't look like a camera on the deck
EXCLUDE = ["underwater", "drone", "aerial", "animation", "cartoon", "minecraft", "roblox"]
MIN_SECONDS, MAX_SECONDS = 10, 15 * 60
FAILURES_IN_A_ROW = 5  # then stop: YouTube is likely blocking downloads for a while
# The video stream only, at most 1080p: H.264 first (OpenCV reads it everywhere), else any mp4
FORMAT = "bv*[height<=1080][vcodec^=avc1]/bv*[height<=1080][ext=mp4]/b[height<=1080][ext=mp4]"
PREFIX = "yt-"  # file names: yt-<YouTube id>.mp4; frame names then start with it too


@dataclass(frozen=True)
class Result:
    id: str
    url: str
    title: str
    duration: float | None
    channel: str
    query: str


# ---------------------------------------------------------------------------------------------------------------
# YouTube, through yt-dlp (imported only here: the rest, and the tests, run without it)

def search(query: str, count: int) -> list[Result]:
    """YouTube's first `count` results for `query` (titles and links only, nothing downloaded)."""
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(f"ytsearch{count}:{query}", download=False)
    return [Result(e["id"], e.get("url") or f"https://www.youtube.com/watch?v={e['id']}", e.get("title") or "",
                   e.get("duration"), e.get("channel") or e.get("uploader") or "", query)
            for e in info.get("entries") or [] if e and e.get("id")]


def download(result: Result, out: Path) -> dict | None:
    """Download a result's video to `out` (yt-<id>.mp4). None if it isn't filmed landscape or is a live stream
    (checked before downloading). Returns its details for sources.json."""
    import yt_dlp

    options = {
        "quiet": True, "no_warnings": True, "noprogress": True, "format": FORMAT, "noplaylist": True,
        "outtmpl": str(out / f"{PREFIX}%(id)s.%(ext)s"),
        "js_runtimes": {"node": {}, "deno": {}},  # YouTube's player checks need a JavaScript runtime
        "sleep_interval": 2, "max_sleep_interval": 6,  # a pause between videos, as a person would
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(result.url, download=False)  # the chosen format's size, before downloading
        width, height = info.get("width") or 0, info.get("height") or 0
        if info.get("is_live") or width <= height:
            return None
        info = ydl.process_ie_result(info, download=True)
        downloads = info.get("requested_downloads") or [{}]
        path = Path(downloads[0].get("filepath") or ydl.prepare_filename(info))
    return {
        "file": path.name, "id": result.id, "url": info.get("webpage_url") or result.url,
        "title": info.get("title") or result.title, "channel": info.get("channel") or info.get("uploader") or result.channel,
        "license": info.get("license"), "duration": info.get("duration"), "width": width, "height": height,
        "query": result.query, "source": "youtube", "downloaded": date.today().isoformat(),
    }


# ---------------------------------------------------------------------------------------------------------------
# Choosing and keeping track

def skip_reason(result: Result, have: set[str], min_seconds: float, max_seconds: float, exclude: list[str]) -> str | None:
    """Why a search result is not downloaded, or None."""
    if result.id in have:
        return "downloaded already"
    if "/shorts/" in result.url:
        return "a Short (filmed upright)"
    if result.duration is not None and result.duration < min_seconds:
        return f"too short ({result.duration:.0f} s)"
    if result.duration is not None and result.duration > max_seconds:
        return f"too long ({result.duration / 60:.0f} min)"
    word = next((w for w in exclude if w.lower() in result.title.lower()), None)
    return f"title has '{word}'" if word else None


def load_sources(out: Path) -> dict[str, dict]:
    """sources.json: {file name: where it came from}."""
    path = out / "sources.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_sources(out: Path, sources: dict[str, dict]) -> None:
    (out / "sources.json").write_text(json.dumps(dict(sorted(sources.items())), indent=1, ensure_ascii=False))


def interleave(lists: list[list[Result]]) -> list[Result]:
    """The first result of each search, then the second of each, ...: every search term gets its turn."""
    return [r for group in zip_longest(*lists) for r in group if r is not None]


def fetch(queries: list[str], out: Path, max_new: int, min_seconds: float = MIN_SECONDS,
          max_seconds: float = MAX_SECONDS, exclude: list[str] = EXCLUDE, list_only: bool = False,
          log=print) -> list[dict]:
    """Search each term, then download up to `max_new` new videos (or with `list_only`, show them). Returns the
    downloaded videos' details (or the results that would be tried)."""
    sources = load_sources(out)
    have = {entry["id"] for entry in sources.values()} | {p.stem.removeprefix(PREFIX) for p in out.glob(f"{PREFIX}*")}
    per_query = max(10, 3 * max_new // max(len(queries), 1) + 5)  # more than needed: some are skipped
    results, seen = [], set()
    for query in queries:
        found = search(query, per_query)
        log(f"Search '{query}': {len(found)} results")
        results.append(found)
    candidates = []
    for result in interleave(results):
        if result.id in seen:
            continue
        seen.add(result.id)
        reason = skip_reason(result, have, min_seconds, max_seconds, exclude)
        if reason:
            log(f"  skip: {result.title[:70]} ({reason})")
        else:
            candidates.append(result)

    if list_only:
        log(f"\nWould try these ({min(len(candidates), max_new)} of {len(candidates)}):")
        for result in candidates[:max_new]:
            log(f"  {duration(result.duration):>7}  {result.title[:70]}  ({result.channel})  {result.url}")
        return [vars(result) for result in candidates[:max_new]]

    out.mkdir(parents=True, exist_ok=True)
    added, failures = [], 0
    for result in candidates:
        if len(added) >= max_new:
            break
        log(f"Downloading {len(added) + 1}/{max_new}: {result.title[:70]} ({result.channel}, {duration(result.duration)})")
        try:
            entry = download(result, out)
        except Exception as error:  # private, age-restricted, removed, no fitting format...: the next one
            log(f"  skip: {str(error).splitlines()[0][:150]}")
            failures += 1
            if failures >= FAILURES_IN_A_ROW:
                log(f"{failures} downloads failed in a row: YouTube may be blocking downloads for now. Stopped; run "
                    "again later (what is downloaded stays, and is skipped next time).")
                break
            continue
        failures = 0
        if entry is None:
            log("  skip: not filmed landscape (or live)")
            continue
        sources[entry["file"]] = entry
        save_sources(out, sources)  # after every video: an interrupted run keeps its record
        size = (out / entry["file"]).stat().st_size / 1e6
        log(f"  saved {entry['file']} ({entry['width']}×{entry['height']}, {size:.0f} MB)")
        added.append(entry)
    return added


def duration(seconds: float | None) -> str:
    return "?" if seconds is None else f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="scripts/fetch_videos.py",
        description="Search YouTube for swimming pool videos and download them for the labeling dataset.")
    parser.add_argument("queries", nargs="*", help=f"Search terms (default: {', '.join(repr(q) for q in QUERIES)})")
    parser.add_argument("--max", type=int, default=10, help="New videos to download, at most (default 10)")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help=f"Folder for the videos (default: {OUT_DIR})")
    parser.add_argument("--min-seconds", type=float, default=MIN_SECONDS, help=f"Shortest video (default {MIN_SECONDS})")
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS, help=f"Longest video (default {MAX_SECONDS})")
    parser.add_argument("--exclude", action="append", default=[], metavar="WORD",
                        help=f"Also skip titles with this word (again for more; always skipped: {', '.join(EXCLUDE)})")
    parser.add_argument("--list", action="store_true", help="Only show what would be downloaded")
    args = parser.parse_args(argv)
    if args.max < 1:
        parser.error("--max must be 1 or more")
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        parser.exit(1, 'Needs yt-dlp: .venv/bin/pip install "yt-dlp[default]"\n')

    added = fetch(args.queries or QUERIES, args.out, args.max, args.min_seconds, args.max_seconds,
                  EXCLUDE + args.exclude, args.list)
    if not args.list:
        print(f"\n{len(added)} new videos in {args.out} (where they came from: {args.out / 'sources.json'}). "
              "Next: open the folder on the Add videos page and tick the ones filmed like your camera.")


if __name__ == "__main__":
    sys.exit(main())
