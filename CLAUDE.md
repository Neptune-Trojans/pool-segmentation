# pool-segmentation

Find a swimming pool's water in video frames (a small U-Net), and the tools to make its training data: pick frames,
first labels from the newest trained model, the Labels page to check them. Used by swim-analytics (`../swim-analytics`), which installs
it.

## Rules

- **Commit and push only when the user asks for it.** Never commit or push on your own initiative; the user reviews
  changes first.
- **Every research request gets its own file in `research/`** (`research/<topic>.md`), even if the answer is also
  given in chat. Start it with `_Research date: YYYY-MM-DD_` and end it with a sources/references list with links.
  `research/` is git-ignored (local only).

## Environment

- Python venv in `.venv/`: run everything with `.venv/bin/python`
- Dependencies are in `pyproject.toml`: `.venv/bin/pip install -e ".[app,dev,train]"`
- swim-analytics uses this repo through an editable install in its own venv (`pip install -e ../pool-segmentation`):
  a change here is live there. After changing the public API (`pool_segmentation/__init__.py`, `PoolModel`,
  `find_model`, `create_app`), run swim-analytics' tests too

## Conventions

- Design and status: `docs/model.md` (network, training, results) and `docs/labeling.md` (classes, the model's first labels, the
  Labels page). Keep them up to date
- `pool_segmentation/model/`: the network (`python -m pool_segmentation.model train`); `labeling/`: training data
  (`python -m pool_segmentation.labeling`); `app/`: the Labels and Add videos pages (`python -m pool_segmentation.app`)
- Import torch only where a model is needed: `import pool_segmentation`, the labeling commands and the Labels page
  start without it (`PoolModel` is loaded on first use; `tests/test_package.py` checks this)
- The Labels and Add videos pages use relative links only (`static/...`, `api/...`): they run on their own at `/` and
  inside another app under a path (swim-analytics: `/labels/`). Plain HTML / CSS / JS (ES modules), no build step,
  no framework
- Paths default to `data/labels/pool/` and `data/models/pool/`, relative to the folder the command runs in
- `data/` stays local (git-ignored): the labeling dataset holds frames of real people

## Testing

- Unit tests: `.venv/bin/python -m pytest`; files in `tests/`, same folder structure as `pool_segmentation/`
- `tests/app/test_labels_browser.py` drives headless Chrome (skipped without it, ~3 s); leave it out with
  `-m "not browser"`. Run it after changing `pool_segmentation/app/static/`
- Tests that train use tiny frames, the CPU and `pretrained=False` (no weight download); tests never load a trained model (a stand-in or an untrained one)
- Write test outputs to a temp folder, never to `data/labels/` (checked labels are hand work) or `data/models/`
- Tests never reach the Hugging Face Hub: `tests/conftest.py` makes downloading and publishing fail as offline;
  tests of the published model put stand-ins in place (`tests/model/test_hub.py`)

## Releasing (PyPI: `pip install pool-segmentation`)

- `.github/workflows/tests.yml` runs the tests (not the browser ones) on every push to `main` and pull request
- A release: bump `version` in `pyproject.toml` (0.1.0 → 0.1.1 for fixes, 0.2.0 for new features), commit, then
  `git tag v<version> && git push origin v<version>`. `.github/workflows/release.yml` runs the tests, checks the tag
  matches the version, builds and publishes to PyPI (trusted publishing: no token in GitHub). Tag and push only when
  the user asks for a release
- Check the package locally first: `python -m build` and `twine check --strict dist/*`
- Model weights are published apart from the package, to the Hugging Face Hub (`nlateralus/pool-segmentation`,
  `docs/model.md`): `python -m pool_segmentation.model publish [run]`, with `HF_TOKEN` in `.env` (git-ignored; never
  print or commit it). Only when the user asks. Installs get new weights without a new release

## Workflow

- Work directly on `main`
