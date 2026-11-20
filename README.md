# pool-segmentation

Find a swimming pool's **water** in video frames, and make the training data for it. A small U-Net (ResNet18
encoder) that takes about 25 ms per frame on a MacBook, plus the tools to label your own pool's frames and train it
in minutes.

Built for swim-analytics (swimming video analysis), which uses it to drop people outside the pool and, later, to
find the lanes. Lane ropes as a second class are planned.

## Install

```bash
pip install pool-segmentation                     # the model
pip install "pool-segmentation[app,train]"   # and all tools
```

Extras: `app` the Labels page, `train` TensorBoard, `dev` the tests.

## Use a trained model

```python
from pool_segmentation import PoolModel, find_model

model = PoolModel.load(find_model())   # your newest training run in data/models/pool/, else the published model
classes = model.predict(frame)         # BGR frame -> class per pixel: 0 other, 1 water
water = model.water(frame)             # the water's outline: one polygon [(x, y), ...] per piece of water
```

The published model is on the Hugging Face Hub:
[nlateralus/pool-segmentation](https://huggingface.co/nlateralus/pool-segmentation) (MIT). Without a training run of
your own, `find_model()` downloads it the first time and keeps it in `~/.cache/huggingface`; when online, it gets the
newest published version. A given version: `find_model("hub:20261119-231345")` (versions are tagged with their
training run). Your own frames train it further for your pool (below).

## Make training data and train

Run these in the folder that holds your data; paths are relative to it (`data/labels/pool/`, `data/models/pool/`).

```bash
python -m pool_segmentation.labeling pick data/input/*.MOV   # 1. pick frames: one every 2 s, at most 25 per video
python -m pool_segmentation.labeling auto                     # 2. first labels (water) from the newest model
python -m pool_segmentation.app                               # 3. check and fix them: http://127.0.0.1:8001
python -m pool_segmentation.model train                       # 4. train on the checked frames (about 5 min, M4 Pro)
```

Steps 1 and 2 can also be done in the browser: the Labels page's **+ Add videos** (http://127.0.0.1:8001/add) lists
the videos in a folder (or takes video files you choose or drop on it), picks frames from the ones you tick at the
sample rate you choose, and lets your newest trained model label them (`--models`, `--videos` to point it at your
folders).

Without a trained model of your own, step 2 uses the published model (or skip it and draw the water and ropes on
the Labels page). About 100 checked frames are enough to start: 91 frames gave a validation water IoU of 0.996. After
that, each round's model labels the next frames' water, and you only fix it and draw the ropes. Details:
[docs/labeling.md](https://github.com/Neptune-Trojans/pool-segmentation/blob/main/docs/labeling.md) (classes, how the model's first labels are made, the Labels page) and
[docs/model.md](https://github.com/Neptune-Trojans/pool-segmentation/blob/main/docs/model.md) (network, training, results).

The Labels and Add videos pages can also be shown inside another FastAPI app, under a path of its own:

```python
from pool_segmentation.app.server import create_app

app.mount("/labels", create_app("data/labels/pool", home=("← Videos", "/"),
                                models_dir="data/models/pool", videos_dir="data/input"))
```

## Publish new weights (maintainers)

The published model on the Hugging Face Hub,
[nlateralus/pool-segmentation](https://huggingface.co/nlateralus/pool-segmentation), is what every install uses when
it has no training run of its own. To replace it with a newer one, in the repo folder:

1. **Train, and check the run.** Training prints the validation water IoU and the worst frames, and the run's `val/`
   pictures show where the model is wrong. Compare with the published version (its numbers are on the model page and
   in the "Published model" table of [docs/model.md](https://github.com/Neptune-Trojans/pool-segmentation/blob/main/docs/model.md)),
   and publish only a run that is at least as good.

   ```bash
   .venv/bin/python -m pool_segmentation.model train
   ```

2. **Put a Hugging Face token in `.env`.** It needs write access to `nlateralus`
   ([create one](https://huggingface.co/settings/tokens)). `.env` is git-ignored: never commit the token. Instead of
   `.env`, `HF_TOKEN` in the environment or `hf auth login` work too.

   ```
   HF_TOKEN=hf_...
   ```

3. **Publish.** The newest run in `data/models/pool/`, or a given one:

   ```bash
   .venv/bin/python -m pool_segmentation.model publish                    # the newest run
   .venv/bin/python -m pool_segmentation.model publish 20261119-231345    # a given run
   ```

   This uploads `model.safetensors` (the weights only), `config.json` and the model card as the repo's newest
   version, tagged with the run's name. The run's `val/` pictures and the training frames are never uploaded: they
   show real people.

4. **Record it.** Add a row to the "Published model" table in docs/model.md, and commit.

After that, every install gets the new weights the next time it loads the model while online: no new PyPI release is
needed. Installs that pinned a version (`find_model("hub:<run>")`) keep theirs. An app with training runs of its own
in its `data/models/pool/` keeps using its newest local run: copy the new run there too.

**Roll back** by publishing an earlier run again (`publish <run>`): it becomes the newest version again. Its run
folder must still be in `data/models/pool/`.

## Layout

```
pool_segmentation/
  __init__.py      # PoolModel, find_model, water_polygons
  polygons.py      # water mask -> outline polygons
  video.py         # read a video's frames
  model/           # the network: network.py, data.py, train.py, predict.py, runs.py, hub.py (python -m pool_segmentation.model)
  labeling/        # training data: frames, labels, the model's first labels, shapes (python -m pool_segmentation.labeling)
  app/             # the Labels and Add videos pages: server.py, add_videos.py + static/ (python -m pool_segmentation.app)
docs/              # model.md, labeling.md
tests/             # same layout as pool_segmentation/
```

## Development

```bash
git clone https://github.com/Neptune-Trojans/pool-segmentation.git && cd pool-segmentation
python -m venv .venv && .venv/bin/pip install -e ".[app,dev,train]"
.venv/bin/python -m pytest                    # all tests (the browser tests need Chrome)
.venv/bin/python -m pytest -m "not browser"   # without them
```

`data/` is git-ignored: the labeling dataset holds frames of real people, and models are large. Keep both local.

## License

MIT ([LICENSE](https://github.com/Neptune-Trojans/pool-segmentation/blob/main/LICENSE)).
