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

model = PoolModel.load(find_model())   # the newest training run in data/models/pool/ (or a path to a model.pt)
classes = model.predict(frame)         # BGR frame -> class per pixel: 0 other, 1 water
water = model.water(frame)             # the water's outline: one polygon [(x, y), ...] per piece of water
```

No trained weights are published yet: train your own (below).

## Make training data and train

Run these in the folder that holds your data; paths are relative to it (`data/labels/pool/`, `data/models/pool/`).

```bash
python -m pool_segmentation.labeling pick data/input/*.MOV   # 1. pick frames: one every 2 s, at most 25 per video
python -m pool_segmentation.labeling auto                     # 2. first labels (water) from your newest trained model
python -m pool_segmentation.app                               # 3. check and fix them: http://127.0.0.1:8001
python -m pool_segmentation.model train                       # 4. train on the checked frames (about 5 min, M4 Pro)
```

The first time there is no trained model: skip step 2 and draw the water and ropes on the Labels page. About 100
checked frames are enough to start: 91 frames gave a validation water IoU of 0.996. After that, each round's model
labels the next frames' water, and you only fix it and draw the ropes. Details:
[docs/labeling.md](https://github.com/Neptune-Trojans/pool-segmentation/blob/main/docs/labeling.md) (classes, how the model's first labels are made, the Labels page) and
[docs/model.md](https://github.com/Neptune-Trojans/pool-segmentation/blob/main/docs/model.md) (network, training, results).

The Labels page can also be shown inside another FastAPI app, under a path of its own:

```python
from pool_segmentation.app.server import create_app

app.mount("/labels", create_app("data/labels/pool", home=("← Videos", "/")))
```

## Layout

```
pool_segmentation/
  __init__.py      # PoolModel, find_model, water_polygons
  polygons.py      # water mask -> outline polygons
  video.py         # read a video's frames
  model/           # the network: network.py, data.py, train.py, predict.py, runs.py (python -m pool_segmentation.model)
  labeling/        # training data: frames, labels, the model's first labels, shapes (python -m pool_segmentation.labeling)
  app/             # the Labels page: server.py + static/ (python -m pool_segmentation.app)
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
