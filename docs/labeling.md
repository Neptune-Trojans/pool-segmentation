# Labeling (training data for the pool segmentation model)

_Agreed: 2026-11-16 (in swim-analytics; moved to this repo 2026-11-18). Status: built (frame picking, first labels
from the trained model, Labels page, training masks); the first model is trained on 91 checked frames
(`docs/model.md`)._

Goal: a small, fast model that finds the pool's **water** and the **lane ropes** in every frame (swim-analytics uses
the lanes to join a swimmer's broken tracks). The loop: pick frames from new videos, let the newest trained model
label their water, check and fix the labels by hand (draw the ropes), train again. The first videos were filmed in two
pools; frames from other pools make the model work there too.

## Classes

| Number | Class | What it covers |
|---|---|---|
| 0 | other | everything outside the pool (deck, gutter, buildings, trees, sky, the sign, people on the deck) and any pixel that can't be decided |
| 1 | water | the pool and everything in it: water, splashes, foam, glare, reflections, the floor lines, and swimmers (cap, goggles and fins included) |
| 2 | lane rope | the floats of every rope, all colors |

When classes meet: lane rope > water > other. Not classes, on purpose: swimmers (they come from the detection
stage), floor lane lines, each rope on its own (separated afterwards by their straight lines), red vs blue rope
sections (from the rope colors along each rope).

## Labels are shapes, not pixels

- **Water**: polygons (the pool area, swimmers included). A sign or a person can split it into several. Polygons
  have no holes: everything inside the pool's outline counts as pool (a kickboard in the middle of the pool too).
- **Lane rope**: a polygon around each rope (its outline), usually 4 corners: a straight rope stays straight in the
  image, even in perspective, and its near end looks thicker than its far end. A swimmer crossing a rope doesn't cut
  it: one outline across. (Until 2026-11-16 a rope was a line with a width at each end; drawing ropes by clicking dots
  around them turned out simpler.)

Fixing shapes means dragging a few corners instead of painting pixels. Measured on 42 frames (2026-11-16): the pool area
as a polygon with about 50 corners (a corner wherever the mask's outline bends more than 8 px) matches the mask within
IoU 0.99. With notches at the frame's edge filled and corners along the frame's edge dropped, the 91 frames of the first
set have a median of 9 water corners (90% have 22 or fewer).

## Files

One folder per dataset, `data/labels/pool/` (git-ignored):

```
frames.json             the picked frames: name, video, frame index, size
images/<name>.jpg       the frames (JPEG, quality 95)
labels/<name>.json      the labels: the model's first labels (`auto`), which you check and fix, or drawn by hand
masks/<name>.png        training masks drawn from labels/: one class number per pixel (0, 1, 2)
```

A frame's name is `<video name>_<frame index, 6 digits>`, e.g. `IMG_4884_000261`. A label file, in pixels of the
full-size frame:

```json
{
 "video": "data/input/IMG_4884.MOV", "frame_idx": 261, "image": "IMG_4884_000261.jpg", "width": 1920, "height": 1080,
 "status": "auto",
 "water": [[[0, 352], [1740, 340], [1918, 366], [1918, 1079], [0, 1079]]],
 "ropes": [[[4.2, 487.5], [1915.9, 405.5], [1916.1, 410.5], [3.8, 496.5]]],
 "source": "pool-unet 20261116-224516"
}
```

`status`: `auto` (the model's labels, not checked yet), `fixed` (changed or drawn by hand), `accepted` (checked, the
model was right). Training uses only `fixed` and `accepted` frames. The Labels page sets it when you save: `accepted`
if the shapes are still the model's, `fixed` if you changed them (a fixed frame stays fixed). `source`: the model that
made the first labels (the training run), empty for frames drawn by hand.

Files written before 2026-11-16 store a rope as `{"points": [[x1, y1], [x2, y2]], "widths": [w1, w2]}`; it is read as
the 4 corners of that band (the same training mask, checked on all 182 files) and written as a polygon at the next
save.

## Steps

```bash
python -m pool_segmentation.labeling pick data/input/*.MOV   # 1. pick frames: one every 2 s, at most 25 per video
python -m pool_segmentation.labeling auto                     # 2. the newest trained model labels their water
python -m pool_segmentation.app                               # 3. check and fix them: http://127.0.0.1:8001
python -m pool_segmentation.labeling masks                    # 4. training masks from the checked frames
python -m pool_segmentation.labeling status                   # how many frames are checked
```

- `pick` skips frames already in the dataset, so it can be run again with more videos or `--every 1`.
- `auto` labels only frames without labels (`--redo`: also those not checked yet, again), with the newest training
  run in `data/models/pool/` (`--model <run>` for another). It never replaces labels you checked. No trained model yet (a
  first dataset): skip it and draw the water and ropes on the Labels page.
- `masks` writes masks for the checked frames only (`--all`: also the unchecked ones) and removes masks of frames no
  longer checked, so `masks/` is exactly the training set.
- Start with about 100 frames, train, look where the model fails, then add frames like those.
- Paths are relative to the folder you run them in: the dataset is `data/labels/pool/` (`--dataset` to change it).
  The Labels page needs the `app` extra (`pip install "pool-segmentation[app]"`).

## How the model's labels are made (`pool_segmentation/labeling/model_labeler.py`, `shapes.py`)

- Water: the trained model's water mask, with notches that reach the frame's edge filled (along each edge, non-pool
  stretches shorter than 20% of the edge between two pool stretches: a swimmer or a floor marking running out of the
  picture; stretches touching a corner, usually deck, are left alone), holes filled, pieces under 0.5% of the frame
  dropped, then simplified to a polygon (8 px tolerance), with corners near the frame's edge moved onto it and the
  corners in between along that edge dropped.
- Ropes: none yet (the model has no rope class); draw them on the Labels page.

## The Labels page (`python -m pool_segmentation.app`)

| Part | What it does |
|---|---|
| Frame list | Every frame with its status (orange: not checked, green: checked), filter (all / not checked / fixed / accepted), how many are checked. Click to open |
| Editor | The frame with the water (blue) and ropes (orange) drawn on top. A tooltip next to the mouse names the class under it, as the training mask will have it (Lane rope 2, Water, Other; over a corner also what dragging it does). Corners: squares for water, circles for ropes; drag one to move it. Click a rope or water area to select it; the selected area shows + handles on its edges to add a corner. Drag on the frame to move it, mouse wheel to zoom |
| Tools | + Rope (R), + Water (W): click dots around the area. Each dot connects to the last, and from the third dot on the area fills in (the closing edge dashed); nothing follows the mouse. A drag moves the frame meanwhile (a click without moving is a dot). Click the first dot, double-click or press Enter to finish; Backspace (or Z) removes the last dot, Esc cancels. A click beyond the frame puts the dot on its edge. A new rope is added as drawn; new water is merged into the water and the outlines are simplified again on the server (the fast fix for a swimmer cut out of the pool). Too much water (deck marked as water): move or delete corners. Delete (the selected corner, rope or water area), Undo / Redo (Z / Shift+Z) |
| Save | Save (S) and Save & next (A). The status is set from the shapes: `accepted` if they are still the model's, `fixed` if you changed them. ← / → open the previous / next frame and save changes first. An area still being drawn (3 dots or more) is finished before saving or leaving the frame: what is on screen is saved |

Other apps can show the page too, under a path of their own: `app.mount("/labels", create_app(labels_dir, home=("← Videos",
"/")))` (`pool_segmentation/app/server.py`); `home` adds a link back to that app. The page uses relative links only, so
it works at the root and under such a path. swim-analytics shows it this way at `/labels/`.

API (`pool_segmentation/app/server.py`): `GET /api/labels` (frames, statuses, the link back), `GET /api/labels/<name>` (the frame and its labels),
`GET /api/labels/<name>/image`, `PUT /api/labels/<name>` (water, ropes; checked against the label format; the status
is set from the shapes, the frame's video, index and size come from the dataset, not the page), `POST /api/labels/<name>/water` (+ Water: the
water polygons with an area merged in; nothing saved).

## Next

- Train the model: `docs/model.md` (water first; training reads the checked labels directly, with the same
  `rasterize()` as `masks`).
- Ropes as a third class of the model (then `auto` labels them too), then as lines per frame (swim-analytics: the
  lane of each swimmer).
