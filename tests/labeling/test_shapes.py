import numpy as np

from pool_segmentation.labeling.shapes import add_water, mask_polygons

H, W = 200, 300


def test_water_mask_becomes_a_polygon_with_few_corners_and_no_holes():
    water = np.zeros((H, W), bool)
    water[50:180, 20:280] = True
    water[100:120, 100:160] = False  # a swimmer inside the pool the mask left out: a hole

    [polygon] = mask_polygons(water)

    assert len(polygon) == 4
    xs, ys = zip(*polygon)
    assert (min(xs), max(xs), min(ys), max(ys)) == (20, 279, 50, 179)


def test_tiny_pool_pieces_are_dropped():
    water = np.zeros((H, W), bool)
    water[50:180, 20:280] = True
    water[5:8, 5:8] = True  # a few pixels of "water" on the deck

    assert len(mask_polygons(water)) == 1


def test_corners_along_the_frame_edge_are_dropped():
    water = np.zeros((H, W), bool)
    water[60:, :] = True
    for x in range(0, W, 20):
        water[60:200, x:x + 3] = False if x % 40 else True  # jagged mask just inside the bottom-left
    water[:, :2] = False  # the mask stops 2 px short of the frame's left edge
    water[150:, 296:] = False  # and a bit short of the right edge, lower down

    [polygon] = mask_polygons(water)

    xs = sorted({x for x, _ in polygon})
    assert xs[0] == 0 and xs[-1] == W - 1  # snapped onto the frame's edges
    on_left = [p for p in polygon if p[0] == 0]
    assert len(on_left) == 2  # only the two ends of the run along the left edge


def test_a_swimmer_cut_off_by_the_frame_edge_is_filled_but_the_deck_corner_is_not():
    water = np.zeros((H, W), bool)
    water[:, :] = True
    water[80:110, :40] = False  # a swimmer entering the picture from the left: a notch, not a hole
    water[:50, 200:] = False  # the deck in the top-right corner

    [polygon] = mask_polygons(water)

    assert all(not (0 < x < 40 and 80 < y < 110) for x, y in polygon)  # no corners around the swimmer
    assert (200, 0) in polygon or any(abs(x - 200) <= 8 and y <= 8 for x, y in polygon)  # the deck stays out
    assert len(polygon) <= 7


def test_no_water():
    assert mask_polygons(np.zeros((H, W), bool)) == []


def test_add_water_merges_the_added_area():
    pool = [[(0, 60), (299, 60), (299, 199), (0, 199)]]
    notch = [[(0, 60), (299, 60), (299, 199), (0, 199), (0, 130), (40, 130), (40, 100), (0, 100)]]  # a swimmer cut out

    [merged] = add_water(notch, [(0, 90), (60, 90), (60, 140), (0, 140)], W, H)
    assert sorted(merged) == sorted(pool[0])  # the notch is gone: back to 4 corners

    assert add_water(pool, [(50, 0), (100, 0), (100, 20), (50, 20)], W, H)[1:]  # a separate area: two polygons
