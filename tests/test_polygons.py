import numpy as np

from pool_segmentation import water_polygons


def test_water_outline_from_the_model_mask():
    water = np.zeros((100, 200), bool)
    water[40:, :] = True  # the lower part of the frame
    water[60:70, 90:110] = False  # a swimmer inside: a hole, not kept
    water[5:7, 5:7] = True  # a speck: dropped

    [polygon] = water_polygons(water)

    assert sorted(polygon) == [(0, 40), (0, 99), (199, 40), (199, 99)]


def test_two_pieces_of_water_are_two_polygons():
    water = np.zeros((100, 200), bool)
    water[40:, :90] = True
    water[40:, 110:] = True  # split by a pillar in front of the camera

    assert len(water_polygons(water)) == 2


def test_no_water():
    assert water_polygons(np.zeros((100, 200), bool)) == []  # camera on the deck
