"""The water's outline: a mask (True = water) as polygons with few corners."""

import cv2
import numpy as np

TOLERANCE = 3  # pixels: corners of the outline stay within this distance of the mask's edge
MIN_PIECE = 0.001  # pieces of water smaller than this share of the frame are dropped (specks)

Polygon = list[tuple[int, int]]


def water_polygons(water: np.ndarray, tolerance: float = TOLERANCE, min_piece: float = MIN_PIECE) -> list[Polygon]:
    """The outline of each piece of water in a mask (True = water), simplified to fewer corners. Holes are not kept:
    whatever the water surrounds (a swimmer the model missed) counts as pool. No water: an empty list."""
    contours, _ = cv2.findContours(water.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_piece * water.size:
            continue
        corners = cv2.approxPolyDP(contour, tolerance, True)[:, 0]
        if len(corners) >= 3:
            polygons.append([(int(x), int(y)) for x, y in corners])
    return polygons
