"""Masks to label shapes: the pool's water as polygons with few corners, easy to fix by dragging corners on the Labels
page (docs/labeling.md). Used for the model's first labels (`auto`) and the page's + Water.

Measured on 42 frames (2026-11-16): a corner wherever the outline bends more than 8 px keeps the pool area within IoU
0.99 of the mask with about 50 corners instead of about 6,000 outline points.
"""

import cv2
import numpy as np

from pool_segmentation.labeling.labels import Point

WATER_TOLERANCE = 8.0  # pixels: the polygon stays within this distance of the mask's outline
MIN_POOL_PIECE = 0.005  # pieces of pool smaller than this share of the frame are dropped
EDGE_NOTCH = 0.2  # non-pool stretches along a frame edge shorter than this share of the edge are notches to fill


def mask_polygons(water: np.ndarray, tolerance: float = WATER_TOLERANCE) -> list[list[Point]]:
    """A water mask (True = pool) as label polygons: notches at the frame's edge and holes filled (a swimmer the
    mask leaves out counts as pool), pieces under 0.5% of the frame dropped, corners along the frame's edge snapped
    onto it."""
    return area_polygons(close_edge_notches(water.astype(np.uint8)), tolerance)


def area_polygons(area: np.ndarray, tolerance: float = WATER_TOLERANCE, min_piece: float = MIN_POOL_PIECE) -> list[list[Point]]:
    """Outer outlines of an area mask as polygons with few corners (holes are filled: our polygons have none)."""
    area = area.astype(np.uint8)
    contours, _ = cv2.findContours(area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    height, width = area.shape
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_piece * area.size:
            continue
        corners = cv2.approxPolyDP(contour, tolerance, True)[:, 0].astype(float)
        corners = along_frame_edges(corners, width, height, tolerance)
        if len(corners) >= 3:
            polygons.append([(float(x), float(y)) for x, y in corners])
    return polygons


def add_water(polygons: list[list[Point]], added: list[Point], width: int, height: int,
              tolerance: float = WATER_TOLERANCE) -> list[list[Point]]:
    """Water polygons with an area merged in, outlines simplified again (the Labels page's + Water)."""
    area = np.zeros((height, width), np.uint8)
    for polygon in polygons + [added]:
        _fill(area, polygon, 1)
    # Keep pieces drawn by hand even if small (a few pixels of pool beside a sign)
    return area_polygons(area, tolerance, min_piece=0.0005)


def _fill(area: np.ndarray, polygon: list[Point], value: int) -> None:
    cv2.fillPoly(area, [np.round(np.array(polygon, float) * 16).astype(np.int32)], value, lineType=cv2.LINE_8, shift=4)


def close_edge_notches(area: np.ndarray, max_share: float = EDGE_NOTCH) -> np.ndarray:
    """Close the pool's outline along the frame's edges, so notches that reach the edge are filled like holes.

    Along each edge of the frame, a stretch of non-pool between two stretches of pool, shorter than `max_share` of the
    edge, is something narrow in the pool running out of the picture (a swimmer, a floor marking): it is marked as
    pool on that edge row / column. Stretches touching the frame's corners are left alone (the deck usually is one).
    """
    area = area.copy()
    for line in (area[0, :], area[-1, :], area[:, 0], area[:, -1]):  # views: changes go into `area`
        pool = np.flatnonzero(line)
        if len(pool) < 2:
            continue
        for a, b in zip(pool[:-1], pool[1:]):
            if 1 < b - a <= max_share * len(line):
                line[a + 1:b] = 1
    return area


def along_frame_edges(corners: np.ndarray, width: int, height: int, tolerance: float) -> np.ndarray:
    """Corners near the frame's edge moved onto it, and corners in between two others on the same edge dropped: where
    the pool runs out of the picture, its outline is the frame's edge, with a corner at each end."""
    corners = corners.copy()
    for axis, last in ((0, width - 1), (1, height - 1)):
        corners[corners[:, axis] <= tolerance, axis] = 0
        corners[corners[:, axis] >= last - tolerance, axis] = last

    def edges(p):
        return {(axis, p[axis]) for axis, last in ((0, width - 1), (1, height - 1)) if p[axis] in (0, last)}

    keep = []
    n = len(corners)
    for k in range(n):
        shared = edges(corners[k]) & edges(corners[k - 1]) & edges(corners[(k + 1) % n])
        if not shared:
            keep.append(corners[k])
    return np.array(keep) if len(keep) >= 3 else corners
