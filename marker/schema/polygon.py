from __future__ import annotations
import copy
from functools import cached_property
from typing import List

from pydantic import BaseModel, field_validator, computed_field


class PolygonBox(BaseModel):
    polygon: List[List[float]]

    @field_validator("polygon")
    @classmethod
    def check_elements(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != 4:
            raise ValueError("corner must have 4 elements")

        for corner in v:
            if len(corner) != 2:
                raise ValueError("corner must have 2 elements")

        min_x = min([corner[0] for corner in v])
        min_y = min([corner[1] for corner in v])

        # Ensure corners are clockwise from top left
        corner_error = f" .Corners are {v}"
        assert v[2][1] >= min_y, (
            "bottom right corner should have a greater y value than top right corner"
            + corner_error
        )
        assert v[3][1] >= min_y, (
            "bottom left corner should have a greater y value than top left corner"
            + corner_error
        )
        assert v[1][0] >= min_x, (
            "top right corner should have a greater x value than top left corner"
            + corner_error
        )
        assert v[2][0] >= min_x, (
            "bottom right corner should have a greater x value than bottom left corner"
            + corner_error
        )
        return v

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def area(self):
        return self.width * self.height

    @property
    def center(self):
        return [(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2]

    @property
    def size(self):
        return [self.width, self.height]

    @property
    def x_start(self):
        return self.bbox[0]

    @property
    def y_start(self):
        return self.bbox[1]

    @property
    def x_end(self):
        return self.bbox[2]

    @property
    def y_end(self):
        return self.bbox[3]

    @computed_field
    @cached_property
    def bbox(self) -> List[float]:
        # Cached: a PolygonBox's corners are never mutated in place (expand /
        # rescale / merge / fit_to_bounds all return new instances), so bbox is
        # immutable for the lifetime of an instance. bbox and the derived
        # width/height/area/center/x_start... accessors are the hottest
        # primitives in the pipeline (intersection matrices, line merge,
        # sorting, marginalia), so memoizing removes a large amount of repeated
        # min/max work.
        min_x = min([corner[0] for corner in self.polygon])
        min_y = min([corner[1] for corner in self.polygon])
        max_x = max([corner[0] for corner in self.polygon])
        max_y = max([corner[1] for corner in self.polygon])
        return [min_x, min_y, max_x, max_y]

    def expand(self, x_margin: float, y_margin: float) -> PolygonBox:
        x_margin = x_margin * self.width
        y_margin = y_margin * self.height
        p = self.polygon
        new_polygon = [
            [p[0][0] - x_margin, p[0][1] - y_margin],
            [p[1][0] + x_margin, p[1][1] - y_margin],
            [p[2][0] + x_margin, p[2][1] + y_margin],
            [p[3][0] - x_margin, p[3][1] + y_margin],
        ]
        return PolygonBox.model_construct(polygon=new_polygon)

    def minimum_gap(self, other: PolygonBox):
        if self.intersection_pct(other) > 0:
            return 0

        def dist(p1, p2):
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        left = other.bbox[2] < self.bbox[0]
        right = self.bbox[2] < other.bbox[0]
        bottom = other.bbox[3] < self.bbox[1]
        top = self.bbox[3] < other.bbox[1]
        if top and left:
            return dist((self.bbox[0], self.bbox[3]), (other.bbox[2], other.bbox[1]))
        elif left and bottom:
            return dist((self.bbox[0], self.bbox[1]), (other.bbox[2], other.bbox[3]))
        elif bottom and right:
            return dist((self.bbox[2], self.bbox[1]), (other.bbox[0], other.bbox[3]))
        elif right and top:
            return dist((self.bbox[2], self.bbox[3]), (other.bbox[0], other.bbox[1]))
        elif left:
            return self.bbox[0] - other.bbox[2]
        elif right:
            return other.bbox[0] - self.bbox[2]
        elif bottom:
            return self.bbox[1] - other.bbox[3]
        elif top:
            return other.bbox[1] - self.bbox[3]
        else:
            return 0

    def center_distance(
        self,
        other: PolygonBox,
        x_weight: float = 1,
        y_weight: float = 1,
        absolute=False,
    ):
        if not absolute:
            return (
                (self.center[0] - other.center[0]) ** 2 * x_weight
                + (self.center[1] - other.center[1]) ** 2 * y_weight
            ) ** 0.5
        else:
            return (
                abs(self.center[0] - other.center[0]) * x_weight
                + abs(self.center[1] - other.center[1]) * y_weight
            )

    def rescale(self, old_size, new_size):
        # Point is in x, y format
        page_width, page_height = old_size
        img_width, img_height = new_size

        width_scaler = img_width / page_width
        height_scaler = img_height / page_height

        new_corners = [
            [corner[0] * width_scaler, corner[1] * height_scaler]
            for corner in self.polygon
        ]
        return PolygonBox.model_construct(polygon=new_corners)

    def fit_to_bounds(self, bounds):
        b0, b1, b2, b3 = bounds[0], bounds[1], bounds[2], bounds[3]
        new_corners = [
            [max(min(corner[0], b2), b0), max(min(corner[1], b3), b1)]
            for corner in self.polygon
        ]
        return PolygonBox.model_construct(polygon=new_corners)

    def overlap_x(self, other: PolygonBox):
        return max(
            0, min(self.bbox[2], other.bbox[2]) - max(self.bbox[0], other.bbox[0])
        )

    def overlap_y(self, other: PolygonBox):
        return max(
            0, min(self.bbox[3], other.bbox[3]) - max(self.bbox[1], other.bbox[1])
        )

    def intersection_area(self, other: PolygonBox):
        return self.overlap_x(other) * self.overlap_y(other)

    def intersection_pct(self, other: PolygonBox):
        if self.area == 0:
            return 0

        intersection = self.intersection_area(other)
        return intersection / self.area

    def merge(self, others: List[PolygonBox]) -> PolygonBox:
        corners = []
        for i in range(len(self.polygon)):
            x_coords = [self.polygon[i][0]] + [other.polygon[i][0] for other in others]
            y_coords = [self.polygon[i][1]] + [other.polygon[i][1] for other in others]
            min_x = min(x_coords)
            min_y = min(y_coords)
            max_x = max(x_coords)
            max_y = max(y_coords)

            if i == 0:
                corners.append([min_x, min_y])
            elif i == 1:
                corners.append([max_x, min_y])
            elif i == 2:
                corners.append([max_x, max_y])
            elif i == 3:
                corners.append([min_x, max_y])
        return PolygonBox(polygon=corners)

    @classmethod
    def from_bbox(cls, bbox: List[float], ensure_nonzero_area=False):
        x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        if ensure_nonzero_area:
            x1 = max(x1, x0 + 1.0)
            y1 = max(y1, y0 + 1.0)
        elif x1 < x0 or y1 < y0:
            return cls(
                polygon=[
                    [x0, y0],
                    [x1, y0],
                    [x1, y1],
                    [x0, y1],
                ]
            )
        obj = cls.model_construct(
            polygon=[
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
            ]
        )
        obj.__dict__["bbox"] = [x0, y0, x1, y1]
        return obj
