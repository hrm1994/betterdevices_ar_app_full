import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

Point = Tuple[float, float]

def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1]

def _sub(a, b):
    return (a[0]-b[0], a[1]-b[1])

def _cross(a, b):
    return a[0]*b[1] - a[1]*b[0]

def ray_segment_intersection(ray_o: Point, ray_d: Point, p1: Point, p2: Point) -> Optional[Tuple[float, Point]]:
    """Return (t, point) where ray_o + t*ray_d intersects segment p1-p2, or None."""
    v1 = _sub(ray_o, p1)
    v2 = _sub(p2, p1)
    r = ray_d
    denom = _cross(r, v2)
    if abs(denom) < 1e-8:
        return None
    t = _cross(v2, v1) / denom
    u = _cross(r, v1) / denom
    if t >= 0 and 0 <= u <= 1:
        inter = (ray_o[0] + t*ray_d[0], ray_o[1] + t*ray_d[1])
        return t, inter
    return None

@dataclass
class Shape:
    def bbox(self) -> Tuple[float, float, float, float]:
        raise NotImplementedError
    def draw(self, frame, selected: bool = False):
        raise NotImplementedError
    def intersects_ray(self, ray_o: Point, ray_d: Point) -> Optional[Tuple[float, Point]]:
        raise NotImplementedError
    def move_to(self, new_center: Point):
        raise NotImplementedError
    def center(self) -> Point:
        raise NotImplementedError

@dataclass
class Rectangle(Shape):
    x: float
    y: float
    w: float
    h: float
    angle: float = 0.0  # degrees

    def bbox(self):
        return (min(self.x, self.x + self.w), min(self.y, self.y + self.h),
                max(self.x, self.x + self.w), max(self.y, self.y + self.h))

    def corners(self) -> List[Point]:
        c = (self.x + self.w/2, self.y + self.h/2)
        pts = [
            (self.x, self.y),
            (self.x + self.w, self.y),
            (self.x + self.w, self.y + self.h),
            (self.x, self.y + self.h)
        ]
        if abs(self.angle) < 1e-6:
            return pts
        ang = math.radians(self.angle)
        ca, sa = math.cos(ang), math.sin(ang)
        out = []
        for px, py in pts:
            dx, dy = px - c[0], py - c[1]
            rx = dx*ca - dy*sa + c[0]
            ry = dx*sa + dy*ca + c[1]
            out.append((rx, ry))
        return out

    def draw(self, frame, selected=False):
        import cv2
        pts = self.corners()
        pts_i = np.array(pts, dtype=np.int32)
        thickness = 4 if selected else 2
        cv2.polylines(frame, [pts_i], True, (0,255,0) if selected else (255,255,255), thickness)

    def intersects_ray(self, ray_o: Point, ray_d: Point):
        x1,y1,x2,y2 = self.bbox()
        edges = [((x1,y1),(x2,y1)),((x2,y1),(x2,y2)),((x2,y2),(x1,y2)),((x1,y2),(x1,y1))]
        best = None
        for e in edges:
            hit = ray_segment_intersection(ray_o, ray_d, e[0], e[1])
            if hit is not None:
                if best is None or hit[0] < best[0]:
                    best = hit
        return best

    def move_to(self, new_center: Point):
        cx, cy = self.center()
        dx, dy = new_center[0]-cx, new_center[1]-cy
        self.x += dx
        self.y += dy

    def center(self) -> Point:
        return (self.x + self.w/2, self.y + self.h/2)

@dataclass
class Polygon(Shape):
    points: List[Point] = field(default_factory=list)

    def bbox(self):
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))

    def draw(self, frame, selected=False):
        import cv2
        if len(self.points) < 2: return
        pts = np.array(self.points, dtype=np.int32)
        thickness = 4 if selected else 2
        cv2.polylines(frame, [pts], True, (0,255,0) if selected else (255,255,255), thickness)

    def intersects_ray(self, ray_o: Point, ray_d: Point):
        n = len(self.points)
        if n < 2: return None
        best = None
        for i in range(n):
            p1 = self.points[i]
            p2 = self.points[(i+1)%n]
            hit = ray_segment_intersection(ray_o, ray_d, p1, p2)
            if hit is not None:
                if best is None or hit[0] < best[0]:
                    best = hit
        return best

    def move_to(self, new_center: Point):
        cx, cy = self.center()
        dx, dy = new_center[0]-cx, new_center[1]-cy
        self.points = [(p[0]+dx, p[1]+dy) for p in self.points]

    def center(self) -> Point:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (sum(xs)/len(xs), sum(ys)/len(ys))
