import cv2
import numpy as np
import argparse
import json
from geometry import Rectangle, Polygon
from hands import HandTracker
from persistence import BackgroundTracker


# ---------- Helpers ----------
def clamp_rect(x, y, w, h, frame_w, frame_h):
    """Clamp a rectangle's top-left so it stays fully inside the frame."""
    if w < 1 or h < 1:
        return int(x), int(y), int(w), int(h)
    x = max(0, min(int(x), frame_w - int(w)))
    y = max(0, min(int(y), frame_h - int(h)))
    return int(x), int(y), int(w), int(h)

def clamp_center_for_rect(cx, cy, w_rect, h_rect, frame_w, frame_h):
    """Clamp a rectangle's center so the rect stays fully inside the frame."""
    half_w = max(1, int(w_rect / 2))
    half_h = max(1, int(h_rect / 2))
    cx = min(max(int(cx), half_w), frame_w - half_w)
    cy = min(max(int(cy), half_h), frame_h - half_h)
    return int(cx), int(cy)


# ---------- Argparse / Utils ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--max-fps", type=int, default=60)
    return ap.parse_args()

def draw_text(frame, text, y=30):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

def load_shapes(path="shapes.json"):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        shapes = []
        for item in data:
            if item["type"] == "rect":
                shapes.append(Rectangle(**item["data"]))
            elif item["type"] == "poly":
                shapes.append(Polygon(points=[tuple(p) for p in item["data"]["points"]]))
        return shapes
    except Exception:
        return []

def save_shapes(shapes, path="shapes.json"):
    data = []
    for s in shapes:
        if isinstance(s, Rectangle):
            data.append({"type":"rect","data":{"x":int(s.x),"y":int(s.y),"w":int(s.w),"h":int(s.h),"angle":float(s.angle)}})
        elif isinstance(s, Polygon):
            data.append({"type":"poly","data":{"points":[(float(px), float(py)) for (px,py) in s.points]}})
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------- Main ----------
def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    tracker = HandTracker()
    bg_tracker = BackgroundTracker()

    mode = "EDIT"  # or "INTERACT"
    shapes = load_shapes()

    # Interaction / editing state
    polygon_mode = False
    poly_pts = []
    drawing = False                # currently dragging a new rect
    start_pt = None                # anchor corner for new rect
    curr_mouse = None
    drag_has_moved = False         # to avoid odd first-frame artifacts
    selected_idx = -1
    grabbed = False
    grab_offset = (0, 0)

    # Pointing ray smoothing
    ray_o = None
    ray_d = None
    alpha = 0.5

    # Create window & bind mouse once
    cv2.namedWindow("2D AR App", cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, start_pt, curr_mouse, drag_has_moved, shapes, polygon_mode, poly_pts
        curr_mouse = (x, y)

        if mode != "EDIT":
            return

        # Polygon editor
        if polygon_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                # clamp the clicked vertex into frame bounds using last-known frame size
                frame_h, frame_w = param["frame_shape"]
                px = max(0, min(x, frame_w - 1))
                py = max(0, min(y, frame_h - 1))
                poly_pts.append((px, py))
            return

        # Rectangle editor (corner-expand)
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_pt = (x, y)
            drag_has_moved = False

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            drag_has_moved = True

        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            if start_pt is not None:
                x1, y1 = start_pt
                x2, y2 = x, y
                x0, y0 = min(x1, x2), min(y1, y2)
                w0, h0 = abs(x2 - x1), abs(y2 - y1)

                # clamp to current frame size
                frame_h, frame_w = param["frame_shape"]
                x0c, y0c, w0c, h0c = clamp_rect(x0, y0, w0, h0, frame_w, frame_h)

                if w0c > 5 and h0c > 5:
                    shapes.append(Rectangle(x0c, y0c, w0c, h0c))
            start_pt = None

    # we pass a dict param so callback can read frame size for clamping polygon points
    mouse_param = {"frame_shape": (args.height, args.width)}
    cv2.setMouseCallback("2D AR App", on_mouse, mouse_param)

    # --------------- Loop ---------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror for a natural UX
        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]
        mouse_param["frame_shape"] = (frame_h, frame_w)

        # Background optical-feature transform (for persistence demo)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H = bg_tracker.estimate_transform(gray)

        # Hand tracking
        hand = tracker.process(frame)
        if hand:
            o = np.array(hand.index_mcp, dtype=np.float32)
            tip = np.array(hand.index_tip, dtype=np.float32)
            d = tip - o
            n = np.linalg.norm(d) + 1e-6
            d = d / n
            if ray_o is None:
                ray_o = o
                ray_d = d
            else:
                ray_o = alpha * o + (1 - alpha) * ray_o
                ray_d = alpha * d + (1 - alpha) * ray_d

        # Apply persistence (when not holding anything): move shapes a bit with H, then clamp
        if not grabbed:
            for s in shapes:
                if isinstance(s, Rectangle):
                    cx, cy = s.center()
                    pt = np.array([cx, cy, 1], dtype=np.float32)
                    new = H @ pt
                    s.move_to((float(new[0]), float(new[1])))
                elif isinstance(s, Polygon):
                    new_points = []
                    for (px, py) in s.points:
                        pt = np.array([px, py, 1], dtype=np.float32)
                        new = H @ pt
                        new_points.append((float(new[0]), float(new[1])))
                    s.points = new_points

            # Clamp shapes inside window after persistence motion
            for s in shapes:
                if isinstance(s, Rectangle):
                    cx, cy = s.center()
                    x_rect = cx - s.w / 2
                    y_rect = cy - s.h / 2
                    x_rect, y_rect, _, _ = clamp_rect(x_rect, y_rect, s.w, s.h, frame_w, frame_h)
                    s.move_to((x_rect + s.w / 2, y_rect + s.h / 2))
                elif isinstance(s, Polygon):
                    clamped = []
                    for (px, py) in s.points:
                        clamped.append((max(0, min(int(px), frame_w - 1)),
                                        max(0, min(int(py), frame_h - 1))))
                    s.points = clamped

        # Selection + grabbing in INTERACT mode
        selected_idx = -1
        best_t = 1e9
        if mode == "INTERACT" and hand and ray_o is not None:
            # ray-select closest shape
            for i, s in enumerate(shapes):
                hit = s.intersects_ray(tuple(ray_o), tuple(ray_d))
                if hit is not None and hit[0] < best_t:
                    best_t = hit[0]
                    selected_idx = i

            # Grab / release
            if selected_idx != -1 and hand.pinch and not grabbed:
                grabbed = True
                center = shapes[selected_idx].center()
                # offset between hit point & center – approximate with current best_t
                grab_offset = (center[0] - (ray_o[0] + best_t * ray_d[0]),
                               center[1] - (ray_o[1] + best_t * ray_d[1]))
            elif grabbed and not hand.pinch:
                grabbed = False

            # While grabbed: move, but keep inside frame
            if grabbed and selected_idx != -1:
                t = best_t if best_t < 1e8 else 50.0
                intended_cx = ray_o[0] + t * ray_d[0] + grab_offset[0]
                intended_cy = ray_o[1] + t * ray_d[1] + grab_offset[1]

                s = shapes[selected_idx]
                if isinstance(s, Rectangle):
                    clamped_cx, clamped_cy = clamp_center_for_rect(intended_cx, intended_cy, s.w, s.h, frame_w, frame_h)
                    s.move_to((clamped_cx, clamped_cy))
                elif isinstance(s, Polygon):
                    # Move by delta, then clamp each vertex
                    cx_old, cy_old = s.center()
                    dx = intended_cx - cx_old
                    dy = intended_cy - cy_old
                    moved = []
                    for (px, py) in s.points:
                        nx = max(0, min(int(px + dx), frame_w - 1))
                        ny = max(0, min(int(py + dy), frame_h - 1))
                        moved.append((nx, ny))
                    s.points = moved

        # Draw shapes
        for i, s in enumerate(shapes):
            s.draw(frame, selected=(i == selected_idx))

        # --- EDIT MODE PREVIEW (draw BEFORE text/imshow) ---
        if mode == "EDIT":
            if drawing and start_pt is not None and curr_mouse is not None and drag_has_moved:
                x1, y1 = start_pt
                mx, my = curr_mouse
                # raw rectangle
                x0, y0 = min(x1, mx), min(y1, my)
                w0, h0 = abs(mx - x1), abs(my - y1)
                # clamp preview
                x0c, y0c, w0c, h0c = clamp_rect(x0, y0, w0, h0, frame_w, frame_h)

                # outline
                cv2.rectangle(frame, (x0c, y0c), (x0c + w0c, y0c + h0c), (0, 255, 255), 2)
                # fill
                overlay = frame.copy()
                cv2.rectangle(overlay, (x0c, y0c), (x0c + w0c, y0c + h0c), (0, 255, 255), -1)
                frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

            # polygon live preview (existing vertices + live edge)
            if polygon_mode and len(poly_pts) >= 1:
                for (px, py) in poly_pts:
                    cv2.circle(frame, (int(px), int(py)), 4, (255, 200, 0), -1)
                pts_np = np.array(poly_pts, dtype=np.int32)
                cv2.polylines(frame, [pts_np], False, (255, 200, 0), 2)
                if curr_mouse is not None:
                    last = poly_pts[-1]
                    cv2.line(frame, (int(last[0]), int(last[1])),
                             (int(curr_mouse[0]), int(curr_mouse[1])), (0, 255, 0), 2)

        # HUD text
        draw_text(frame, f"Mode: {mode}", y=30)
        if hand:
            draw_text(frame, f"Pinch: {hand.pinch}", y=60)

        # (Optional) visualize pointing ray for debugging
        if ray_o is not None and ray_d is not None:
            p1 = tuple(map(int, ray_o))
            p2 = (int(ray_o[0] + 200 * ray_d[0]), int(ray_o[1] + 200 * ray_d[1]))
            cv2.line(frame, p1, p2, (0, 0, 255), 2)

        # Show
        cv2.imshow("2D AR App", frame)
        key = cv2.waitKey(1) & 0xFF

        # Keys
        if key in [27, ord('q'), ord('Q')]:
            break
        elif key in [ord('e'), ord('E')]:
            mode = "EDIT"
        elif key in [ord('i'), ord('I')]:
            mode = "INTERACT"
        elif key in [ord('s'), ord('S')]:
            save_shapes(shapes)
        elif key in [ord('l'), ord('L')]:
            shapes = load_shapes()
        elif key in [ord('p'), ord('P')] and mode == "EDIT":
            polygon_mode = not polygon_mode
            if not polygon_mode and len(poly_pts) >= 3:
                shapes.append(Polygon(points=list(poly_pts)))
            poly_pts = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
