import cv2
import numpy as np
import argparse
import json
from geometry import Rectangle, Polygon
from hands import HandTracker
from persistence import BackgroundTracker




def _angle_ok(ray_d, pt_from_o, cos_min=0.995):  # ≈ 5.7° cone
    v = np.array(pt_from_o, dtype=np.float32)
    vn = np.linalg.norm(v) + 1e-6
    v = v / vn
    d = np.array(ray_d, dtype=np.float32)
    dn = np.linalg.norm(d) + 1e-6
    d = d / dn
    return float(d @ v) > cos_min

def _is_valid_H(H, frame_w, frame_h):
    if H is None or not isinstance(H, np.ndarray) or H.shape != (3, 3):
        return False
    if not np.all(np.isfinite(H)):
        return False
    # Expect affine-like homography with bottom row [0,0,1]
    if np.linalg.norm(H[2, :] - np.array([0.0, 0.0, 1.0], dtype=np.float32)) > 1e-3:
        return False
    if abs(float(H[2, 2])) < 1e-6:
        return False
    # Translation should be reasonable (guard wild estimates)
    tx, ty = float(H[0, 2]), float(H[1, 2])
    if abs(tx) > 0.25 * frame_w or abs(ty) > 0.25 * frame_h:
        return False
    # Scale/rotation not absurd (avoid near-zero scale / reflection explosions)
    a, b, c, d = float(H[0,0]), float(H[0,1]), float(H[1,0]), float(H[1,1])
    det = a*d - b*c
    if abs(det) < 1e-6 or abs(det) > 10.0:
        return False
    return True



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

    cooldown = 0  # frames to skip bg tracking after release or shape creation

    args = parse_args()
    import time
    spf = 1.0 / max(1, args.max_fps)
    last = 0.0

    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    tracker = HandTracker(
    pinch_on_scale=0.30,     # pinch engages around 0.30 * hand_size
    pinch_off_scale=0.40,    # releases around 0.40 * hand_size
    angle_threshold_deg=160  # index should be fairly straight while pointing
)


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
    grabbed_idx = -1 
    grab_offset = (0, 0)
    grabbed_ref_angle = None
    persistence_enabled = False  # <- keep OFF to stop any auto-move; toggle with 'O'
    Last_selected_idx = -1

    grab_start_tip = None      # (x,y) fingertip position when grab begins
    grab_start_center = None   # (cx,cy) shape center when grab begins


    # Pointing ray smoothing
    ray_o = None
    ray_d = None
    alpha = 0.5

    # Create window & bind mouse once
    cv2.namedWindow("2D AR App", cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, start_pt, curr_mouse, drag_has_moved, shapes, polygon_mode, poly_pts, cooldown

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
                    cooldown = 10
                    
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


        # Background optical-feature transform (for persistence)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if mode == "INTERACT" and not grabbed and cooldown == 0 and persistence_enabled:
            H_cand = bg_tracker.estimate_transform(gray)  # 3x3
            H = H_cand if _is_valid_H(H_cand, frame_w, frame_h) else np.eye(3, dtype=np.float32)
        else:
            H = np.eye(3, dtype=np.float32)



        # tick cooldown (if active)
        if cooldown > 0:
            cooldown -= 1


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

        # ---- Pinch detection (robust) ----
        pinch_now = False
        pinch_dist_px = None
        if hand:
            idx = np.array(hand.index_tip, dtype=np.float32)
            thb = np.array(hand.thumb_tip, dtype=np.float32)
            pinch_dist_px = float(np.linalg.norm(idx - thb))
            pinch_now = bool(getattr(hand, "pinch", False))
            # Fallback if model’s pinch flag is too strict on this camera
            if not pinch_now:
                PINCH_PX_THRESHOLD = 50.0  # adjust 40–60 if needed
                pinch_now = pinch_dist_px < PINCH_PX_THRESHOLD
        else:
            pinch_now = False


        # Apply persistence (when not holding anything): move shapes a bit with H, then clamp
        # only apply background tracking if no shape is being grabbed
        # Apply persistence (background motion) only when appropriate
        if mode == "INTERACT" and not grabbed and cooldown == 0 and persistence_enabled:

            I = np.eye(3, dtype=np.float32)
            if not np.allclose(H, I, atol=1e-6):
                for s in shapes:
                    if isinstance(s, Rectangle):
                        cx, cy = s.center()
                        pt = np.array([cx, cy, 1.0], dtype=np.float32)
                        new = H @ pt
                        w = float(new[2]) if new.shape[0] == 3 else 1.0
                        if abs(w) < 1e-6:
                            continue  # skip bad transform
                        nx, ny = float(new[0] / w), float(new[1] / w)

                        # smooth & clamp to keep inside frame
                        if not hasattr(s, "smoothed_center"):
                            s.smoothed_center = (cx, cy)
                        alpha_move = 0.15
                        smx = int((1 - alpha_move) * s.smoothed_center[0] + alpha_move * nx)
                        smy = int((1 - alpha_move) * s.smoothed_center[1] + alpha_move * ny)

                        # clamp fully inside
                        smx, smy = clamp_center_for_rect(smx, smy, s.w, s.h, frame_w, frame_h)
                        s.smoothed_center = (smx, smy)
                        s.move_to(s.smoothed_center)

                    elif isinstance(s, Polygon):
                        new_points = []
                        for (x, y) in s.points:
                            pt = np.array([x, y, 1.0], dtype=np.float32)
                            new = H @ pt
                            w = float(new[2]) if new.shape[0] == 3 else 1.0
                            if abs(w) < 1e-6:
                                new_points.append((x, y))  # keep original point on bad H
                                continue
                            nx, ny = float(new[0] / w), float(new[1] / w)
                            # clamp point-by-point
                            nx = max(0, min(int(nx), frame_w - 1))
                            ny = max(0, min(int(ny), frame_h - 1))
                            new_points.append((nx, ny))
                        s.points = new_points



                    # Clamp shapes inside window after persistence motion
            for s in shapes:
                if isinstance(s, Rectangle):
                    cx, cy = s.center()
                    x_rect = cx - s.w / 2
                    y_rect = cy - s.h / 2
                    x_rect, y_rect, w_rect, h_rect = clamp_rect(x_rect, y_rect, s.w, s.h, frame_w, frame_h)
                    # store last smoothed center in the shape
                    # compute clamped center
                    clamped_cx = int(x_rect + w_rect / 2)
                    clamped_cy = int(y_rect + h_rect / 2)
                    if not hasattr(s, "smoothed_center"):
                        s.smoothed_center = (clamped_cx, clamped_cy)

                    alpha_move = 0.15  # smaller = smoother, larger = snappier
                    smoothed_cx = int(alpha_move * clamped_cx + (1 - alpha_move) * s.smoothed_center[0])
                    smoothed_cy = int(alpha_move * clamped_cy + (1 - alpha_move) * s.smoothed_center[1])

                    s.smoothed_center = (smoothed_cx, smoothed_cy)
                    s.move_to(s.smoothed_center)

                elif isinstance(s, Polygon):
                    clamped = []
                    for (px, py) in s.points:
                        clamped.append((max(0, min(int(px), frame_w - 1)),
                                        max(0, min(int(py), frame_h - 1))))
                    s.points = clamped

        # Selection + grabbing (INTERACT only)
        selected_idx = -1
        best_t = 1e9

        if mode == "INTERACT" and (hand is not None) and (ray_o is not None) and (ray_d is not None):
            # --- 1) Ray-cast selection with angle cone filter ---
            for i, s in enumerate(shapes):
                hit = s.intersects_ray(tuple(ray_o), tuple(ray_d))
                if hit is not None:
                    t, p = hit
                    if t >= 0 and _angle_ok(ray_d, (p[0]-ray_o[0], p[1]-ray_o[1])) and t < best_t:
                        best_t = t
                        selected_idx = i

            # --- 2) Pinch to grab (only if a target is selected) ---
            if (selected_idx != -1) and pinch_now and not grabbed:
                grabbed = True
                last_selected_idx = selected_idx  # <— latch the target

                # record starting positions (no snapping)
                grab_start_tip = tuple(hand.index_tip)           # (x,y) at grab start
                grab_start_center = shapes[selected_idx].center()# (cx,cy) at grab start

                # rotation reference (optional)
                v0x = hand.index_mcp[0] - hand.wrist[0]
                v0y = hand.index_mcp[1] - hand.wrist[1]
                grabbed_ref_angle = np.degrees(np.arctan2(v0y, v0x))


            # --- 3) Release pinch ---
            elif grabbed and not pinch_now:
                grabbed = False
                grab_start_tip = None
                grab_start_center = None
                cooldown = max(cooldown, 10)  # short cooldown after release

            # --- 4) While grabbed: move selected object ---
            if grabbed and last_selected_idx != -1 and selected_idx == last_selected_idx:
                s = shapes[last_selected_idx]

                # Move by the delta of fingertip since grab started (no snap)
                tipx, tipy = hand.index_tip
                if (grab_start_tip is not None) and (grab_start_center is not None):
                    dx = float(tipx - grab_start_tip[0])
                    dy = float(tipy - grab_start_tip[1])
                    target_cx = float(grab_start_center[0] + dx)
                    target_cy = float(grab_start_center[1] + dy)
                else:
                    # fallback (shouldn't happen), keep current center
                    target_cx, target_cy = s.center()


                # 4.2 keep inside frame
                if isinstance(s, Rectangle):
                    target_cx, target_cy = clamp_center_for_rect(
                        target_cx, target_cy, s.w, s.h, frame_w, frame_h
                    )

                # 4.3 smooth
                alpha_move = 0.25
                if not hasattr(s, "smoothed_center"):
                    s.smoothed_center = s.center()
                smoothed_cx = int((1 - alpha_move) * s.smoothed_center[0] + alpha_move * target_cx)
                smoothed_cy = int((1 - alpha_move) * s.smoothed_center[1] + alpha_move * target_cy)
                s.smoothed_center = (smoothed_cx, smoothed_cy)

                # 4.4 apply movement
                if isinstance(s, Rectangle):
                    s.move_to(s.smoothed_center)
                    # rotation while grabbed (optional bonus)
                    if grabbed_ref_angle is not None:
                        vx = hand.index_mcp[0] - hand.wrist[0]
                        vy = hand.index_mcp[1] - hand.wrist[1]
                        cur_ang = np.degrees(np.arctan2(vy, vx))
                        dtheta = cur_ang - grabbed_ref_angle
                        s.angle = float(s.angle + dtheta)
                        grabbed_ref_angle = cur_ang
                else:
                    # --- POLYGON: translate to target center, then rotate by wrist twist (if any) ---

                    # 1) compute translation to reach the smoothed target center
                    cx_old, cy_old = s.center()
                    dx = smoothed_cx - cx_old
                    dy = smoothed_cy - cy_old

                    # translate points first
                    moved = [(px + dx, py + dy) for (px, py) in s.points]

                    # 2) apply incremental rotation around the *new* center using wrist twist
                    if grabbed_ref_angle is not None:
                        vx = hand.index_mcp[0] - hand.wrist[0]
                        vy = hand.index_mcp[1] - hand.wrist[1]
                        cur_ang_deg = np.degrees(np.arctan2(vy, vx))
                        dtheta_rad = np.radians(cur_ang_deg - grabbed_ref_angle)

                        if abs(dtheta_rad) > 1e-6:
                            cos_t = np.cos(dtheta_rad)
                            sin_t = np.sin(dtheta_rad)
                            # rotate around smoothed center (smoothed_cx, smoothed_cy)
                            roted = []
                            for (px, py) in moved:
                                rx = smoothed_cx + cos_t * (px - smoothed_cx) - sin_t * (py - smoothed_cy)
                                ry = smoothed_cy + sin_t * (px - smoothed_cx) + cos_t * (py - smoothed_cy)
                                # clamp after rotation
                                rx = max(0, min(int(rx), frame_w - 1))
                                ry = max(0, min(int(ry), frame_h - 1))
                                roted.append((rx, ry))
                            moved = roted

                        # update reference for incremental rotation
                        grabbed_ref_angle = cur_ang_deg

                    # 3) write back points if we didn’t rotate (or after rotation)
                    if 'moved' in locals():
                        s.points = [(int(px), int(py)) for (px, py) in moved]

        else:
            # Not interacting: ensure no stale selection/grab carries into EDIT
            selected_idx = -1
            if mode != "INTERACT":
                grabbed = False


        # Draw shapes
        for i, s in enumerate(shapes):
            s.draw(frame, selected=(mode == "INTERACT" and i == selected_idx))

        # --- EDIT MODE PREVIEW (draw BEFORE text/imshow) ---
        # --- EDIT MODE PREVIEW (draw BEFORE text/imshow) ---
        if mode == "EDIT":
            if drawing and start_pt is not None and curr_mouse is not None and drag_has_moved:
                x1, y1 = start_pt
                mx, my = curr_mouse
                # raw rectangle
                x0, y0 = min(x1, mx), min(y1, my)
                w0, h0 = abs(mx - x1), abs(my - y1)

                # clamp preview
                rect = clamp_rect(x0, y0, w0, h0, frame_w, frame_h)
                if rect is not None:
                    x0c, y0c, w0c, h0c = rect

                    # outline
                    cv2.rectangle(frame,
                                (int(x0c), int(y0c)),
                                (int(x0c + w0c), int(y0c + h0c)),
                                (0, 255, 255), 2)

                    # fill
                    overlay = frame.copy()
                    cv2.rectangle(overlay,
                                (int(x0c), int(y0c)),
                                (int(x0c + w0c), int(y0c + h0c)),
                                (0, 255, 255), -1)
                    frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

            # polygon live preview (existing vertices + live edge)
            if polygon_mode and len(poly_pts) >= 1:
                for (px, py) in poly_pts:
                    cv2.circle(frame, (int(px), int(py)), 4, (255, 200, 0), -1)
                pts_np = np.array(poly_pts, dtype=np.int32)
                cv2.polylines(frame, [pts_np], False, (255, 200, 0), 2)
                if curr_mouse is not None:
                    last_pt = poly_pts[-1]   # <- renamed; no longer clobbers FPS 'last'
                    cv2.line(frame,
                            (int(last_pt[0]), int(last_pt[1])),
                            (int(curr_mouse[0]), int(curr_mouse[1])),
                            (0, 255, 0), 2)



        # HUD text
        draw_text(frame, f"Mode: {mode}", y=30)
        # Status HUD
        y = 60
        hand_ok = hand is not None
        sel_txt = "None" if selected_idx == -1 else str(selected_idx)
        grab_txt = "YES" if grabbed else "NO"
        pinch_txt = "YES" if pinch_now else "NO"
        draw_text(frame, f"Hand:{'YES' if hand_ok else 'NO'}  Pinch:{pinch_txt}  Selected:{sel_txt}  Grabbed:{grab_txt}", y)
        if pinch_dist_px is not None:
            draw_text(frame, f"PinchDist(px): {pinch_dist_px:.1f}", y+30)

        # show thumb-index distance and index angle
        if hand:
            p1, p2 = tuple(map(int, hand.index_tip)), tuple(map(int, hand.thumb_tip))
            cv2.line(frame, p1, p2, (0,255,0) if hand.pinch else (0,0,255), 2)
            # draw index vector
            imcp = tuple(map(int, hand.index_mcp))
            itip = tuple(map(int, hand.index_tip))
            cv2.line(frame, imcp, itip, (255,0,0), 2)
            # (print text values if you expose angle/thresholds from HandTracker)


        # (Optional) visualize pointing ray for debugging
        if ray_o is not None and ray_d is not None:
            p1 = tuple(map(int, ray_o))
            p2 = (int(ray_o[0] + 200 * ray_d[0]), int(ray_o[1] + 200 * ray_d[1]))
            cv2.line(frame, p1, p2, (0, 0, 255), 2)

        now = time.perf_counter()
        sleep = spf - (now - last)
        if sleep > 0:
            time.sleep(sleep)
        last = time.perf_counter()


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
            cooldown = 15
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
