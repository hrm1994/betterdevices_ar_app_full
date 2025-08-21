from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class HandState:
    index_tip: Tuple[float, float]
    index_mcp: Tuple[float, float]
    thumb_tip: Tuple[float, float]
    wrist: Tuple[float, float]
    pinch: bool



class HandTracker:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        pinch_threshold: Optional[float] = None,   # legacy API
        angle_threshold_deg: float = 165.0,
        pinch_on_scale: Optional[float] = None,
        pinch_off_scale: Optional[float] = None
    ):
        """
        MediaPipe Hands wrapper with robust pinch detection.

        - angle_threshold_deg: larger -> stricter about index being straight
        - pinch_on_scale / pinch_off_scale (hysteresis): preferred modern API
        - pinch_threshold: legacy single-threshold mode (falls back if scales not provided)
        """
        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=1
        )

        # --- Angle config ---
        self.angle_threshold_deg = float(angle_threshold_deg)

        # --- Pinch config ---
        if pinch_on_scale is not None and pinch_off_scale is not None:
            # modern hysteresis mode
            self.pinch_on_scale = float(pinch_on_scale)
            self.pinch_off_scale = float(pinch_off_scale)
        elif pinch_threshold is not None:
            # fallback: legacy single threshold
            self.pinch_on_scale = float(pinch_threshold)
            self.pinch_off_scale = float(pinch_threshold) * 1.3  # add hysteresis
        else:
            # defaults if nothing passed
            self.pinch_on_scale = 0.35
            self.pinch_off_scale = 0.45

        # --- State for hysteresis ---
        self._pinch_state = False


    def process(self, frame_bgr) -> Optional[HandState]:
        import cv2
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None

        lm = results.multi_hand_landmarks[0].landmark
        idx_tip = (lm[8].x * w, lm[8].y * h)
        idx_mcp = (lm[5].x * w, lm[5].y * h)
        thb_tip = (lm[4].x * w, lm[4].y * h)
        wrist   = (lm[0].x * w, lm[0].y * h)

        # --- Distances (normalize by a hand-sized metric, not frame diagonal) ---
        tip_dist   = np.linalg.norm(np.array(idx_tip) - np.array(thb_tip))
        hand_scale = np.linalg.norm(np.array(idx_mcp) - np.array(wrist)) + 1e-6  # proxy for size

        # --- Angle gate: index finger should be (nearly) straight ---
        v_index = np.array(idx_tip) - np.array(idx_mcp)
        v_palm  = np.array(wrist)   - np.array(idx_mcp)
        cosang = np.dot(v_index, v_palm) / (np.linalg.norm(v_index) * np.linalg.norm(v_palm) + 1e-6)
        cosang = np.clip(cosang, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cosang))
        index_extended = angle_deg > self.angle_threshold_deg

        # --- Hysteresis on distance threshold ---
        on_th  = self.pinch_on_scale  * hand_scale
        off_th = self.pinch_off_scale * hand_scale

        if index_extended:
            if not self._pinch_state and (tip_dist < on_th):
                self._pinch_state = True
            elif self._pinch_state and (tip_dist > off_th):
                self._pinch_state = False
        else:
            # if index not extended, force not-pinching
            self._pinch_state = False

        pinch = self._pinch_state

        return HandState(
            index_tip=idx_tip,
            index_mcp=idx_mcp,
            thumb_tip=thb_tip,
            wrist=wrist,
            pinch=pinch
        )

    def draw(self, frame_bgr):
        # (Optional) add landmark or debug drawing here later
        pass
