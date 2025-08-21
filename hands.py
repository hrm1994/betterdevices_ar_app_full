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
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=1
        )

    def process(self, frame_bgr):
        import cv2
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None
        lm = results.multi_hand_landmarks[0].landmark
        idx_tip = (lm[8].x*w, lm[8].y*h)
        idx_mcp = (lm[5].x*w, lm[5].y*h)
        thb_tip = (lm[4].x*w, lm[4].y*h)
        wrist = (lm[0].x*w, lm[0].y*h)
        diag = (w**2 + h**2) ** 0.5
        d = ((idx_tip[0]-thb_tip[0])**2 + (idx_tip[1]-thb_tip[1])**2) ** 0.5
        pinch = (d/diag) < 0.05
        return HandState(index_tip=idx_tip, index_mcp=idx_mcp, thumb_tip=thb_tip, wrist=wrist, pinch=pinch)

    def draw(self, frame_bgr):
        pass
