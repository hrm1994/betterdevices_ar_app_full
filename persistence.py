import cv2
import numpy as np

class BackgroundTracker:
    def __init__(self):
        self.prev_gray = None
        self.prev_kp = None
        self.prev_desc = None
        self.detector = cv2.ORB_create(500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def estimate_transform(self, frame_gray):
        if self.prev_gray is None:
            self.prev_gray = frame_gray.copy()
            self.prev_kp, self.prev_desc = self.detector.detectAndCompute(self.prev_gray, None)
            return np.eye(3, dtype=np.float32)
        kp, desc = self.detector.detectAndCompute(frame_gray, None)
        if desc is None or self.prev_desc is None or len(desc) < 4 or len(self.prev_desc) < 4:
            self.prev_gray = frame_gray.copy()
            self.prev_kp, self.prev_desc = kp, desc
            return np.eye(3, dtype=np.float32)
        matches = self.matcher.match(self.prev_desc, desc)
        if len(matches) < 8:
            H = np.eye(3, dtype=np.float32)
        else:
            src = np.float32([self.prev_kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            dst = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
            H, mask = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if H is None:
                H = np.eye(2,3, dtype=np.float32)
            H = np.vstack([H, [0,0,1]]).astype(np.float32)
        self.prev_gray = frame_gray.copy()
        self.prev_kp, self.prev_desc = kp, desc
        return H
