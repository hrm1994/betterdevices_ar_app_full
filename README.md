# ✋ 2D AR Pointer & Pinch App

An experimental **Augmented Reality (AR) 2D interaction app** using **hand tracking**.  
Draw shapes, select them with your **index finger ray**, grab them via **pinch gesture**, and move or rotate them in real time.  
Shapes can be kept **world-locked** to the background through persistence tracking.

⚡ Built with [OpenCV](https://opencv.org/) + [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker).

---

## ✨ Features
- 🖱️ **Edit Mode**: draw rectangles & polygons directly on the screen  
- ✋ **Interact Mode**: point, pinch, drag, and rotate shapes  
- 📷 **Cross-platform camera support** (macOS, Linux, Windows)  
- 📦 **Shape persistence**: shapes remain aligned with the scene while the camera moves  
- 💾 Save & load your shapes (`S` / `L`)  

---

## 📸 Demo
(*Add a GIF or screenshot here!*)  

---

## 🚀 Setup

### 1. Environment
```bash
python3 -m venv .venv
source .venv/bin/activate     # (Linux/macOS)
# .venv\Scripts\activate      # (Windows PowerShell)

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If `mediapipe` wheels complain:
```bash
pip install mediapipe
```

### 2. Run App
```bash
python main.py --width 960 --height 540 --max-fps 60
```

### 3. Platform Tips
- **Windows**  
  ```bash
  python main.py --backend dshow --mjpg
  ```
  (`--mjpg` can boost FPS by enabling MJPG capture)  

- **Linux**  
  ```bash
  python main.py --backend v4l2
  ```
  May require:  
  ```bash
  sudo apt install v4l-utils libgl1
  ```

- **macOS**  
  ```bash
  python main.py --backend avfoundation
  ```

### 4. Find Available Cameras
```bash
python main.py --list-cams
```

---

## 🎮 Controls

| Key          | Action                                                                 |
|--------------|------------------------------------------------------------------------|
| **E**        | Switch to **Edit mode**                                                |
| **I**        | Switch to **Interact mode**                                            |
| **Left Drag**| Draw a rectangle (Edit mode)                                           |
| **P**        | Polygon mode (click multiple vertices, press **Enter** to finish)      |
| **S** / **L**| Save / load shapes                                                     |
| **Index Finger** | Point at a shape (Interact mode)                                   |
| **Pinch**    | Grab & move selected shape                                             |
| Release pinch| Drop shape                                                             |
| **R** (hold) | Rotate grabbed shape via wrist rotation (experimental)                 |
| **O**        | Toggle persistence (keep shapes world-locked)                          |
| **Q** / Esc  | Quit                                                                   |

---

## 📝 Notes
- Good lighting improves hand tracking.  
- Lower resolution (e.g. `--width 640 --height 360`) helps on low-end CPUs.  
- Persistence uses background optical flow — works best with **textured, static backgrounds**.  

---

## ⚠️ Known Issues
- 🔄 **Persistence drift**: shapes may slowly drift if background has few visual features.  
- 📷 **Camera backend quirks**:  
  - macOS requires `avfoundation`  
  - Windows runs best with `dshow --mjpg`  
  - Linux needs `v4l2`  
- 🤚 **Pinch sensitivity** varies across lighting and camera quality.  

---

## 🤝 Contributing
Contributions are welcome!  
- File issues for bugs or feature requests.  


---

## 📜 License
MIT License — free to use, modify, and distribute.  

---



---
