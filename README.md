# 2D AR Pointer & Pinch App (Full)

## Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate     # (Linux/macOS)
# .venv\Scripts\activate      # (Windows PowerShell)

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt




If `mediapipe` wheels complain:
```bash
pip install mediapipe
```
## Run App
python main.py --width 960 --height 540 --max-fps 60


## Controls
- `E` — Edit mode
- `I` — Interact mode
- Edit mode:
  - Left-drag: draw rectangle
  - `P`: polygon mode (click multiple vertices), press `Enter` to finish
  - `S` / `L`: save / load shapes
- Interact mode:
  - Point with **index finger** to highlight shape along the pointing **ray**
  - **Pinch** (thumb–index) to grab & move
  - Release pinch to drop
  - Hold **R** while grabbing to enable simple wrist-angle rotation (experimental)
- General
  - `O` — toggle persistence (world-lock shapes)
  - `Q` or `Esc` — Quit

## Notes
- Good lighting improves hand tracking.
- Reduce resolution via `--width 640 --height 360` if your CPU struggles.
