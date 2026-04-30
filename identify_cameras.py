"""Snap one frame from every available camera index and save as PNG.

Usage:
    uv run python identify_cameras.py

Produces camera_snaps/cam_0.png, cam_1.png, ... — open them to see which
physical camera each index maps to, then pass the right numbers to
run_system_groot_n17.py via --camera-index / --robot-camera-index /
--wrist-camera-index.
"""
from __future__ import annotations

import os

import cv2

OUT_DIR = "camera_snaps"
MAX_INDEX = 6


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    found = []
    for i in range(MAX_INDEX):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            print(f"index {i}: not available")
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Some cams need a warm-up read
        for _ in range(3):
            cap.read()
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"index {i}: opened but no frame")
            continue
        path = os.path.join(OUT_DIR, f"cam_{i}.png")
        cv2.imwrite(path, frame)
        print(f"index {i}: saved {path}  shape={frame.shape}")
        found.append(i)

    print()
    print(f"Saved {len(found)} camera(s): {found}")
    print(f"Open the PNGs in {OUT_DIR}/ to identify which index is which.")


if __name__ == "__main__":
    main()
