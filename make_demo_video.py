# make_demo_video.py
# Usage: python make_demo_video.py --orig Images/blurred_face.jpg --enh output/enhanced.png --out output/demo.mp4

import cv2
import numpy as np
import argparse
import os

def make_side_by_side(orig_path, enh_path, out_video, duration_sec=4, fps=20):
    orig = cv2.imread(orig_path)
    enh = cv2.imread(enh_path)
    if orig is None or enh is None:
        print("Error reading images.")
        return

    # Resize both to same height
    h = 480
    def resize_by_height(img, target_h):
        scale = target_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1]*scale), target_h))
    orig_r = resize_by_height(orig, h)
    enh_r = resize_by_height(enh, h)

    # Make side-by-side canvas
    gap = 10
    canvas_h = h
    canvas_w = orig_r.shape[1] + gap + enh_r.shape[1]
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8) + 255
    canvas[:, :orig_r.shape[1]] = orig_r
    canvas[:, orig_r.shape[1]+gap:orig_r.shape[1]+gap+enh_r.shape[1]] = enh_r

    # Add text labels
    cv2.putText(canvas, "Original", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Enhanced", (orig_r.shape[1]+gap+20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2, cv2.LINE_AA)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video, fourcc, fps, (canvas_w, canvas_h))
    frames = int(duration_sec * fps)
    for i in range(frames):
        writer.write(canvas)
    writer.release()
    print("Demo video saved to:", out_video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig", required=True)
    parser.add_argument("--enh", required=True)
    parser.add_argument("--out", default="output/demo.mp4")
    parser.add_argument("--duration", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    make_side_by_side(args.orig, args.enh, args.out, duration_sec=args.duration)





