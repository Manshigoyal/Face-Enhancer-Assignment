# enhance.py
# Usage: python enhance.py --input Images/blurred_face.jpg --output output/enhanced.png

import cv2
import numpy as np
import argparse
import os

def unsharp_mask(img, kernel_size=(5,5), sigma=1.0, amount=1.2, threshold=0):
    """Unsharp mask sharpening."""
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(img - blurred) < threshold
        np.copyto(sharpened, img, where=low_contrast_mask)
    return sharpened

def enhance_image_color(input_bgr):
    """Enhancement pipeline:
       - Convert to YCrCb, apply CLAHE on Y (contrast)
       - Convert back, then apply unsharp mask and bilateral denoise
       - Final mild color boost and gamma correction
    """
    # Convert to YCrCb and apply CLAHE on luminance channel
    ycrcb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y2 = clahe.apply(y)
    ycrcb2 = cv2.merge([y2, cr, cb])
    img_clahe = cv2.cvtColor(ycrcb2, cv2.COLOR_YCrCb2BGR)

    # Denoise a bit (preserve edges)
    img_denoised = cv2.bilateralFilter(img_clahe, d=9, sigmaColor=75, sigmaSpace=75)

    # Sharpen using unsharp mask
    sharpened = unsharp_mask(img_denoised, kernel_size=(5,5), sigma=1.0, amount=1.4, threshold=2)

    # Slight gamma correction to brighten midtones
    gamma = 1.05
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(sharpened, table)

    # Optional mild color boost in HSV space
    hsv = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] = hsv[...,1] * 1.05  # increase saturation a little
    hsv[...,1] = np.clip(hsv[...,1], 0, 255)
    final = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return final

def center_crop_to_face(img, min_size=100):
    # Attempt simple face detection to crop to face region (helps show improvement)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) == 0:
        return img  # no face found, return original
    # pick largest face
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    x, y, w, h = faces[0]
    # add margin
    margin = int(0.4 * max(w, h))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img.shape[1], x + w + margin)
    y2 = min(img.shape[0], y + h + margin)
    crop = img[y1:y2, x1:x2]
    return crop

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Path to input blurred image")
    parser.add_argument("--output", "-o", default="output/enhanced.png", help="Path to save enhanced image")
    parser.add_argument("--facecrop", action="store_true", help="Crop to detected face region before enhancing")
    return parser.parse_args()

def main():
    args = parse_args()
    inp = args.input
    outp = args.output
    os.makedirs(os.path.dirname(outp), exist_ok=True)

    img = cv2.imread(inp)
    if img is None:
        print("Error: Could not read input image:", inp)
        return

    # Optionally crop to face to focus enhancement
    if args.facecrop:
        img_proc = center_crop_to_face(img)
    else:
        img_proc = img.copy()

    enhanced = enhance_image_color(img_proc)

    # If we cropped, paste enhanced crop back into original size (optional)
    if args.facecrop and img_proc.shape != img.shape:
        # very simple: show enhanced crop separately by saving only the crop result
        cv2.imwrite(outp, enhanced)
    else:
        cv2.imwrite(outp, enhanced)

    print("Saved enhanced image to:", outp)

if __name__ == "__main__":
    main()
