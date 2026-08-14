import os
import cv2
import numpy as np
from PIL import Image

from image_classifier import FakeImageClassifier

# Faces smaller than this (in px, on the longest side of the crop's bbox)
# are usually too low-resolution for the classifier to say anything
# meaningful about, so we still report them but flag low confidence.
MIN_RELIABLE_FACE_PX = 60

# Pad each detected face box by this fraction on every side before cropping,
# so the classifier sees a bit of context (hairline, jaw) instead of a
# tight crop straight to the eyes/nose/mouth bounding box.
FACE_PADDING_FRACTION = 0.35


class MultiFaceAnalyzer:
    def __init__(self, classifier: FakeImageClassifier = None):
        self.classifier = classifier or FakeImageClassifier()
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")

    def detect_faces(self, pil_image: Image.Image):
        """Returns a list of (x, y, w, h) boxes in the image's own pixel
        coordinates, largest-first."""
        rgb = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        boxes = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        boxes = sorted(boxes.tolist() if len(boxes) else [], key=lambda b: b[2] * b[3], reverse=True)
        return [tuple(int(v) for v in b) for b in boxes]

    def _crop_with_padding(self, pil_image, box):
        x, y, w, h = box
        pad_x = int(w * FACE_PADDING_FRACTION)
        pad_y = int(h * FACE_PADDING_FRACTION)
        W, H = pil_image.size
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(W, x + w + pad_x)
        bottom = min(H, y + h + pad_y)
        return pil_image.crop((left, top, right, bottom))

    def analyze(self, pil_image: Image.Image, verbose: bool = True) -> dict:
        boxes = self.detect_faces(pil_image)

        faces = []
        for i, box in enumerate(boxes):
            x, y, w, h = box
            crop = self._crop_with_padding(pil_image, box)
            fake_prob = self.classifier.score(crop)
            faces.append({
                "index": i,
                "box": {"x": x, "y": y, "w": w, "h": h},
                "probability_fake": round(fake_prob, 4),
                "verdict": "RED_SPOOF" if fake_prob > 0.5 else "GREEN_HUMAN",
                "confidence_percent": round(max(fake_prob, 1 - fake_prob) * 100, 1),
                "low_resolution_warning": min(w, h) < MIN_RELIABLE_FACE_PX,
            })

        if verbose:
            print("\n" + "=" * 60)
            print(f" MULTI-FACE SCAN: {len(faces)} face(s) detected")
            for f in faces:
                flag = " (low-res)" if f["low_resolution_warning"] else ""
                print(f"   face[{f['index']}] {f['verdict']} {f['confidence_percent']}%{flag}")
            print("=" * 60 + "\n")

        any_face_fake = any(f["verdict"] == "RED_SPOOF" for f in faces)

        return {
            "face_count": len(faces),
            "faces": faces,
            "any_face_flagged_fake": any_face_fake,
        }