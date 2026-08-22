import os
from PIL import Image

from image_classifier import FakeImageClassifier, FAKE_VERDICT_THRESHOLD


class ImageTripwire:
    def __init__(self, classifier: FakeImageClassifier = None):
        self.classifier = classifier or FakeImageClassifier()

    def analyze(self, file_path: str, verbose: bool = True) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found at path: {file_path}")

        try:
            image = Image.open(file_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Could not read image file: {e}")

        width, height = image.size
        size_warning = None
        if min(width, height) < 64:
            size_warning = f"Image is very small ({width}x{height}); result may be unreliable."

        fake_prob = self.classifier.score(image)
        is_deepfake = fake_prob > FAKE_VERDICT_THRESHOLD
        confidence = round(max(fake_prob, 1 - fake_prob) * 100, 1)

        if verbose:
            print("\n" + "=" * 60)
            print(f" IMAGE DIAGNOSTIC RUN FOR: {os.path.basename(file_path)}")
            print("=" * 60)
            print(f" • Dimensions            : {width}x{height}")
            if size_warning:
                print(f" • ⚠ {size_warning}")
            print(f" • P(fake)               : {fake_prob:.4f}")
            print(f" • Verdict               : {'FAKE' if is_deepfake else 'REAL'}")
            print(f" • Confidence            : {confidence}%")
            print("=" * 60 + "\n")

        return {
            "verdict": "RED_SPOOF" if is_deepfake else "GREEN_HUMAN",
            "confidence_percent": confidence,
            "size_warning": size_warning,
            "mathematical_metrics": {
                "model": self.classifier.model.name_or_path,
                "dimensions": f"{width}x{height}",
                "probability_fake": f"{fake_prob:.4f}",
                "probability_real": f"{1 - fake_prob:.4f}",
                "fake_verdict_threshold": f"{FAKE_VERDICT_THRESHOLD:.2f}",
            },
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python image_detector.py <path-to-image>")
        sys.exit(0)
    detector = ImageTripwire()
    result = detector.analyze(sys.argv[1])
    print(result)