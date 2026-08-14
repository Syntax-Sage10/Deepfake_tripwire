import os
import librosa
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"

MIN_VALIDATED_DURATION = 2.5
MAX_VALIDATED_DURATION = 13.0


class NeuralVoiceTripwire:
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[NeuralVoiceTripwire] Loading {model_name} on {self.device} ...")
        self.model = AutoModelForAudioClassification.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("[NeuralVoiceTripwire] Model loaded.")

    def analyze(self, file_path: str, verbose: bool = True) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found at path: {file_path}")

        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        duration = float(librosa.get_duration(y=audio, sr=sr))

        if duration < 0.3:
            raise ValueError("Audio clip is too short to analyze (minimum 0.3s required).")

        duration_warning = None
        if duration < MIN_VALIDATED_DURATION or duration > MAX_VALIDATED_DURATION:
            duration_warning = (
                f"Clip is {duration:.1f}s; model was validated on "
                f"{MIN_VALIDATED_DURATION}-{MAX_VALIDATED_DURATION}s clips. "
                f"Result may be less reliable outside that range."
            )

        inputs = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        prob_real = float(probs[0][0].item())
        prob_fake = float(probs[0][1].item())
        is_deepfake = prob_fake > 0.5
        confidence = round(max(prob_real, prob_fake) * 100, 1)

        if verbose:
            print("\n" + "=" * 60)
            print(f" NEURAL DIAGNOSTIC RUN FOR: {os.path.basename(file_path)}")
            print("=" * 60)
            print(f" • Duration              : {duration:.2f}s")
            if duration_warning:
                print(f" • ⚠ {duration_warning}")
            print(f" • P(real)               : {prob_real:.4f}")
            print(f" • P(fake)               : {prob_fake:.4f}")
            print(f" • Verdict               : {'FAKE' if is_deepfake else 'REAL'}")
            print(f" • Confidence            : {confidence}%")
            print("=" * 60 + "\n")

        return {
            "duration_seconds": round(duration, 2),
            "verdict": "RED_SPOOF" if is_deepfake else "GREEN_HUMAN",
            "confidence_percent": confidence,
            "duration_warning": duration_warning,
            "mathematical_metrics": {
                "model": MODEL_NAME,
                "probability_real": f"{prob_real:.4f}",
                "probability_fake": f"{prob_fake:.4f}",
                "duration_seconds": f"{duration:.2f}",
            },
        }


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Run the neural voice detector on one or more files.")
    parser.add_argument("--real", nargs="*", default=[], help="Known REAL clips")
    parser.add_argument("--fake", nargs="*", default=[], help="Known FAKE clips")
    parser.add_argument("files", nargs="*", help="Unlabeled files")
    args = parser.parse_args()

    if not args.real and not args.fake and not args.files:
        parser.print_help()
        sys.exit(0)

    detector = NeuralVoiceTripwire()
    rows = []
    for label, paths in (("REAL", args.real), ("FAKE", args.fake), ("?", args.files)):
        for path in paths:
            try:
                result = detector.analyze(path, verbose=True)
                correct = ""
                if label in ("REAL", "FAKE"):
                    predicted = "REAL" if result["verdict"] == "GREEN_HUMAN" else "FAKE"
                    correct = "✓" if predicted == label else "✗ MISCLASSIFIED"
                rows.append((label, os.path.basename(path), result["verdict"], result["confidence_percent"], correct))
            except Exception as e:
                print(f"[!] Failed to analyze {path}: {e}")

    if rows:
        print(f"\n{'Label':<8}{'File':<30}{'Verdict':<14}{'Confidence':<12}{'Correct'}")
        print("-" * 80)
        for row in rows:
            print(f"{row[0]:<8}{row[1]:<30}{row[2]:<14}{str(row[3])+'%':<12}{row[4]}")