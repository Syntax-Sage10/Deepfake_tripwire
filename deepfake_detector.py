import os
import librosa
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"

MIN_VALIDATED_DURATION = 2.5
MAX_VALIDATED_DURATION = 13.0

# Real-world clips (browser-mic recordings: compressed webm/opus, room
# noise, varied mic hardware) sit in a different distribution than the
# clean TTS-vs-human dataset this model was fine-tuned on, which tends to
# push P(fake) up even for genuine human speech. 0.5 is too aggressive for
# that shift, so default a bit higher and make it tunable like the other
# detectors' thresholds, instead of hardcoding it.
FAKE_VERDICT_THRESHOLD = float(os.environ.get("TRIPWIRE_AUDIO_FAKE_THRESHOLD", "0.75"))

# If label resolution below ever guesses wrong, force it here instead of
# waiting on a code fix - e.g. TRIPWIRE_AUDIO_FAKE_INDEX=0
_FORCE_FAKE_INDEX = os.environ.get("TRIPWIRE_AUDIO_FAKE_INDEX")

_FAKE_LABEL_SYNONYMS = {"fake", "ai", "ai-generated", "artificial", "synthetic", "generated", "deepfake", "spoof"}
_REAL_LABEL_SYNONYMS = {"real", "human", "authentic", "genuine", "natural", "bonafide", "bona-fide"}


class NeuralVoiceTripwire:
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[NeuralVoiceTripwire] Loading {model_name} on {self.device} ...")
        # NOTE: local_files_only=True means this will hard-fail with a
        # confusing error on any machine that hasn't already cached this
        # model once with internet access. Unlike the image/text detectors,
        # which load with local_files_only=False. If you're setting this up
        # somewhere fresh and it errors here, that's why - flip this to
        # False (or pre-download the model once) rather than assuming a bad
        # result is a silent failure.
        self.model = AutoModelForAudioClassification.from_pretrained(model_name, local_files_only=True)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

        id2label = {int(k): str(v) for k, v in (self.model.config.id2label or {0: "real", 1: "fake"}).items()}

        if _FORCE_FAKE_INDEX is not None:
            self.fake_idx = int(_FORCE_FAKE_INDEX)
            print(f"[NeuralVoiceTripwire] Fake index FORCED to {self.fake_idx} via "
                  f"TRIPWIRE_AUDIO_FAKE_INDEX (raw id2label={id2label!r})")
        else:
            fake_matches = [i for i, l in id2label.items() if l.lower() in _FAKE_LABEL_SYNONYMS]
            real_matches = [i for i, l in id2label.items() if l.lower() in _REAL_LABEL_SYNONYMS]

            if len(fake_matches) == 1:
                self.fake_idx = fake_matches[0]
            elif len(real_matches) == 1 and len(id2label) == 2:
                self.fake_idx = next(i for i in id2label if i != real_matches[0])
            else:
                # Model card sample code uses probs[0]=real, probs[0][1]=fake,
                # so 1 is a better blind default than 0 here specifically -
                # but this is still a guess, not a verified mapping.
                self.fake_idx = 1
                print(
                    f"[NeuralVoiceTripwire] ⚠⚠⚠ COULD NOT CONFIDENTLY RESOLVE the 'fake' label from "
                    f"id2label={id2label!r}. Defaulting to index {self.fake_idx}, which may be BACKWARDS - "
                    f"if every real recording is being flagged as fake, this is almost certainly why. "
                    f"To fix immediately: run a known-real clip through 'python deepfake_detector.py "
                    f"--real your_clip.wav', check whether probability_fake is high for it, and if so set "
                    f"TRIPWIRE_AUDIO_FAKE_INDEX=0 (or back to 1) as an environment variable and restart."
                )
        self.real_idx = 1 - self.fake_idx

        print(f"[NeuralVoiceTripwire] Label mapping resolved: fake=idx{self.fake_idx} "
              f"(raw id2label={id2label!r})")
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

        prob_real = float(probs[0][self.real_idx].item())
        prob_fake = float(probs[0][self.fake_idx].item())
        is_deepfake = prob_fake > FAKE_VERDICT_THRESHOLD
        confidence = round(max(prob_real, prob_fake) * 100, 1)

        print(f"[NeuralVoiceTripwire] analyze() prob_real(idx{self.real_idx})={prob_real:.4f} "
              f"prob_fake(idx{self.fake_idx})={prob_fake:.4f}")

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
                "fake_verdict_threshold": f"{FAKE_VERDICT_THRESHOLD:.2f}",
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