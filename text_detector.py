import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "yaya36095/xlm-roberta-text-detector"

MIN_WORDS_FOR_RELIABLE_READING = 25  # short snippets are noisy for this class of model

# Same rationale as FAKE_VERDICT_THRESHOLD in image_classifier.py: this is a
# small fine-tuned classifier and can run hot on real-world text that wasn't
# in its training distribution. 0.6 is a starting point, not a measured
# calibration - override with TRIPWIRE_TEXT_AI_THRESHOLD once you've seen how
# it scores your own known-human and known-AI samples.
AI_VERDICT_THRESHOLD = float(os.environ.get("TRIPWIRE_TEXT_AI_THRESHOLD", "0.6"))


class TextTripwire:
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TextTripwire] Loading {model_name} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        id2label = {k: str(v).lower() for k, v in (self.model.config.id2label or {}).items()}
        human_idx = next((i for i, l in id2label.items() if "human" in l), None)
        ai_idx = next((i for i, l in id2label.items() if "ai" in l or "machine" in l or "generated" in l), None)

        if human_idx is None or ai_idx is None:
            print(f"[TextTripwire] ⚠ Could not confidently resolve label mapping from "
                  f"config.id2label={self.model.config.id2label!r}; falling back to "
                  f"index 0=HUMAN, 1=AI per model card. Verify this against real samples.")
            human_idx, ai_idx = 0, 1

        self.human_idx = human_idx
        self.ai_idx = ai_idx
        print(f"[TextTripwire] Label mapping resolved: HUMAN=idx{human_idx}, AI=idx{ai_idx} "
              f"(raw id2label={self.model.config.id2label!r})")
        print("[TextTripwire] Model loaded.")

    def analyze(self, text: str, verbose: bool = True) -> dict:
        if not text or not text.strip():
            raise ValueError("No text provided to analyze.")

        text = text.strip()
        word_count = len(text.split())

        length_warning = None
        if word_count < MIN_WORDS_FOR_RELIABLE_READING:
            length_warning = (
                f"Only {word_count} words; readings under "
                f"{MIN_WORDS_FOR_RELIABLE_READING} words are considerably less reliable."
            )

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        prob_human = float(probs[0][self.human_idx].item())
        prob_ai = float(probs[0][self.ai_idx].item())
        is_ai = prob_ai > AI_VERDICT_THRESHOLD
        confidence = round(max(prob_human, prob_ai) * 100, 1)

        truncated = len(self.tokenizer.encode(text)) > 512

        if verbose:
            print("\n" + "=" * 60)
            print(" TEXT DIAGNOSTIC RUN")
            print("=" * 60)
            print(f" • Word count            : {word_count}")
            if length_warning:
                print(f" • ⚠ {length_warning}")
            if truncated:
                print(" • ⚠ Text truncated to first 512 tokens for analysis")
            print(f" • P(human)              : {prob_human:.4f}")
            print(f" • P(AI-generated)       : {prob_ai:.4f}")
            print(f" • Verdict               : {'AI' if is_ai else 'HUMAN'}")
            print(f" • Confidence            : {confidence}%")
            print("=" * 60 + "\n")

        return {
            "verdict": "RED_SPOOF" if is_ai else "GREEN_HUMAN",
            "confidence_percent": confidence,
            "length_warning": length_warning,
            "mathematical_metrics": {
                "model": MODEL_NAME,
                "word_count": str(word_count),
                "probability_human": f"{prob_human:.4f}",
                "probability_ai": f"{prob_ai:.4f}",
                "truncated_to_512_tokens": str(truncated),
                "ai_verdict_threshold": f"{AI_VERDICT_THRESHOLD:.2f}",
            },
        }


if __name__ == "__main__":
    import sys

    detector = TextTripwire()
    if len(sys.argv) > 1:
        sample = " ".join(sys.argv[1:])
    else:
        sample = input("Paste text to analyze: ")
    result = detector.analyze(sample)
    print(result)