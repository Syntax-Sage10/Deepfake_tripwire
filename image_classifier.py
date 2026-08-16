import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Swapped from Ateeqq/ai-vs-human-image-detector, which was giving very
# high-confidence "ai" scores on plain real-world photos (verified via the
# per-call score() logging below - label mapping was correct, the model
# itself was just unreliable on this kind of input). Organika/sdxl-detector
# is a widely-used, actively maintained fine-tune of umm-maybe/AI-image-detector
# aimed at general (non-artistic) imagery - re-verify its calibration the
# same way (watch the score() log against a few known-real photos) before
# trusting it blindly too.
MODEL_NAME = os.environ.get("TRIPWIRE_IMAGE_MODEL", "Organika/sdxl-detector")

FAKE_VERDICT_THRESHOLD = float(os.environ.get("TRIPWIRE_IMAGE_FAKE_THRESHOLD", "0.6"))

# If the model's label resolution below ever guesses wrong (see the warning
# it prints at startup), set this to force the correct index instead of
# waiting on a code fix - e.g. TRIPWIRE_IMAGE_FAKE_INDEX=1
_FORCE_FAKE_INDEX = os.environ.get("TRIPWIRE_IMAGE_FAKE_INDEX")

_FAKE_LABEL_SYNONYMS = {"fake", "ai", "ai-generated", "artificial", "synthetic", "generated", "deepfake"}
_REAL_LABEL_SYNONYMS = {"real", "human", "authentic", "genuine", "natural"}


class FakeImageClassifier:
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[FakeImageClassifier] Loading {model_name} on {self.device} ...")
        self.model = AutoModelForImageClassification.from_pretrained(model_name, local_files_only=False)
        self.processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=False)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label or {0: "Fake", 1: "Real"}

        if _FORCE_FAKE_INDEX is not None:
            self.fake_idx = int(_FORCE_FAKE_INDEX)
            print(f"[FakeImageClassifier] Fake index FORCED to {self.fake_idx} via "
                  f"TRIPWIRE_IMAGE_FAKE_INDEX (raw id2label={self.id2label!r})")
        else:
            fake_matches = [i for i, l in self.id2label.items() if str(l).lower() in _FAKE_LABEL_SYNONYMS]
            real_matches = [i for i, l in self.id2label.items() if str(l).lower() in _REAL_LABEL_SYNONYMS]

            if len(fake_matches) == 1:
                # Direct, unambiguous match on the "fake" label itself.
                self.fake_idx = fake_matches[0]
            elif len(real_matches) == 1 and len(self.id2label) == 2:
                # Couldn't match "fake" directly, but found exactly one
                # "real"-ish label in a binary classifier - fake must be
                # the other index. More reliable than defaulting to 0.
                self.fake_idx = next(i for i in self.id2label if i != real_matches[0])
            else:
                self.fake_idx = 0
                print(
                    f"[FakeImageClassifier] ⚠⚠⚠ COULD NOT CONFIDENTLY RESOLVE the 'fake' label from "
                    f"id2label={self.id2label!r}. Defaulting to index 0, which may be BACKWARDS - "
                    f"if every real photo/video is being flagged as fake, this is almost certainly why. "
                    f"To fix immediately: run a known-real image through image_detector.py, check whether "
                    f"'probability_fake' is high for it, and if so set TRIPWIRE_IMAGE_FAKE_INDEX=1 "
                    f"(or back to 0 if you'd already set it to 1) as an environment variable and restart."
                )

        print(f"[FakeImageClassifier] Label mapping resolved: fake=idx{self.fake_idx} "
              f"(raw id2label={self.id2label!r}) - probs will print per-image as "
              f"[FakeImageClassifier] score=... so you can sanity-check this against a known-real photo.")
        print("[FakeImageClassifier] Model loaded.")
        self._heatmap_unsupported_warned = False

    def score(self, pil_image) -> float:
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()
        labeled = {str(self.id2label.get(i, i)): round(p, 4) for i, p in enumerate(probs)}
        print(f"[FakeImageClassifier] score() raw probs by label={labeled!r} "
              f"-> using idx{self.fake_idx} as fake = {probs[self.fake_idx]:.4f}")
        return float(probs[self.fake_idx])

    def score_and_heatmap(self, pil_image, target: str = "fake"):
        target_idx = self.fake_idx if target == "fake" else (1 - self.fake_idx)

        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        try:
            # This hook path (vision_model.encoder.layers[-1]) matches
            # Siglip's internal module layout specifically. Other
            # architectures (Swin, ViT, ConvNeXT, etc.) name/nest their
            # layers differently, so this raises AttributeError/IndexError
            # for those - caught below rather than generalized, since a
            # missing heatmap is a cosmetic loss, not a correctness one.
            vision_model = self.model.vision_model
            last_layer = vision_model.encoder.layers[-1]
        except (AttributeError, IndexError):
            if not self._heatmap_unsupported_warned:
                print(f"[FakeImageClassifier] Grad-CAM heatmap not supported for this model's "
                      f"architecture ({type(self.model).__name__}) - verdicts/scores are unaffected, "
                      f"heatmap overlay will just be unavailable.")
                self._heatmap_unsupported_warned = True
            return self.score(pil_image), None

        captured = {}

        def _capture_hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            hidden.retain_grad()
            captured["patch_tokens"] = hidden

        handle = last_layer.register_forward_hook(_capture_hook)

        try:
            self.model.zero_grad(set_to_none=True)
            outputs = self.model(**inputs)
            logits = outputs.logits
            target_score = logits[0, target_idx]
            target_score.backward()

            patch_tokens = captured.get("patch_tokens")
            if patch_tokens is None or patch_tokens.grad is None:
                return self.score(pil_image), None

            grads = patch_tokens.grad[0]
            activations = patch_tokens[0].detach()

            weights = grads.mean(dim=-1)
            cam = (weights.unsqueeze(-1) * activations).sum(dim=-1)
            cam = torch.relu(cam)

            num_patches = cam.shape[0]
            side = int(round(num_patches ** 0.5))
            if side * side != num_patches:
                return self.score(pil_image), None

            cam = cam.reshape(side, side)
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            cam_np = cam.cpu().numpy().astype(np.float32)

            heatmap = self._colorize_heatmap(cam_np, pil_image.size)

            with torch.no_grad():
                probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()
            fake_prob = float(probs[self.fake_idx])

            return fake_prob, heatmap
        finally:
            handle.remove()
            self.model.zero_grad(set_to_none=True)

    @staticmethod
    def _colorize_heatmap(cam_np, output_size):
        small = Image.fromarray((cam_np * 255).astype(np.uint8), mode="L")
        small = small.resize(output_size, resample=Image.BICUBIC)
        gray = np.asarray(small).astype(np.float32) / 255.0
        gray = np.clip(gray, 0.0, 1.0)

        r = np.full_like(gray, 255)
        g = (gray * 255).astype(np.uint8)
        b = np.zeros_like(gray, dtype=np.uint8)
        a = (gray * 200).astype(np.uint8)

        rgba = np.stack([r.astype(np.uint8), g, b, a], axis=-1)
        return Image.fromarray(rgba, mode="RGBA")