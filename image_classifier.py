"""Model: prithivMLmods/Deepfake-Detect-Siglip2 (Apache-2.0)
https://huggingface.co/prithivMLmods/Deepfake-Detect-Siglip2"""


import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

MODEL_NAME = "prithivMLmods/Deepfake-Detect-Siglip2"


class FakeImageClassifier:
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[FakeImageClassifier] Loading {model_name} on {self.device} ...")
        self.model = SiglipForImageClassification.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        # id2label from the model config; fall back to documented mapping if missing
        self.id2label = self.model.config.id2label or {0: "Fake", 1: "Real"}
        self.fake_idx = next((i for i, l in self.id2label.items() if str(l).lower() == "fake"), 0)
        print("[FakeImageClassifier] Model loaded.")

    def score(self, pil_image) -> float:
        """Returns P(fake) for a single PIL image, in [0, 1]."""
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()
        return float(probs[self.fake_idx])

    def score_and_heatmap(self, pil_image, target: str = "fake"):
        target_idx = self.fake_idx if target == "fake" else (1 - self.fake_idx)

        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        vision_model = self.model.vision_model
        captured = {}

        def _capture_hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            hidden.retain_grad()
            captured["patch_tokens"] = hidden

        # Hook the final encoder layer so we capture patch-token embeddings
        # right before the post-layernorm/pooling head.
        last_layer = vision_model.encoder.layers[-1]
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

            grads = patch_tokens.grad[0]        # (num_patches, hidden_dim)
            activations = patch_tokens[0].detach()  # (num_patches, hidden_dim)

            weights = grads.mean(dim=-1)        # (num_patches,) - Grad-CAM channel weights, GAP'd
            cam = (weights.unsqueeze(-1) * activations).sum(dim=-1)  # (num_patches,)
            cam = torch.relu(cam)

            num_patches = cam.shape[0]
            side = int(round(num_patches ** 0.5))
            if side * side != num_patches:
                # Non-square patch grid (unexpected for this model) - bail
                # out gracefully rather than reshape into something wrong.
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

        # Simple red->yellow ramp (low->high) instead of pulling in a full
        # colormap dependency like matplotlib.
        r = np.full_like(gray, 255)
        g = (gray * 255).astype(np.uint8)
        b = np.zeros_like(gray, dtype=np.uint8)
        a = (gray * 200).astype(np.uint8)  # fades out where activation is low

        rgba = np.stack([r.astype(np.uint8), g, b, a], axis=-1)
        return Image.fromarray(rgba, mode="RGBA")