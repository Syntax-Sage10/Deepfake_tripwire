import os
import cv2
from PIL import Image

from image_classifier import FakeImageClassifier

DEFAULT_NUM_FRAMES = 8
MIN_FRAMES_FOR_CONFIDENCE = 3


class VideoTripwire:
    def __init__(self, classifier: FakeImageClassifier = None):
        self.classifier = classifier or FakeImageClassifier()

    def _extract_frames(self, file_path: str, num_frames: int):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(
                "Could not open video file. The codec/container may be unsupported by "
                "OpenCV's FFmpeg build - try re-encoding to .mp4."
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        duration = (total_frames / fps) if fps else None

        if total_frames <= 0:
            cap.release()
            raise ValueError("Video reports zero readable frames.")

        indices = [
            int(total_frames * (i + 1) / (num_frames + 1)) for i in range(num_frames)
        ]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame_bgr = cap.read()
            if ok:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

        cap.release()
        return frames, duration, total_frames

    def analyze(self, file_path: str, num_frames: int = DEFAULT_NUM_FRAMES, verbose: bool = True) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found at path: {file_path}")

        frames, duration, total_frames = self._extract_frames(file_path, num_frames)

        if len(frames) == 0:
            raise ValueError("Could not extract any readable frames from this video.")

        frame_warning = None
        if len(frames) < MIN_FRAMES_FOR_CONFIDENCE:
            frame_warning = (
                f"Only {len(frames)} frame(s) could be read; result is low-confidence."
            )

        fake_scores = [self.classifier.score(frame) for frame in frames]

        # Timestamps (seconds) for each sampled frame, used to plot a
        # per-frame timeline graph on the frontend. Falls back to a simple
        # 0..1 fractional position if we don't have a reliable duration.
        if duration:
            sampled_positions = [
                int(total_frames * (i + 1) / (len(frames) + 1)) for i in range(len(frames))
            ]
            fps_est = (total_frames / duration) if duration else 0
            timestamps = [round(idx / fps_est, 2) if fps_est else None for idx in sampled_positions]
        else:
            timestamps = [None] * len(frames)

        avg_fake = sum(fake_scores) / len(fake_scores)
        max_fake = max(fake_scores)
        flagged_frame_count = sum(1 for s in fake_scores if s > 0.5)
        # Index of the single most-suspicious sampled frame, so the frontend
        # can request/display a Grad-CAM heatmap for that frame specifically
        # rather than an arbitrary one.
        peak_frame_index = int(max(range(len(fake_scores)), key=lambda i: fake_scores[i]))

        # Verdict on the average across sampled frames rather than any single
        # frame, so one odd frame (motion blur, lighting) doesn't flip the result.
        is_deepfake = avg_fake > 0.5
        confidence = round(max(avg_fake, 1 - avg_fake) * 100, 1)

        if verbose:
            print("\n" + "=" * 60)
            print(f" VIDEO DIAGNOSTIC RUN FOR: {os.path.basename(file_path)}")
            print("=" * 60)
            print(f" • Duration              : {f'{duration:.1f}s' if duration else 'unknown'}")
            print(f" • Total frames in file  : {total_frames}")
            print(f" • Frames sampled        : {len(frames)}")
            if frame_warning:
                print(f" • ⚠ {frame_warning}")
            print(f" • Per-frame fake scores : {[round(s, 3) for s in fake_scores]}")
            print(f" • Avg fake probability  : {avg_fake:.4f}")
            print(f" • Frames flagged fake   : {flagged_frame_count}/{len(frames)}")
            print(f" • Verdict               : {'FAKE' if is_deepfake else 'REAL'}")
            print(f" • Confidence            : {confidence}%")
            print("=" * 60 + "\n")

        return {
            "duration_seconds": round(duration, 2) if duration else None,
            "verdict": "RED_SPOOF" if is_deepfake else "GREEN_HUMAN",
            "confidence_percent": confidence,
            "frame_warning": frame_warning,
            # Per-frame scoring timeline, for the frontend's timeline graph.
            "frame_timeline": {
                "timestamps_seconds": timestamps,
                "fake_scores": [round(s, 4) for s in fake_scores],
                "peak_frame_index": peak_frame_index,
            },
            "peak_frame_image": frames[peak_frame_index] if frames else None,
            "mathematical_metrics": {
                "model": self.classifier.model.name_or_path,
                "frames_sampled": str(len(frames)),
                "frames_flagged_fake": f"{flagged_frame_count}/{len(frames)}",
                "avg_fake_probability": f"{avg_fake:.4f}",
                "max_fake_probability": f"{max_fake:.4f}",
            },
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python video_detector.py <path-to-video> [num_frames]")
        sys.exit(0)
    detector = VideoTripwire()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_NUM_FRAMES
    result = detector.analyze(sys.argv[1], num_frames=n)
    print(result)