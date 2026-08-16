import os
os.environ.setdefault(
    "HF_HOME",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache"),
)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import sys
import time
import shutil
import base64
import subprocess
import threading
import uuid
import webbrowser
from io import BytesIO

from flask import Flask, render_template, request, jsonify, abort
from werkzeug.utils import secure_filename
from PIL import Image

from deepfake_detector import NeuralVoiceTripwire
from image_classifier import FakeImageClassifier
from video_detector import VideoTripwire
from image_detector import ImageTripwire
from text_detector import TextTripwire
from face_detector import MultiFaceAnalyzer
from provenance import check_image_provenance

from test import ComprehensiveVoiceTripwire
import shared_store

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("TRIPWIRE_MAX_UPLOAD_MB", "100")) * 1024 * 1024

audio_analyzer = NeuralVoiceTripwire()
signal_voice_analyzer = ComprehensiveVoiceTripwire()

_shared_image_classifier = FakeImageClassifier()
video_analyzer = VideoTripwire(classifier=_shared_image_classifier)
image_analyzer = ImageTripwire(classifier=_shared_image_classifier)
face_analyzer = MultiFaceAnalyzer(classifier=_shared_image_classifier)

text_analyzer = TextTripwire()

TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

TEMP_VIDEO_DIR = "temp_video"
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)

TEMP_IMAGE_DIR = "temp_image"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)


def _verdict_payload(results, extra: dict = None):
    is_fake = results["verdict"] == "RED_SPOOF"
    confidence = results["confidence_percent"]

    if is_fake:
        message = f"🔴 {confidence}% likely AI-generated"
    else:
        message = f"🟢 {confidence}% likely human / authentic"

    warning = (
        results.get("duration_warning")
        or results.get("frame_warning")
        or results.get("length_warning")
        or results.get("size_warning")
    )
    if warning:
        message += f" (⚠ {warning})"

    payload = {
        "is_deepfake": is_fake,
        "confidence": confidence,
        "raw_result": results["mathematical_metrics"],
        "message": message,
    }
    if extra:
        payload.update(extra)

    ensemble_extra = results.get("_ensemble_extra")
    if ensemble_extra:
        payload.update(ensemble_extra)

    return payload


def _verdict_response(results):
    return jsonify(_verdict_payload(results))


def _pil_to_base64_png(pil_image) -> str:
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _handle_file_upload(temp_dir, prefix, default_ext, analyzer_fn):
    if 'file' not in request.files:
        return jsonify({"error": "No file received"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty file submitted"}), 400

    original_filename = secure_filename(file.filename)
    ext = os.path.splitext(original_filename)[1] or default_ext
    unique_filename = f"{prefix}_{uuid.uuid4().hex}{ext}"
    temp_file_path = os.path.join(temp_dir, unique_filename)
    file.save(temp_file_path)

    try:
        results = analyzer_fn(temp_file_path)
        return _verdict_response(results)
    except Exception as e:
        print(f"\n[!] {prefix} backend error: {str(e)}\n")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def _audio_ensemble_analyze(file_path):
    neural_result = audio_analyzer.analyze(file_path, verbose=True)

    ensemble_extra = {}
    try:
        signal_result = signal_voice_analyzer.analyze(file_path, verbose=True)
    except Exception as e:

        print(f"[!] signal-based second opinion failed (non-fatal): {e}")
        signal_result = None

    if signal_result is not None:
        agrees = (neural_result["verdict"] == signal_result["verdict"])
        ensemble_extra["second_opinion"] = {
            "source": "signal_analysis",
            "description": "Independent jitter/shimmer/HNR/phase/spectral-flatness analysis (test.py)",
            "verdict": signal_result["verdict"],
            "confidence_percent": signal_result["confidence_percent"],
            "detected_anomalies": signal_result["detected_anomalies"],
            "agrees_with_primary": agrees,
        }
        if not agrees:
            ensemble_extra["ensemble_warning"] = (
                f"Neural model says {neural_result['verdict']} but the independent "
                f"signal-based analysis says {signal_result['verdict']} - "
                "treat this result with extra caution."
            )

    neural_result["_ensemble_extra"] = ensemble_extra
    return neural_result


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_audio():
    return _handle_file_upload(TEMP_DIR, "audio", ".webm", _audio_ensemble_analyze)


@app.route("/analyze-video", methods=["POST"])
def analyze_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file received"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty file submitted"}), 400

    original_filename = secure_filename(file.filename)
    ext = os.path.splitext(original_filename)[1] or ".mp4"
    temp_file_path = os.path.join(TEMP_VIDEO_DIR, f"video_{uuid.uuid4().hex}{ext}")
    file.save(temp_file_path)

    try:
        results = video_analyzer.analyze(temp_file_path)
        peak_frame = results.pop("peak_frame_image", None)

        extra = {"frame_timeline": results.get("frame_timeline")}
        if peak_frame is not None:
            try:
                _, heatmap = _shared_image_classifier.score_and_heatmap(peak_frame)
                if heatmap is not None:
                    extra["heatmap_overlay"] = _pil_to_base64_png(heatmap)
                extra["peak_frame_preview"] = _pil_to_base64_png(peak_frame)
                extra["peak_frame_faces"] = face_analyzer.analyze(peak_frame, verbose=False)
            except Exception as e:
                print(f"[!] video heatmap/face pass failed (non-fatal): {e}")

        return jsonify(_verdict_payload(results, extra))
    except Exception as e:
        print(f"\n[!] video backend error: {str(e)}\n")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file received"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty file submitted"}), 400

    original_filename = secure_filename(file.filename)
    ext = os.path.splitext(original_filename)[1] or ".png"
    temp_file_path = os.path.join(TEMP_IMAGE_DIR, f"image_{uuid.uuid4().hex}{ext}")
    file.save(temp_file_path)

    try:
        results = image_analyzer.analyze(temp_file_path)

        extra = {}
        try:
            pil_image = Image.open(temp_file_path).convert("RGB")
            extra["image_preview"] = _pil_to_base64_png(pil_image)

            _, heatmap = _shared_image_classifier.score_and_heatmap(pil_image)
            if heatmap is not None:
                extra["heatmap_overlay"] = _pil_to_base64_png(heatmap)

            extra["faces"] = face_analyzer.analyze(pil_image, verbose=False)
            extra["provenance"] = check_image_provenance(temp_file_path, pil_image, verbose=False)
        except Exception as e:
            print(f"[!] image heatmap/face/provenance pass failed (non-fatal): {e}")

        return jsonify(_verdict_payload(results, extra))
    except Exception as e:
        print(f"\n[!] image backend error: {str(e)}\n")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.route("/share", methods=["POST"])
def create_share_link():
    data = request.get_json(silent=True) or {}
    if not data.get("verdict") or "confidence" not in data:
        return jsonify({"error": "Missing case data to share"}), 400

    safe_payload = {k: v for k, v in data.items() if not (isinstance(v, str) and v.startswith("data:"))}

    share_id = shared_store.create_share(safe_payload)
    share_url = request.host_url.rstrip("/") + f"/r/{share_id}"
    return jsonify({"share_id": share_id, "share_url": share_url})


@app.route("/r/<share_id>", methods=["GET"])
def view_shared_case(share_id):
    payload = shared_store.get_share(share_id)
    if payload is None:
        abort(404)
    return render_template("shared_result.html", case=payload)


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"})


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/r/"):
        return render_template("shared_result.html", case=None), 404
    return e, 404


@app.errorhandler(413)
def file_too_large(e):
    max_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    return jsonify({"error": f"File exceeds the server's {max_mb}MB upload limit."}), 413


@app.route("/analyze-text", methods=["POST"])
def analyze_text():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text submitted"}), 400

    try:
        results = text_analyzer.analyze(text)
        return _verdict_response(results)
    except Exception as e:
        print(f"\n[!] Text backend error: {str(e)}\n")
        return jsonify({"error": str(e)}), 500


APP_URL = "http://127.0.0.1:5000/?source=app"


def _find_chromium_browser():
    if sys.platform.startswith("win"):
        bases = [
            os.environ.get("PROGRAMFILES", r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
            os.environ.get("LOCALAPPDATA", ""),
        ]
        rel_paths = [
            r"Google\Chrome\Application\chrome.exe",
            r"Microsoft\Edge\Application\msedge.exe",
            r"Chromium\Application\chrome.exe",
        ]
        candidates = [os.path.join(base, rel) for base in bases for rel in rel_paths if base]
    elif sys.platform == "darwin":
        candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
    else:
        candidates = [
            shutil.which(name)
            for name in ("google-chrome", "google-chrome-stable", "chromium-browser", "chromium", "microsoft-edge")
        ]

    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def launch_app_window(url, delay=1.5):
    def _open():
        time.sleep(delay)
        browser_path = _find_chromium_browser()
        if browser_path:
            try:
                subprocess.Popen([browser_path, f"--app={url}", "--window-size=480,900"])
                return
            except Exception as e:
                print(f"[!] Could not open app-mode window ({e}); falling back to a normal browser tab.")
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()

DEBUG_MODE = os.environ.get("TRIPWIRE_DEBUG", "0") == "1"

if __name__ == "__main__":
    launch_app_window(APP_URL)
    app.run(host="127.0.0.1", port=5000, debug=DEBUG_MODE, use_reloader=False)