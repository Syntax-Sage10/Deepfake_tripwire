import os
from PIL import Image, ExifTags

C2PA_JUMBF_SIGNATURE = b"jumb"
C2PA_TYPE_SIGNATURE = b"c2pa"

GENERATIVE_TOOL_HINTS = [
    "midjourney", "dall-e", "dalle", "stable diffusion", "firefly",
    "runway", "sora", "gemini", "imagen", "leonardo.ai", "ideogram",
]


def _decode_exif(pil_image: Image.Image) -> dict:
    exif_raw = pil_image.getexif()
    if not exif_raw:
        return {}
    decoded = {}
    for tag_id, value in exif_raw.items():
        tag = ExifTags.TAGS.get(tag_id, str(tag_id))
        if isinstance(value, bytes):
            try:
                value = value.decode(errors="replace")
            except Exception:
                value = repr(value)
        decoded[tag] = value
    return decoded


def _scan_for_c2pa_signature(file_path: str, chunk_size: int = 1_000_000) -> bool:
    try:
        with open(file_path, "rb") as f:
            head = f.read(chunk_size)
            if C2PA_JUMBF_SIGNATURE in head or C2PA_TYPE_SIGNATURE in head:
                return True
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size > chunk_size:
                f.seek(max(0, size - chunk_size))
                tail = f.read(chunk_size)
                if C2PA_JUMBF_SIGNATURE in tail or C2PA_TYPE_SIGNATURE in tail:
                    return True
    except OSError:
        pass
    return False


def check_image_provenance(file_path: str, pil_image: Image.Image = None, verbose: bool = True) -> dict:
    if pil_image is None:
        pil_image = Image.open(file_path)

    exif = _decode_exif(pil_image)
    software = str(exif.get("Software", "")) or None
    has_gps = any(k for k in exif.keys() if "GPS" in str(k))
    has_datetime = "DateTime" in exif or "DateTimeOriginal" in exif

    generative_hint = None
    if software:
        low = software.lower()
        for hint in GENERATIVE_TOOL_HINTS:
            if hint in low:
                generative_hint = hint
                break

    c2pa_present = _scan_for_c2pa_signature(file_path)

    findings = []
    if not exif:
        findings.append("No EXIF metadata found - common for screenshots, downloads/re-saves, or stripped images.")
    if software:
        findings.append(f"Software tag present: '{software}'.")
    if generative_hint:
        findings.append(f"Software tag references a known generative tool ('{generative_hint}').")
    if has_gps:
        findings.append("GPS location metadata present.")
    if has_datetime:
        findings.append("Original capture timestamp present.")
    findings.append(
        f"C2PA/Content-Credentials manifest signature {'detected' if c2pa_present else 'not detected'} in file "
        f"(presence check only - not a signature verification)."
    )

    if verbose:
        print("\n" + "=" * 60)
        print(" PROVENANCE / METADATA CHECK")
        print("=" * 60)
        for line in findings:
            print(f" • {line}")
        print("=" * 60 + "\n")

    return {
        "exif_present": bool(exif),
        "exif_fields": {str(k): str(v) for k, v in exif.items()} if exif else {},
        "software_tag": software,
        "generative_tool_hint": generative_hint,
        "gps_present": has_gps,
        "datetime_present": has_datetime,
        "c2pa_manifest_detected": c2pa_present,
        "findings": findings,
    }