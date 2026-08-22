import json
import os
import secrets
import time
import threading

STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared_cases.json")
_lock = threading.Lock()

# Shared links expire after this long so old cases don't accumulate forever
# in the store file. Purely a cleanup convenience, not a security control.
DEFAULT_TTL_SECONDS = 30 * 24 * 60 * 60  # 30 days


def _load() -> dict:
    if not os.path.exists(STORE_PATH):
        return {}
    try:
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save(data: dict):
    tmp_path = STORE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp_path, STORE_PATH)


def _purge_expired(data: dict) -> dict:
    now = time.time()
    return {k: v for k, v in data.items() if v.get("expires_at", now + 1) > now}


def create_share(case_payload: dict) -> str:
    """Stores a case's result payload and returns a short, URL-safe id."""
    share_id = secrets.token_urlsafe(6)
    with _lock:
        data = _purge_expired(_load())
        data[share_id] = {
            "payload": case_payload,
            "created_at": time.time(),
            "expires_at": time.time() + DEFAULT_TTL_SECONDS,
        }
        _save(data)
    return share_id


def get_share(share_id: str):
    with _lock:
        data = _purge_expired(_load())
        entry = data.get(share_id)
    return entry["payload"] if entry else None