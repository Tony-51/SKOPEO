import os
import time
import base64
import json
import requests
from dotenv import load_dotenv

load_dotenv()

FACECHECK_API_TOKEN = os.getenv("FACECHECK_API_TOKEN")
FACECHECK_TESTING_MODE = os.getenv("FACECHECK_TESTING_MODE", "false").lower() == "true"
FACECHECK_BASE_URL = "https://facecheck.id"


def _headers():
    if not FACECHECK_API_TOKEN:
        raise ValueError(
            "FACECHECK_API_TOKEN not found in .env. "
            "Create a FaceCheck API account and add the token to .env."
        )
    return {
        "accept": "application/json",
        "Authorization": FACECHECK_API_TOKEN,
    }


def search_by_face(image_path, timeout_seconds=20, poll_seconds=1):
    """Search the public web by face using FaceCheck's face-search API.

    This is intentionally not Google Lens and does not require a public image URL.
    Returned items contain the source page URL and an optional base64 thumbnail.
    """
    headers = _headers()

    with open(image_path, "rb") as image_file:
        response = requests.post(
            f"{FACECHECK_BASE_URL}/api/upload_pic",
            headers=headers,
            files={"images": image_file},
            data={"id_search": ""},
            timeout=10,
        )

    response.raise_for_status()
    upload = response.json()

    if upload.get("error"):
        raise RuntimeError(
            f"FaceCheck upload failed: {upload.get('error')} "
            f"({upload.get('code', 'unknown')})"
        )

    search_id = upload.get("id_search")
    if not search_id:
        raise RuntimeError(f"FaceCheck did not return id_search: {upload}")

    payload = {
        "id_search": search_id,
        "with_progress": True,
        "status_only": False,
        "demo": FACECHECK_TESTING_MODE,
    }

    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        remaining = max(0.5, deadline - time.time())
        response = requests.post(
            f"{FACECHECK_BASE_URL}/api/search",
            headers={**headers, "Content-Type": "application/json"},
            json=payload,
            timeout=min(8, remaining),
        )
        response.raise_for_status()
        data = response.json()

        if data.get("error"):
            raise RuntimeError(
                f"FaceCheck search failed: {data.get('error')} "
                f"({data.get('code', 'unknown')})"
            )

        if data.get("output"):
            items = data["output"].get("items", [])
            return normalize_results(items)

        time.sleep(poll_seconds)

    raise TimeoutError("FaceCheck search timed out before results were ready.")


def normalize_results(items):
    """Normalize FaceCheck results to the candidate format used by face_matcher."""
    candidates = []

    for item in items or []:
        url = item.get("url")
        if not url:
            continue

        candidates.append({
            "title": "FaceCheck public-web face match",
            "source": "FaceCheck",
            "link": url,
            "image_base64": item.get("base64"),
            "face_score": item.get("score"),
            "guid": item.get("guid"),
            "index": item.get("index"),
        })

    return candidates


def save_results(results, path="data/face_search_candidates.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return path
