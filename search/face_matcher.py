import sys
import os
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import cv2
import numpy as np

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from face.detector import detect_faces
from face.encoder import generate_embedding


# --------------------------------------------------
# PERFORMANCE SETTINGS
# --------------------------------------------------

# Network is the largest variable in candidate processing. A short timeout
# prevents one dead image host from blocking the entire pipeline.
DOWNLOAD_CONNECT_TIMEOUT = 3
DOWNLOAD_READ_TIMEOUT = 5
DOWNLOAD_TIMEOUT = (DOWNLOAD_CONNECT_TIMEOUT, DOWNLOAD_READ_TIMEOUT)

# Keep this moderate: downloads are I/O-bound, but too many simultaneous
# connections can trigger throttling from public hosts.
DOWNLOAD_WORKERS = 12

# Do not feed enormous web images into YuNet/SFace.
MAX_IMAGE_SIDE = 1280

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0 Safari/537.36"
)


def cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two face embeddings."""
    a = embedding1.flatten().astype(np.float32)
    b = embedding2.flatten().astype(np.float32)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def get_image_url(candidate):
    """Get the best available candidate image URL."""
    return (
        candidate.get("image")
        or candidate.get("thumbnail")
    )


def _cache_path_for_url(url):
    """Stable local cache filename for a candidate URL."""
    digest = hashlib.sha256(url.encode("utf-8", errors="ignore")).hexdigest()
    return os.path.join("data", "candidates", f"{digest}.img")


def download_image(url, output_path=None):
    """
    Download an image with short network timeouts.

    Returns the downloaded bytes, or None on failure.
    If output_path is supplied, the bytes are also written there for
    compatibility with the existing project layout.
    """
    if not url:
        return None

    try:
        response = requests.get(
            url,
            timeout=DOWNLOAD_TIMEOUT,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            },
            allow_redirects=True,
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        content = response.content

        # Some hosts omit Content-Type. Let OpenCV validate those responses.
        if content_type and not content_type.startswith("image/"):
            return None

        if not content:
            return None

        if output_path:
            with open(output_path, "wb") as f:
                f.write(content)

        return content

    except requests.RequestException as exc:
        print(f"Download failed: {url[:120]} -> {exc}")
        return None
    except OSError as exc:
        print(f"Cache/write failed: {url[:120]} -> {exc}")
        return None


def _download_candidate(index, candidate):
    """
    Download one candidate.

    Returns (index, candidate, image_bytes).
    Existing URL cache is reused so repeated runs do not redownload the
    same image.
    """
    url = get_image_url(candidate)

    if not url:
        return index, candidate, None

    cache_path = _cache_path_for_url(url)

    try:
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            with open(cache_path, "rb") as f:
                return index, candidate, f.read()
    except OSError:
        pass

    data = download_image(url)

    if data:
        try:
            with open(cache_path, "wb") as f:
                f.write(data)
        except OSError:
            # Cache failure should never fail the candidate.
            pass

    return index, candidate, data


def _decode_image(image_bytes):
    """Decode bytes directly into an RGB OpenCV image."""
    if not image_bytes:
        return None

    try:
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        if image is None:
            return None

        height, width = image.shape[:2]

        # Resize only oversized images. This reduces YuNet/SFace CPU cost
        # without unnecessarily changing normal candidate images.
        largest_side = max(height, width)

        if largest_side > MAX_IMAGE_SIDE:
            scale = MAX_IMAGE_SIDE / float(largest_side)
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            image = cv2.resize(
                image,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA,
            )

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    except Exception as exc:
        print(f"Image decode failed: {exc}")
        return None


def _process_downloaded_candidate(index, candidate, image_bytes, input_embedding):
    """
    Run YuNet + SFace on one already-downloaded image.

    Model inference is intentionally kept in the main processing thread.
    Downloads are parallelized, while OpenCV inference remains stable and
    avoids creating many simultaneous ONNX sessions.
    """
    image_url = get_image_url(candidate)

    if not image_bytes:
        return None

    candidate_rgb = _decode_image(image_bytes)

    if candidate_rgb is None:
        return None

    candidate_faces = detect_faces(candidate_rgb)

    if len(candidate_faces) == 0:
        return None

    best_score = 0.0

    for face_box in candidate_faces:
        try:
            candidate_embedding = generate_embedding(
                candidate_rgb,
                face_box,
            )

            score = cosine_similarity(
                input_embedding,
                candidate_embedding,
            )

            best_score = max(best_score, score)

        except Exception as exc:
            print(
                f"[{index}] Face comparison failed: {exc}"
            )

    return {
        "rank": index,
        "title": candidate.get("title"),
        "source": candidate.get("source"),
        "link": candidate.get("link"),
        "image_url": image_url,
        "faces_detected": len(candidate_faces),
        "similarity": round(best_score, 4),
    }


def compare_faces(image_path, candidates, progress_callback=None):
    """
    Fast face matching pipeline.

    Important:
      * Crawler candidates are NOT removed.
      * All candidates passed by app.py are processed.
      * Network downloads happen concurrently.
      * SFace model is cached by face.encoder.
      * Images are decoded from memory.
      * Large images are resized before face detection.
      * Local URL cache avoids repeat downloads.
    """

    candidates = list(candidates or [])
    total = len(candidates)

    if total == 0:
        return []

    # --------------------------------------------------
    # 1. Read input image
    # --------------------------------------------------

    input_image = cv2.imread(image_path)

    if input_image is None:
        raise FileNotFoundError(
            f"Could not read {image_path}"
        )

    input_rgb = cv2.cvtColor(
        input_image,
        cv2.COLOR_BGR2RGB,
    )

    input_faces = detect_faces(input_rgb)

    if len(input_faces) == 0:
        raise RuntimeError(
            "No face detected in input image."
        )

    print(
        f"Input image: {len(input_faces)} face(s) detected."
    )

    # Cached SFace model means this is initialized once, not once per face.
    input_embedding = generate_embedding(
        input_rgb,
        input_faces[0],
    )

    os.makedirs(
        "data/candidates",
        exist_ok=True,
    )

    # --------------------------------------------------
    # 2. PARALLEL DOWNLOAD PHASE
    # --------------------------------------------------

    print(
        f"Downloading {total} candidate images "
        f"with {DOWNLOAD_WORKERS} workers..."
    )

    downloaded = {}
    completed_downloads = 0

    with ThreadPoolExecutor(
        max_workers=min(DOWNLOAD_WORKERS, max(1, total))
    ) as executor:

        futures = {
            executor.submit(
                _download_candidate,
                index,
                candidate,
            ): index
            for index, candidate in enumerate(candidates, start=1)
        }

        for future in as_completed(futures):
            index = futures[future]

            try:
                result_index, candidate, image_bytes = future.result()
                downloaded[result_index] = (
                    candidate,
                    image_bytes,
                )
            except Exception as exc:
                print(
                    f"[{index}/{total}] Download worker failed: {exc}"
                )
                downloaded[index] = (
                    candidates[index - 1],
                    None,
                )

            completed_downloads += 1

            if progress_callback:
                progress_callback(
                    completed_downloads,
                    total,
                    candidates[index - 1],
                )

    # --------------------------------------------------
    # 3. FACE MATCHING PHASE
    # --------------------------------------------------

    results = []

    for index in range(1, total + 1):

        candidate, image_bytes = downloaded.get(
            index,
            (candidates[index - 1], None),
        )

        result = _process_downloaded_candidate(
            index,
            candidate,
            image_bytes,
            input_embedding,
        )

        if result is not None:
            results.append(result)

        if progress_callback:
            progress_callback(
                index,
                total,
                candidate,
            )

    # --------------------------------------------------
    # 4. Sort by similarity
    # --------------------------------------------------

    results.sort(
        key=lambda x: x["similarity"],
        reverse=True,
    )

    print(
        f"Face matching complete: "
        f"{len(results)} / {total} candidates contained comparable faces."
    )

    return results


if __name__ == "__main__":

    input_image = "data/test.jpg"

    with open(
        "data/lens_candidates.json",
        "r",
        encoding="utf-8",
    ) as f:
        candidates = json.load(f)

    print(
        f"Loaded {len(candidates)} candidates."
    )

    results = compare_faces(
        input_image,
        candidates,
    )

    with open(
        "data/face_match_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n================================")
    print("FACE MATCHING COMPLETE")
    print("================================")

    for result in results[:10]:
        print(
            f"{result['similarity']:.4f} | "
            f"{result['source']} | "
            f"{result['title']}"
        )
        print(f"    {result['link']}")
