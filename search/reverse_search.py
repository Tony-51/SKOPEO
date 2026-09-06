import os
import json
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")


def _get_serpapi_key():
    """Read SerpApi key from environment or Streamlit Cloud secrets."""
    key = os.getenv("SERPAPI_KEY")
    if key:
        return key.strip()

    try:
        import streamlit as st
        key = st.secrets.get("SERPAPI_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass

    return None


def prepare_image_for_serpapi(image_path):
    """
    Resize/compress the image so it stays below
    SerpApi's 500 KB upload limit.
    """

    output_path = "data/serpapi_upload.jpg"

    os.makedirs("data", exist_ok=True)

    image = Image.open(image_path).convert("RGB")

    # Keep dimensions reasonable
    max_dimension = 1600

    width, height = image.size

    if max(width, height) > max_dimension:

        scale = max_dimension / max(width, height)

        new_size = (
            int(width * scale),
            int(height * scale)
        )

        image = image.resize(
            new_size,
            Image.LANCZOS
        )

    # Try progressively lower JPEG quality
    for quality in [90, 80, 70, 60, 50, 40]:

        image.save(
            output_path,
            format="JPEG",
            quality=quality,
            optimize=True
        )

        file_size = os.path.getsize(output_path)

        print(
            f"Prepared image: "
            f"{file_size / 1024:.1f} KB "
            f"(quality={quality})"
        )

        if file_size <= 450 * 1024:
            break

    # Final safety check
    final_size = os.path.getsize(output_path)

    if final_size > 500 * 1024:

        raise RuntimeError(
            f"Could not compress image below SerpApi's "
            f"500 KB limit. Current size: "
            f"{final_size / 1024:.1f} KB"
        )

    return output_path


def upload_image_to_serpapi(image_path):
    """Upload an image once and return the SerpApi image_id.

    The previous implementation uploaded the same image for every Lens
    query. The pipeline now uploads each prepared image once, then reuses the
    image_id for all Lens searches. This removes several redundant network
    round-trips while keeping the search strategy unchanged.
    """
    api_key = _get_serpapi_key()
    if not api_key:
        raise ValueError(
            "SERPAPI_KEY is not configured. Add it to .env locally or Streamlit Cloud → Settings → Secrets."
        )

    upload_path = prepare_image_for_serpapi(image_path)

    with open(upload_path, "rb") as image_file:
        response = requests.post(
            "https://serpapi.com/image",
            files={"image": ("face.jpg", image_file, "image/jpeg")},
            data={"api_key": api_key},
            timeout=8,
        )

    if not response.ok:
        try:
            error_data = response.json()
        except Exception:
            error_data = response.text
        raise RuntimeError(
            f"SerpApi image upload failed (HTTP {response.status_code}): {error_data}"
        )

    upload_data = response.json()
    image_id = upload_data.get("image_id")
    if not image_id:
        raise RuntimeError(f"Image upload failed: {upload_data}")
    return image_id


def search_google_lens_image_id(image_id, search_type="visual_matches", query=None):
    """Run one Google Lens search against an already-uploaded image."""
    api_key = _get_serpapi_key()
    if not api_key:
        raise ValueError(
            "SERPAPI_KEY is not configured. Add it to .env locally or Streamlit Cloud → Settings → Secrets."
        )

    params = {
        "engine": "google_lens",
        "image_id": image_id,
        "type": search_type,
        "hl": "en",
        "country": "in",
        "api_key": api_key,
    }
    if query:
        params["q"] = query

    response = requests.get(
        "https://serpapi.com/search",
        params=params,
        timeout=8,
    )

    if not response.ok:
        try:
            error_data = response.json()
        except Exception:
            error_data = response.text
        raise RuntimeError(
            f"Google Lens search failed (HTTP {response.status_code}): {error_data}"
        )
    return response.json()


def search_google_lens(image_path, search_type="visual_matches", query=None):
    """Backward-compatible one-shot Lens search."""
    image_id = upload_image_to_serpapi(image_path)
    return search_google_lens_image_id(image_id, search_type=search_type, query=query)


def save_results(results):

    """
    Save the complete Lens response and a simplified
    candidate list for the next stage.
    """

    os.makedirs("data", exist_ok=True)

    # --------------------------------------------------
    # Save complete API response
    # --------------------------------------------------

    with open(
        "data/lens_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------
    # Extract visual matches
    # --------------------------------------------------

    visual_matches = results.get(
        "visual_matches",
        []
    )

    candidates = []

    for match in visual_matches:

        candidate = {
            "title": match.get("title"),
            "source": match.get("source"),
            "link": match.get("link"),
            "image": match.get("image"),
            "thumbnail": match.get("thumbnail"),
            "snippet": match.get("snippet")
        }

        candidates.append(candidate)

    # --------------------------------------------------
    # Save simplified candidates
    # --------------------------------------------------

    with open(
        "data/lens_candidates.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            candidates,
            f,
            indent=2,
            ensure_ascii=False
        )

    return candidates


if __name__ == "__main__":

    image_path = "data/test.jpg"

    results = search_google_lens(
        image_path
    )

    candidates = save_results(
        results
    )

    print(
        "\nGoogle Lens search completed!"
    )

    print(
        "Number of visual matches:",
        len(candidates)
    )

    print("\nTop results:")

    for i, candidate in enumerate(
        candidates[:10],
        start=1
    ):

        print(
            f"\n--- Match {i} ---"
        )

        print(
            "Title:",
            candidate["title"]
        )

        print(
            "Source:",
            candidate["source"]
        )

        print(
            "Link:",
            candidate["link"]
        )

    print(
        "\n--------------------------------"
    )

    print(
        "Results saved successfully!"
    )

    print(
        "--------------------------------"
    )

    print(
        "data/lens_results.json"
    )

    print(
        "data/lens_candidates.json"
    )