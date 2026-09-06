import os
import re
from urllib.parse import urlparse

import requests
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

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


def _normalise_profile_url(profile_url):
    """Return a safe canonical public profile URL and its host/path."""
    value = (profile_url or "").strip()
    if not value:
        return None
    if not re.match(r"^https?://", value, re.I):
        value = "https://" + value

    parsed = urlparse(value)
    host = parsed.netloc.lower().split(":", 1)[0]
    path = parsed.path.rstrip("/")

    if not host or not path or path == "/":
        raise ValueError("Enter a specific public profile URL, not only a social-media homepage.")

    allowed = (
        "instagram.com",
        "facebook.com",
        "x.com",
        "twitter.com",
        "youtube.com",
        "youtu.be",
        "reddit.com",
        "linkedin.com",
        "pinterest.com",
    )
    if not any(host == d or host.endswith("." + d) for d in allowed):
        raise ValueError("The profile URL must belong to a supported public social platform.")

    return f"https://{host}{path}"


def _platform(host):
    if "instagram.com" in host:
        return "Instagram"
    if "facebook.com" in host:
        return "Facebook"
    if "x.com" in host or "twitter.com" in host:
        return "X"
    if "youtube.com" in host or "youtu.be" in host:
        return "YouTube"
    if "reddit.com" in host:
        return "Reddit"
    if "linkedin.com" in host:
        return "LinkedIn"
    if "pinterest.com" in host:
        return "Pinterest"
    return "Web"


def _google_search(params):
    api_key = _get_serpapi_key()
    if not api_key:
        raise ValueError("SERPAPI_KEY is not configured. Add it to .env locally or Streamlit Cloud → Settings → Secrets.")
    response = requests.get(
        "https://serpapi.com/search",
        params={**params, "api_key": api_key, "hl": "en", "gl": "in"},
        timeout=6,
    )
    response.raise_for_status()
    return response.json()


def search_public_profile(profile_url, max_results=30):
    """
    Discover publicly indexed pages/images belonging to a user-supplied
    public social profile. This is intentionally identity-scoped: the
    operator supplies the profile, and SFace later verifies whether the
    uploaded face appears in the returned public material.

    No login, CAPTCHA bypass, private-page access, or platform scraping is
    performed here.
    """
    canonical = _normalise_profile_url(profile_url)
    parsed = urlparse(canonical)
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    platform = _platform(host)

    # Restrict discovery to the supplied public profile path rather than
    # searching the whole platform for an unknown person's face.
    scope = f"site:{host}{path}"
    queries = [
        scope,
        f'{scope} photo',
        f'{scope} post',
    ]

    candidates = []
    seen = set()

    def run_organic(query):
        try:
            return _google_search({
                "engine": "google",
                "q": query,
                "num": min(10, max_results),
            }).get("organic_results", []), None
        except Exception as exc:
            return [], f"Public profile Google search failed: {exc}"

    # These searches are independent. Run them together rather than waiting
    # for each query to finish before starting the next one.
    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        organic_batches = list(executor.map(run_organic, queries))

    for organic, error in organic_batches:
        if error:
            print(error)

        for item in organic:
            link = (item.get("link") or "").strip()
            if not link:
                continue
            key = link.rstrip("/")
            if key in seen:
                continue
            seen.add(key)

            parsed_link = urlparse(link)
            link_host = parsed_link.netloc.lower()
            link_path = parsed_link.path.rstrip("/")
            if link_host != host or not (link_path == path or link_path.startswith(path + "/")):
                continue

            image_url = (
                item.get("thumbnail")
                or item.get("image")
                or item.get("original")
            )

            candidates.append({
                "title": item.get("title", "Public profile result"),
                "source": platform,
                "link": link,
                "image": image_url,
                "thumbnail": image_url,
                "snippet": item.get("snippet", ""),
                "discovery_pipeline": "Public Profile Search",
                "profile_seed": canonical,
                "profile_scoped": True,
            })

    image_queries = [
        f'{scope} photo',
        f'{scope} post',
    ]

    def run_image(query):
        try:
            return _google_search({
                "engine": "google_images",
                "q": query,
                "num": min(20, max_results),
            }).get("images_results", []), None
        except Exception as exc:
            return [], f"Public profile image search failed: {exc}"

    with ThreadPoolExecutor(max_workers=len(image_queries)) as executor:
        image_batches = list(executor.map(run_image, image_queries))

    for image_results, error in image_batches:
        if error:
            print(error)

        for item in image_results:
            source_link = (item.get("link") or item.get("source") or "").strip()
            image_url = (
                item.get("original")
                or item.get("thumbnail")
                or item.get("image")
            )
            if not image_url:
                continue

            parsed_link = urlparse(source_link)
            link_host = parsed_link.netloc.lower()
            link_path = parsed_link.path.rstrip("/")

            if source_link and not (
                link_host == host and
                (link_path == path or link_path.startswith(path + "/"))
            ):
                continue

            key = f"img:{image_url}"
            if key in seen:
                continue
            seen.add(key)

            candidates.append({
                "title": item.get("title", "Public profile image"),
                "source": platform,
                "link": source_link or canonical,
                "image": image_url,
                "thumbnail": item.get("thumbnail"),
                "snippet": item.get("snippet", ""),
                "discovery_pipeline": "Public Profile Search",
                "profile_seed": canonical,
                "profile_scoped": True,
            })

    return candidates[:max_results]
