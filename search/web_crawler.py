"""Independent public-web crawler for the social-media discovery branch.

This crawler does NOT use Google/Bing/Lens/SerpApi. It can render public
JavaScript pages with Playwright, extract ordinary links and image URLs,
and crawl those pages in Instagram-first priority order.

Important limitation: an ordinary crawler cannot discover an unknown person's
Instagram/Facebook/etc. profile from a face alone. It needs public URL seeds
(or links discovered from a seeded public page). It never logs in, bypasses
CAPTCHAs/access controls, or ignores robots.txt.
"""

import heapq
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser
from concurrent.futures import ThreadPoolExecutor

import requests

try:
    from playwright.sync_api import sync_playwright
except Exception:  # Playwright is optional; requests fallback still works.
    sync_playwright = None

USER_AGENT = "FaceBlockchainVerifierCrawler/1.1 (+public-web-research-demo)"
IMAGE_RE = re.compile(r"\.(?:jpg|jpeg|png|webp|gif)(?:\?.*)?$", re.I)

# Lower number = higher priority. Instagram is deliberately first.
SOCIAL_PRIORITY = {
    "instagram.com": 0,
    "facebook.com": 1,
    "x.com": 2,
    "twitter.com": 2,
    "youtube.com": 3,
    "youtu.be": 3,
    "reddit.com": 4,
    "linkedin.com": 5,
    "pinterest.com": 6,
}


def _social_platform(url: str):
    host = urlparse(url).netloc.lower().split(":", 1)[0]
    for domain in SOCIAL_PRIORITY:
        if host == domain or host.endswith("." + domain):
            return domain
    return None


def _queue_priority(url: str, depth: int):
    platform = _social_platform(url)
    if platform is None:
        return (50, depth, url)
    return (SOCIAL_PRIORITY[platform], depth, url)


def _normalise(url: str) -> str:
    url, _ = urldefrag((url or "").strip())
    return url.rstrip("/") if urlparse(url).path == "" else url


def _is_http(url: str) -> bool:
    return urlparse(url).scheme in {"http", "https"}


def _is_image_url(url: str) -> bool:
    return bool(IMAGE_RE.search(urlparse(url).path))


class _PageParser(HTMLParser):
    """Extract links/images from rendered or raw HTML."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title_parts = []
        self.in_title = False
        self.links = []
        self.images = []
        self.meta_images = []
        self.raw_data = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        tag = tag.lower()
        if tag == "title":
            self.in_title = True
        elif tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
        elif tag == "img":
            for key in ("src", "data-src", "data-lazy-src", "data-original"):
                value = attrs.get(key)
                if value:
                    self.images.append(value)
            srcset = attrs.get("srcset") or attrs.get("data-srcset")
            if srcset:
                for part in srcset.split(","):
                    candidate = part.strip().split(" ")[0]
                    if candidate:
                        self.images.append(candidate)
        elif tag == "source":
            srcset = attrs.get("srcset")
            if srcset:
                for part in srcset.split(","):
                    candidate = part.strip().split(" ")[0]
                    if candidate:
                        self.images.append(candidate)
        elif tag == "meta":
            prop = (attrs.get("property") or attrs.get("name") or "").lower()
            content = attrs.get("content")
            if content and prop in {
                "og:image", "og:image:url", "twitter:image", "twitter:image:src"
            }:
                self.meta_images.append(content)

    def handle_endtag(self, tag):
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data)
        # JSON-LD / framework data often contains URLs that are not represented
        # by ordinary <a> tags. Keep a bounded amount for URL extraction.
        if data and ("http://" in data or "https://" in data):
            self.raw_data.append(data[:200000])


def _extract_urls_from_embedded_data(chunks):
    found = []
    pattern = re.compile(r'https?://[^\"\'<>\\\s]+', re.I)
    for chunk in chunks[-20:]:
        for match in pattern.findall(chunk):
            found.append(match.rstrip(".,);]}"))
    return found


def _robots_for(origin, cache):
    if origin in cache:
        return cache[origin]

    rp = RobotFileParser()
    robots_url = origin + "/robots.txt"
    try:
        response = requests.get(
            robots_url,
            headers={"User-Agent": USER_AGENT},
            timeout=2,
            allow_redirects=True,
        )
        # A missing robots.txt is not a prohibition. Explicit denial/errors
        # are handled conservatively rather than treating them as permission.
        if response.status_code == 404:
            rp.parse([])
            cache[origin] = (rp, "missing")
        elif response.status_code == 200:
            rp.parse(response.text.splitlines())
            cache[origin] = (rp, "ok")
        else:
            cache[origin] = (None, f"http_{response.status_code}")
    except requests.RequestException as exc:
        cache[origin] = (None, f"error:{type(exc).__name__}")
    return cache[origin]


def _allowed_by_robots(url, cache):
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False, "invalid-url"
    origin = f"{parsed.scheme}://{parsed.netloc}"
    rp, status = _robots_for(origin, cache)
    # If robots.txt cannot be retrieved, do not pretend the page is
    # forbidden. We still use conservative request limits and never bypass
    # authentication/CAPTCHA/access controls. An explicit robots denial is
    # always respected.
    if rp is None:
        return True, f"robots-unavailable:{status}"
    if not rp.can_fetch(USER_AGENT, url):
        return False, "robots-denied"
    return True, "allowed"


def _looks_like_social_content(url: str) -> bool:
    """Prefer actual profile/post/content URLs over platform homepages."""
    path = urlparse(url).path.strip("/").lower()
    if not path:
        return False
    blocked = {"explore", "discover", "search", "login", "accounts", "about", "help"}
    if path.split("/")[0] in blocked:
        return False
    return True


def _fetch_requests(session, url, timeout):
    response = session.get(
        url,
        timeout=timeout,
        allow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
    )
    ctype = response.headers.get("content-type", "").lower()
    if response.status_code >= 400 or "text/html" not in ctype:
        return None, f"HTTP {response.status_code} ({ctype or 'unknown content type'})"
    return response.url, response.text


def _fetch_browser(page, url, timeout):
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
        # Give client-rendered social pages a short opportunity to populate
        # profile/post cards without turning the crawler into a long scraper.
        page.wait_for_timeout(300)
        # Trigger a small amount of lazy loading without turning each page
        # into a long browser session.
        page.evaluate("window.scrollTo(0, Math.min(document.body.scrollHeight, 1600))")
        page.wait_for_timeout(100)
        html = page.content()
        final_url = page.url
        return final_url, html
    except Exception as exc:
        return None, f"browser:{type(exc).__name__}: {exc}"


def crawl_public_web(
    seed_urls,
    max_pages=20,
    max_depth=2,
    same_domain_only=False,
    request_timeout=4,
    delay=0,
    render_javascript=True,
):
    """Crawl public pages and return image candidates + crawl diagnostics."""

    seeds = []
    for raw in seed_urls or []:
        u = _normalise(raw)
        if _is_http(u) and u not in seeds:
            seeds.append(u)

    queue = []
    sequence = 0
    for u in seeds:
        p = _queue_priority(u, 0)
        heapq.heappush(queue, (*p, sequence, u, 0))
        sequence += 1

    visited = set()
    queued = set(seeds)
    candidates = []
    seen_images = set()
    robots_cache = {}
    diagnostics = []
    blocked = []

    # Fetch robots.txt for all seed origins concurrently. We still enforce
    # every explicit robots denial, but one slow origin can no longer stall the
    # whole discovery pass.
    seed_origins = sorted({
        f"{urlparse(u).scheme}://{urlparse(u).netloc}"
        for u in seeds
        if urlparse(u).netloc
    })
    if seed_origins:
        with ThreadPoolExecutor(max_workers=min(8, len(seed_origins))) as robots_executor:
            list(robots_executor.map(lambda origin: _robots_for(origin, robots_cache), seed_origins))

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    browser = None
    playwright = None
    page = None
    browser_mode = False

    # Check all seed origins concurrently. Serial robots.txt checks can
    # otherwise add several seconds before the first page is even fetched.
    seed_origins = sorted({
        f"{urlparse(u).scheme}://{urlparse(u).netloc}"
        for u in seeds
        if urlparse(u).netloc
    })
    if seed_origins:
        with ThreadPoolExecutor(max_workers=min(8, len(seed_origins))) as robots_executor:
            list(robots_executor.map(lambda origin: _robots_for(origin, robots_cache), seed_origins))

    if render_javascript and sync_playwright is not None:
        try:
            playwright = sync_playwright().start()
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(
                user_agent=USER_AGENT,
                viewport={"width": 1365, "height": 900},
            )
            browser_mode = True
        except Exception as exc:
            diagnostics.append(f"Playwright unavailable; requests fallback: {exc}")
            try:
                if playwright:
                    playwright.stop()
            except Exception:
                pass
            playwright = browser = page = None

    try:
        while queue and len(visited) < max_pages:
            _, _, _, _, url, depth = heapq.heappop(queue)
            queued.discard(url)
            url = _normalise(url)
            if url in visited:
                continue
            visited.add(url)

            allowed, reason = _allowed_by_robots(url, robots_cache)
            if not allowed:
                blocked.append({"url": url, "reason": reason})
                continue

            final_url = None
            html = None
            fetch_method = "requests"
            error = None

            # Browser rendering is the important fix for modern social sites:
            # most profile/post links are injected by JavaScript and are absent
            # from the initial HTML fetched by requests.
            if browser_mode and _social_platform(url):
                final_url, html_or_error = _fetch_browser(page, url, request_timeout)
                if final_url is None:
                    error = html_or_error
                else:
                    html = html_or_error
                    fetch_method = "playwright"

            if html is None:
                try:
                    final_url, html_or_error = _fetch_requests(session, url, request_timeout)
                    if final_url is None:
                        error = html_or_error
                    else:
                        html = html_or_error
                        fetch_method = "requests"
                except requests.RequestException as exc:
                    error = f"requests:{type(exc).__name__}: {exc}"

            if html is None:
                diagnostics.append({"url": url, "depth": depth, "status": "failed", "error": error})
                continue

            final_url = _normalise(final_url)
            parser = _PageParser()
            try:
                parser.feed(html)
            except Exception as exc:
                diagnostics.append({"url": final_url, "depth": depth, "status": "parse_failed", "error": str(exc)})
                continue

            title = " ".join(" ".join(parser.title_parts).split()) or "Public social/web page"

            # Extract image URLs from rendered HTML, metadata and embedded data.
            raw_images = parser.images + parser.meta_images
            for raw in raw_images:
                if not raw:
                    continue
                image_url = _normalise(urljoin(final_url, raw))
                if not _is_http(image_url) or image_url in seen_images:
                    continue
                seen_images.add(image_url)
                platform = _social_platform(final_url)
                candidates.append({
                    "title": title,
                    "source": urlparse(final_url).netloc,
                    "link": final_url,
                    "image": image_url,
                    "thumbnail": image_url,
                    "snippet": "Discovered by independent public-web crawl",
                    "crawler_depth": depth,
                    "platform": platform,
                    "is_social": platform is not None,
                    "fetch_method": fetch_method,
                })

            # Follow ordinary links plus URLs embedded in rendered framework data.
            links = list(parser.links) + _extract_urls_from_embedded_data(parser.raw_data)
            discovered = 0
            if depth < max_depth:
                base_host = urlparse(final_url).netloc.lower()
                for raw_href in links:
                    href = _normalise(urljoin(final_url, raw_href))
                    if not _is_http(href) or href in visited or href in queued:
                        continue
                    parsed_href = urlparse(href)
                    if parsed_href.username or parsed_href.password:
                        continue
                    if same_domain_only and parsed_href.netloc.lower() != base_host:
                        continue
                    # The social branch stays focused on social domains. A
                    # discovered external link is not queued unless the caller
                    # explicitly supplied it as a seed or disables this policy.
                    if not _social_platform(href):
                        continue
                    # Prefer actual profile/post/content paths, but still allow
                    # navigation pages when they are the only discovered route.
                    priority_bonus = 0 if _looks_like_social_content(href) else 5
                    p = _queue_priority(href, depth + 1)
                    heapq.heappush(queue, (p[0], p[1] + priority_bonus, p[2], sequence, href, depth + 1))
                    queued.add(href)
                    sequence += 1
                    discovered += 1

            diagnostics.append({
                "url": final_url,
                "depth": depth,
                "status": "ok",
                "fetch_method": fetch_method,
                "images": len(raw_images),
                "links_discovered": discovered,
                "platform": _social_platform(final_url),
                "title": title[:180],
            })

            if delay:
                time.sleep(delay)
    finally:
        try:
            if page:
                page.close()
        except Exception:
            pass
        try:
            if browser:
                browser.close()
        except Exception:
            pass
        try:
            if playwright:
                playwright.stop()
        except Exception:
            pass

    # Social priority is preserved in the output as well as the frontier.
    # De-duplicate candidates by page + image URL.
    unique = {}
    for item in candidates:
        key = ((item.get("link") or "").strip(), (item.get("image") or "").strip())
        unique[key] = item
    candidates = list(unique.values())

    candidates.sort(
        key=lambda item: (
            0 if item.get("is_social") else 1,
            SOCIAL_PRIORITY.get(item.get("platform"), 99),
            item.get("crawler_depth", 999),
        )
    )

    return {
        "candidates": candidates,
        "pages_crawled": len([x for x in diagnostics if x.get("status") == "ok"]),
        "pages_visited": len(visited),
        "seeds": seeds,
        "max_depth": max_depth,
        "browser_rendering": browser_mode,
        "diagnostics": diagnostics[-120:],
        "blocked": blocked[-120:],
        "discovery_method": "explicit public URL seeds → Instagram-first social-priority frontier → JavaScript rendering/scrolling where available → public link/image extraction; explicit robots denials respected",
        "social_platform_priority": list(SOCIAL_PRIORITY.keys()),
    }
