import os
import sys
import json
import hashlib
from urllib.parse import urlparse

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# --------------------------------------------------
# PROJECT IMPORTS
# --------------------------------------------------

from face.detector import detect_faces
from face.encoder import generate_embedding
from search.reverse_search import (
    search_google_lens,
    upload_image_to_serpapi,
    search_google_lens_image_id,
    save_results,
)
from search.face_matcher import compare_faces
from search.web_crawler import crawl_public_web
from search.profile_search import search_public_profile
from concurrent.futures import ThreadPoolExecutor, wait
from blockchain.chain import Blockchain
from blockchain.verify_result import create_fingerprint


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Face → Web → Blockchain Verifier",
    page_icon="",
    layout="wide"
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --hh-green: #075b35;
        --hh-green-dark: #043b25;
        --hh-yellow: #f5df00;
        --hh-pink: #ff1493;
    }

    .stApp {
        background: var(--hh-green);
        color: var(--hh-yellow);
    }

    [data-testid="stHeader"] { background: transparent; }

    [data-testid="stSidebar"] {
        background: var(--hh-green-dark);
        border-right: 1px solid rgba(245,223,0,.45);
    }

    [data-testid="stSidebar"] * {
        color: var(--hh-yellow) !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-family: "DM Mono", monospace;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    .hh-topbar {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 2.5rem;
        font-family: "DM Mono", monospace;
        letter-spacing: .08em;
        text-transform: uppercase;
        font-size: .85rem;
    }

    .hh-brand {
        font-family: "Space Grotesk", sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        line-height: 1;
        color: var(--hh-yellow);
        text-transform: uppercase;
    }

    .hh-brand span {
        display: block;
        font-family: "DM Mono", monospace;
        font-size: .65rem;
        margin-top: .45rem;
        opacity: .85;
    }

    .hh-nav {
        display: flex;
        gap: 2rem;
        align-items: center;
        color: var(--hh-yellow);
    }

    .hh-nav .apply {
        border: 2px solid var(--hh-yellow);
        background: var(--hh-yellow);
        color: var(--hh-green-dark);
        padding: .8rem 1.5rem;
        font-weight: 700;
        letter-spacing: .08em;
    }

    .hh-kicker {
        font-family: "DM Mono", monospace;
        text-transform: uppercase;
        letter-spacing: .16em;
        font-size: .78rem;
        margin-top: 2rem;
        color: var(--hh-yellow);
    }

    .hh-hero {
        position: relative;
        margin: 1rem 0 2rem 0;
    }

    .hh-title {
        font-family: Georgia, "Times New Roman", serif;
        font-weight: 400;
        font-size: clamp(4.8rem, 11vw, 10rem);
        line-height: .78;
        letter-spacing: -.055em;
        color: var(--hh-yellow);
        text-transform: uppercase;
        margin: 0;
    }

    .hh-title .line2 { margin-left: 6vw; }

    .hh-sticker {
        display: inline-block;
        position: absolute;
        right: 5%;
        top: 42%;
        transform: rotate(-7deg);
        border: 7px solid var(--hh-pink);
        color: var(--hh-pink);
        padding: .35rem .75rem;
        font-family: "Space Grotesk", sans-serif;
        font-weight: 800;
        font-size: 1.5rem;
        text-transform: uppercase;
        background: var(--hh-green);
    }

    .hh-sub {
        font-family: "DM Mono", monospace;
        color: var(--hh-yellow);
        letter-spacing: .12em;
        text-transform: uppercase;
        font-size: .75rem;
        margin-top: 2rem;
    }

    .hh-footer {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        border-top: 1px solid rgba(245,223,0,.5);
        margin-top: 4rem;
        padding-top: 1rem;
        font-family: "DM Mono", monospace;
        font-size: .7rem;
        color: var(--hh-yellow);
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    [data-testid="stFileUploader"] {
        background: transparent;
        border: 1px dashed var(--hh-yellow);
        padding: 1rem;
    }

    [data-testid="stFileUploader"] * { color: var(--hh-yellow) !important; }

    .stButton > button,
    [data-testid="stFormSubmitButton"] button {
        width: 100%;
        border-radius: 0;
        border: 2px solid var(--hh-yellow);
        background: var(--hh-yellow);
        color: var(--hh-green-dark);
        font-family: "DM Mono", monospace;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .08em;
        min-height: 3.2rem;
    }

    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        border-color: var(--hh-pink);
        background: var(--hh-pink);
        color: white;
    }

    .stTextInput input,
    .stTextArea textarea {
        background: var(--hh-green-dark);
        color: var(--hh-yellow);
        border: 1px solid rgba(245,223,0,.65);
        border-radius: 0;
    }

    [data-testid="stTabs"] button {
        color: var(--hh-yellow);
        font-family: "DM Mono", monospace;
        text-transform: uppercase;
    }

    a {
        color: var(--hh-yellow) !important;
        text-decoration: underline;
    }

    hr { border-color: rgba(245,223,0,.35); }

    /* Premium animated pipeline rail */
    .pipeline-wrap {
        margin-top: .5rem;
        font-family: "DM Mono", monospace;
        position: relative;
    }
    .pipeline-item {
        position: relative;
        padding: .78rem .7rem .78rem 2.15rem;
        margin: .32rem 0;
        border: 1px solid rgba(245,223,0,.12);
        border-left: 2px solid rgba(245,223,0,.28);
        border-radius: 10px;
        background: rgba(255,255,255,.018);
        transition: transform .25s ease, background .25s ease, border-color .25s ease, box-shadow .25s ease;
        overflow: hidden;
    }
    .pipeline-item::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(110deg, transparent 20%, rgba(245,223,0,.08) 48%, transparent 76%);
        transform: translateX(-120%);
        animation: pipelineSweep 3.2s ease-in-out infinite;
        pointer-events: none;
    }
    .pipeline-item:hover {
        transform: translateX(4px);
        background: rgba(245,223,0,.055);
        border-color: rgba(245,223,0,.42);
    }
    .pipeline-item::before {
        content: "";
        position: absolute;
        left: .55rem;
        top: 50%;
        transform: translateY(-50%);
        width: .68rem;
        height: .68rem;
        border: 1px solid var(--hh-yellow);
        background: var(--hh-green-dark);
        border-radius: 50%;
        z-index: 2;
    }
    .pipeline-item.done { opacity: .9; }
    .pipeline-item.done::before {
        background: var(--hh-yellow);
        box-shadow: 0 0 0 4px rgba(245,223,0,.10), 0 0 14px rgba(245,223,0,.28);
    }
    .pipeline-item.active {
        border-left: 3px solid var(--hh-pink);
        background: linear-gradient(90deg, rgba(255,20,147,.14), rgba(255,20,147,.035));
        transform: translateX(5px);
        box-shadow: 0 8px 28px rgba(0,0,0,.16), inset 0 0 24px rgba(255,20,147,.04);
    }
    .pipeline-item.active::before {
        border-color: var(--hh-pink);
        background: var(--hh-pink);
        box-shadow: 0 0 0 0 rgba(255,20,147,.55);
        animation: pipelinePulse 1.25s infinite;
    }
    .pipeline-item.pending { opacity: .48; }
    .pipeline-name { font-weight: 700; font-size: .78rem; letter-spacing: .03em; }
    .pipeline-tech { margin-top: .18rem; font-size: .66rem; opacity: .78; line-height: 1.35; }
    .pipeline-state { display: inline-block; margin-top: .3rem; font-size: .56rem; letter-spacing: .13em; text-transform: uppercase; }
    .pipeline-item.active .pipeline-state { color: var(--hh-pink) !important; animation: pipelineBlink .8s infinite alternate; }
    .pipeline-item.done .pipeline-state { color: var(--hh-yellow) !important; }
    .pipeline-line {
        height: 1px; margin: .42rem .25rem .42rem 1rem;
        background: linear-gradient(90deg, rgba(245,223,0,.12), rgba(255,20,147,.75), rgba(245,223,0,.08));
        transform-origin: left; animation: pipelineScan 2.2s ease-in-out infinite;
    }
    .search-activity {
        margin: 1rem 0 1.2rem; padding: 1.1rem 1.25rem; border-radius: 14px;
        border: 1px solid rgba(245,223,0,.28);
        background: linear-gradient(135deg, rgba(4,59,37,.9), rgba(7,91,53,.65));
        box-shadow: 0 12px 35px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.05);
        font-family: "DM Mono", monospace; overflow: hidden; position: relative;
    }
    .search-activity::before { content:""; position:absolute; top:0; left:-30%; width:30%; height:2px; background:var(--hh-pink); box-shadow:0 0 18px var(--hh-pink); animation: searchBeam 1.35s linear infinite; }
    .search-title { font-weight:700; letter-spacing:.1em; text-transform:uppercase; font-size:.78rem; }
    .search-sub { margin-top:.35rem; font-size:.64rem; opacity:.72; letter-spacing:.05em; }
    .search-dots { display:flex; gap:.35rem; margin-top:.85rem; }
    .search-dots span { width:.42rem; height:.42rem; border-radius:50%; background:var(--hh-yellow); animation: dotPulse 1s infinite ease-in-out; }
    .search-dots span:nth-child(2){animation-delay:.16s}.search-dots span:nth-child(3){animation-delay:.32s}.search-dots span:nth-child(4){animation-delay:.48s}
    @keyframes pipelinePulse { 0%{box-shadow:0 0 0 0 rgba(255,20,147,.65)} 70%{box-shadow:0 0 0 9px rgba(255,20,147,0)} 100%{box-shadow:0 0 0 0 rgba(255,20,147,0)} }
    @keyframes pipelineBlink { from{opacity:.55} to{opacity:1} }
    @keyframes pipelineScan { 0%,100%{transform:scaleX(.2);opacity:.25} 50%{transform:scaleX(1);opacity:1} }
    @keyframes pipelineSweep { 0%,45%{transform:translateX(-120%)} 75%,100%{transform:translateX(120%)} }
    @keyframes searchBeam { from{left:-30%} to{left:110%} }
    @keyframes dotPulse { 0%,100%{transform:translateY(0);opacity:.35} 50%{transform:translateY(-4px);opacity:1} }

    @media (max-width: 900px) {
        .hh-title { font-size: clamp(3.5rem, 17vw, 6rem); }
        .hh-title .line2 { margin-left: 0; }
        .hh-sticker { position: static; margin-top: 1.5rem; }
        .hh-footer { flex-direction: column; }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if "input_hash" not in st.session_state:
    st.session_state.input_hash = None

if "last_uploaded_hash" not in st.session_state:
    st.session_state.last_uploaded_hash = None


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def is_social_url(url):
    """Check whether a URL belongs to a social-media platform."""

    if not url:
        return False

    host = urlparse(url).netloc.lower()

    social_domains = [
        "instagram.com",
        "facebook.com",
        "reddit.com",
        "x.com",
        "twitter.com",
        "youtube.com",
        "youtu.be",
        "linkedin.com",
        "pinterest.com",
    ]

    return any(domain in host for domain in social_domains)


SOCIAL_LENS_QUERY = (
    "site:instagram.com OR site:facebook.com OR site:x.com "
    "OR site:twitter.com OR site:youtube.com OR site:youtu.be "
    "OR site:reddit.com OR site:linkedin.com OR site:pinterest.com"
)

SOCIAL_PLATFORM_PRIORITY = {
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

def social_platform(url):
    host = urlparse(url or "").netloc.lower().split(":", 1)[0]
    for domain in SOCIAL_PLATFORM_PRIORITY:
        if host == domain or host.endswith("." + domain):
            return domain
    return None

def social_priority(url):
    platform = social_platform(url)
    return SOCIAL_PLATFORM_PRIORITY.get(platform, 99)


def save_uploaded_image(uploaded_file):
    """Save uploaded image to data/test.jpg."""

    os.makedirs("data", exist_ok=True)

    image_bytes = uploaded_file.getvalue()

    with open("data/test.jpg", "wb") as f:
        f.write(image_bytes)

    return image_bytes


def image_hash(image_bytes):
    """Create SHA-256 fingerprint of input image."""

    return hashlib.sha256(image_bytes).hexdigest()


def create_face_search_image(image_array, face):
    """
    Create a face-focused crop for reverse image search.
    Keeps a little head/shoulder context while reducing
    clothing/background influence.
    """

    height, width = image_array.shape[:2]

    x, y, w, h = face[:4].astype(int)

    margin_x = int(w * 0.55)
    margin_top = int(h * 0.45)
    margin_bottom = int(h * 0.25)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_top)
    x2 = min(width, x + w + margin_x)
    y2 = min(height, y + h + margin_bottom)

    face_crop = image_array[y1:y2, x1:x2]

    if face_crop.size == 0:
        raise RuntimeError("Could not create face-focused search image.")

    face_search_path = "data/face_search.jpg"

    face_crop_bgr = cv2.cvtColor(
        face_crop,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        face_search_path,
        face_crop_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 90]
    )

    return face_search_path


def set_pipeline_step(step, completed=None):
    st.session_state.pipeline_active_step = step
    if completed is not None:
        st.session_state.pipeline_completed_steps = set(completed)
    if "pipeline_sidebar_placeholder" in globals():
        with pipeline_sidebar_placeholder.container():
            render_pipeline_sidebar(
                st.session_state.pipeline_active_step,
                st.session_state.pipeline_completed_steps,
            )


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hh-topbar">
        <div class="hh-brand">
            HACKER HOUSE GOA
            <span>SKOPEO</span>
        </div>
        <div class="hh-nav">
            <span>CHECK PIPELINE</span>
            <span class="apply">RUN</span>
        </div>
    </div>

    <div class="hh-kicker">FACE → WEB → BLOCKCHAIN / PUBLIC EVIDENCE WORKFLOW</div>

    <div class="hh-hero">
        <div class="hh-title">
            <div>SKOPEO</div>
        </div>
        <div class="hh-sticker">ONLINE</div>
        <div class="hh-sub">FIND · COMPARE · VERIFY · PRESERVE</div>
    </div>
    """,
    unsafe_allow_html=True
)


with st.expander("Judge Quick Start", expanded=False):
    st.markdown("""
    1. Create a Python 3.10+ virtual environment.
    2. Run `pip install -r requirements.txt`.
    3. Run `streamlit run app.py`.
    4. Upload an authorized face image.
    5. For an ordinary public user, enter the exact public profile URL.
    6. The app runs social-first Google Lens, then profile verification and a bounded public crawler.

    The repository includes `run_project.ps1` for a one-command Windows start.
    """)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

PIPELINE_STEPS = [
    (1, "Face Detection", "YuNet"),
    (2, "Face Encoding", "SFace"),
    (3, "Web Candidate Search", "Social-first SerpApi"),
    (4, "Face Matching", "Cosine similarity"),
    (5, "Verification", "SHA-256 fingerprint"),
    (6, "Blockchain", "Local simulated blockchain"),
]

def render_pipeline_sidebar(active_step=0, completed_steps=None):
    completed_steps = completed_steps or set()
    cards = []
    for number, name, tech in PIPELINE_STEPS:
        if number in completed_steps:
            state = "done"
            label = "COMPLETE"
        elif number == active_step:
            state = "active"
            label = "RUNNING"
        else:
            state = "pending"
            label = "WAITING"
        cards.append(
            f'<div class="pipeline-item {state}">'
            f'<div class="pipeline-name">{number:02d} / {name}</div>'
            f'<div class="pipeline-tech">{tech}</div>'
            f'<div class="pipeline-state">{label}</div>'
            f'</div>'
        )
    st.markdown(
        '<div class="pipeline-wrap">' +
        '<div class="pipeline-line"></div>'.join(cards) +
        '</div>',
        unsafe_allow_html=True,
    )

if "pipeline_active_step" not in st.session_state:
    st.session_state.pipeline_active_step = 0
if "pipeline_completed_steps" not in st.session_state:
    st.session_state.pipeline_completed_steps = set()

with st.sidebar:

    st.header("PIPELINE")
    pipeline_sidebar_placeholder = st.empty()
    with pipeline_sidebar_placeholder.container():
        render_pipeline_sidebar(
            st.session_state.pipeline_active_step,
            st.session_state.pipeline_completed_steps,
        )

    st.divider()

    st.info(
        "Use only images you are authorized to process. "
        "This project is intended for demonstration and verification."
    )

    st.divider()
    st.subheader("AUTHORIZED PUBLIC PROFILE")
    public_profile_url = st.text_input(
        "Public profile URL (recommended for ordinary users)",
        placeholder="https://www.instagram.com/username/",
        help="Use this for an ordinary public user when you are authorized to verify the supplied image against that public profile. It avoids pretending that an image-only reverse search can discover every non-celebrity account.",
    ).strip()

    st.subheader("PUBLIC SOCIAL DISCOVERY")
    st.caption("PRIORITY: INSTAGRAM → FACEBOOK → X → YOUTUBE → REDDIT → LINKEDIN → PINTEREST")

    # Fixed public search roots keep the demo clean while still giving the
    # crawler real discovery entry points. Exact public profile URLs supplied
    # by the user are added separately when available.
    crawler_seeds = [
        "https://www.instagram.com/explore/",
        "https://www.facebook.com/public/",
        "https://x.com/search",
        "https://www.youtube.com/results",
        "https://www.reddit.com/search/",
        "https://www.linkedin.com/search/results/all/",
        "https://www.pinterest.com/search/",
    ]


# --------------------------------------------------
# UPLOAD FORM
# --------------------------------------------------

with st.form("pipeline_form"):

    uploaded_file = st.file_uploader(
        "UPLOAD A FACE IMAGE",
        type=["jpg", "jpeg", "png"],
        help="Upload an image containing a clearly visible face."
    )

    run_pipeline = st.form_submit_button(
        "RUN COMPLETE PIPELINE",
        width="stretch"
    )


# --------------------------------------------------
# CLEAR RESULTS WHEN THE UPLOADER IS CLEARED
# --------------------------------------------------

# Clicking the X on st.file_uploader makes uploaded_file None and
# triggers a rerun. Clear every session result at that point so the
# next uploaded person always starts with a clean pipeline.
if uploaded_file is None:

    st.session_state.pipeline_result = None
    st.session_state.input_hash = None
    st.session_state.last_uploaded_hash = None


# --------------------------------------------------
# DISPLAY INPUT
# --------------------------------------------------

if uploaded_file:

    # Clear results from a previous image immediately when the user
    # selects a different image. This prevents stale results from
    # remaining visible while the new pipeline is running.
    uploaded_hash = hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()

    if st.session_state.last_uploaded_hash != uploaded_hash:
        st.session_state.pipeline_result = None
        st.session_state.input_hash = None
        st.session_state.last_uploaded_hash = uploaded_hash

    col1, col2 = st.columns([1, 2])

    with col1:

        st.image(
            uploaded_file,
            caption="Input Image",
            width=300
        )

    with col2:

        st.markdown("### Ready to verify")

        st.write(
            "The system will detect the face, generate an embedding, "
            "search the web, compare candidate faces, and record the "
            "verification fingerprint on the local blockchain."
        )


# --------------------------------------------------
# PIPELINE
# --------------------------------------------------

if run_pipeline:

    if uploaded_file is None:

        st.error("Please upload an image first.")

        st.stop()

    # --------------------------------------------------
    # INITIALIZE
    # --------------------------------------------------

    progress = st.progress(0)

    status = st.empty()

    try:

        set_pipeline_step(1, set())

        # --------------------------------------------------
        # STEP 1 — SAVE IMAGE
        # --------------------------------------------------

        status.info("Saving input image...")

        image_bytes = save_uploaded_image(uploaded_file)

        current_hash = image_hash(image_bytes)

        progress.progress(5)

        # --------------------------------------------------
        # STEP 2 — READ IMAGE
        # --------------------------------------------------

        image = Image.open(uploaded_file).convert("RGB")

        image_array = np.array(image)

        status.info("Detecting face using YuNet...")

        faces = detect_faces(image_array)

        progress.progress(15)
        set_pipeline_step(1, {1})
        set_pipeline_step(2, {1})

        if len(faces) == 0:

            st.error(
                " No face detected. Please use an image with a clear face."
            )

            st.stop()

        st.success(
            f" {len(faces)} face(s) detected."
        )

        # --------------------------------------------------
        # FACE DETAILS
        # --------------------------------------------------

        for i, face in enumerate(faces):

            x, y, w, h = face[:4].astype(int)

            confidence = float(face[14])

            st.write(
                f"Face {i + 1}: "
                f"x={x}, y={y}, width={w}, height={h}, "
                f"confidence={confidence:.2f}"
            )

        # --------------------------------------------------
        # STEP 3 — ENCODING
        # --------------------------------------------------

        status.info("Generating SFace face embedding...")

        face_box = faces[0]

        embedding = generate_embedding(
            image_array,
            face_box
        )

        progress.progress(25)
        set_pipeline_step(2, {1, 2})
        set_pipeline_step(3, {1, 2})

        st.success(
            f" Face encoding generated successfully "
            f"(shape: {embedding.shape})"
        )

        # --------------------------------------------------
        # STEP 4 — PARALLEL DISCOVERY
        # --------------------------------------------------

        st.markdown(
            """<div class="search-activity">
                <div class="search-title">LIVE DISCOVERY IN PROGRESS</div>
                <div class="search-sub">Querying visual matches • checking authorized public sources • ranking evidence</div>
                <div class="search-dots"><span></span><span></span><span></span><span></span></div>
            </div>""",
            unsafe_allow_html=True,
        )
        status.info("Starting Google Lens and independent web crawler in parallel...")

        face_search_path = create_face_search_image(
            image_array,
            faces[0]
        )

        # Social-first Google Lens remains the first discovery stream, but the
        # platform searches are consolidated into one social query. This keeps
        # Instagram/Facebook/X/YouTube/Reddit/LinkedIn/Pinterest prioritized
        # without paying for seven separate SerpApi requests.
        lens_jobs = [
            (
                "face-focused social visual",
                "face",
                "visual_matches",
                SOCIAL_LENS_QUERY,
                True,
                "Social",
            ),
            (
                "face-focused broad visual",
                "face",
                "visual_matches",
                None,
                False,
                "Broad",
            ),
            (
                "original broad visual",
                "original",
                "visual_matches",
                None,
                False,
                "Broad",
            ),
            (
                "face-focused exact",
                "face",
                "exact_matches",
                None,
                False,
                "Exact",
            ),
        ]

        def run_lens_pipeline():
            merged = []
            seen = set()
            modes = []
            errors = []

            try:
                with ThreadPoolExecutor(max_workers=2) as upload_executor:
                    face_future = upload_executor.submit(
                        upload_image_to_serpapi, face_search_path
                    )
                    original_future = upload_executor.submit(
                        upload_image_to_serpapi, "data/test.jpg"
                    )
                    image_ids = {
                        "face": face_future.result(),
                        "original": original_future.result(),
                    }
            except Exception as exc:
                return [], [], [f"Lens image upload failed: {exc}"]

            def run_one(job):
                label, image_kind, search_type, query, targeted, platform_name = job
                try:
                    response = search_google_lens_image_id(
                        image_ids[image_kind],
                        search_type=search_type,
                        query=query,
                    )
                    return job, response.get(search_type, []) or [], None
                except Exception as exc:
                    return job, [], f"{label}: {exc}"

            with ThreadPoolExecutor(max_workers=len(lens_jobs)) as executor:
                results = list(executor.map(run_one, lens_jobs))

            for job, matches, error in results:
                label, _, _, query, targeted, platform_name = job
                if error:
                    print(error)
                    errors.append(error)
                    continue
                modes.append(f"{label}: {len(matches)}")
                for candidate in matches:
                    link = (candidate.get("link") or "").strip()
                    image_url = candidate.get("image") or candidate.get("thumbnail")
                    key = (
                        f"link:{link}" if link
                        else f"image:{image_url}" if image_url
                        else f"title:{candidate.get('title', '')}"
                    )
                    if key in seen:
                        for existing in merged:
                            if existing.get("_dedupe_key") == key:
                                existing.setdefault("search_sources", []).append(label)
                                break
                        continue
                    seen.add(key)
                    item = dict(candidate)
                    item["search_sources"] = [label]
                    item["lens_query"] = query or "broad visual search"
                    item["social_targeted"] = bool(targeted)
                    item["social_query_platform"] = platform_name
                    item["_dedupe_key"] = key
                    item["discovery_pipeline"] = "Google Lens"
                    merged.append(item)

            for idx, item in enumerate(merged):
                item.pop("_dedupe_key", None)
                item["lens_original_order"] = idx
                item["social_platform"] = social_platform(item.get("link", ""))
                item["social_priority"] = social_priority(item.get("link", ""))
                item["is_social"] = item.get("social_platform") is not None

            merged.sort(
                key=lambda item: (
                    0 if item.get("social_platform") else 1,
                    item.get("social_priority", 99),
                    0 if item.get("social_targeted") else 1,
                    item.get("lens_original_order", 999999),
                )
            )
            return merged, modes, errors

        # Lens, authorized-profile search, and the bounded crawler can overlap.
        # The crawler uses fixed public search roots plus the supplied profile;
        # it no longer waits for Lens before starting its first-page crawl.
        effective_crawler_seeds = list(crawler_seeds)
        if public_profile_url and public_profile_url not in effective_crawler_seeds:
            effective_crawler_seeds.insert(0, public_profile_url)

        # HARD DISCOVERY BUDGET: never let one slow external service hold the
        # entire judge demo hostage. All three discovery branches still run;
        # we simply use whatever has completed within the budget.
        DISCOVERY_BUDGET_SECONDS = 24
        discovery_executor = ThreadPoolExecutor(max_workers=3)
        lens_future = discovery_executor.submit(run_lens_pipeline)
        profile_future = (
            discovery_executor.submit(search_public_profile, public_profile_url, 10)
            if public_profile_url else None
        )
        crawler_future = discovery_executor.submit(
            crawl_public_web,
            effective_crawler_seeds,
            max_pages=7,
            max_depth=1,
            same_domain_only=True,
            render_javascript=True,
            request_timeout=1.8,
            delay=0,
        )

        futures = [lens_future, crawler_future]
        if profile_future is not None:
            futures.append(profile_future)
        done, not_done = wait(futures, timeout=DISCOVERY_BUDGET_SECONDS)

        lens_candidates, search_modes, lens_errors = [], [], []
        profile_candidates, profile_errors = [], []
        crawler_response = {
            "candidates": [], "pages_crawled": 0, "pages_visited": 0,
            "seeds": effective_crawler_seeds, "diagnostics": [], "blocked": [],
            "discovery_method": "bounded public-web crawler"
        }

        if lens_future in done:
            try:
                lens_candidates, search_modes, lens_errors = lens_future.result()
            except Exception as exc:
                lens_errors.append(f"Lens: {exc}")
        else:
            lens_errors.append(f"Google Lens exceeded {DISCOVERY_BUDGET_SECONDS}s demo budget.")

        if profile_future is not None and profile_future in done:
            try:
                profile_candidates = profile_future.result()
                for item in profile_candidates:
                    item["discovery_pipeline"] = "Public Profile Search"
                    item["search_sources"] = ["authorized public profile search"]
                    item["social_platform"] = social_platform(item.get("link", ""))
                    item["social_targeted"] = True
            except Exception as exc:
                profile_errors.append(str(exc))
        elif profile_future is not None:
            profile_errors.append(f"Authorized profile search exceeded {DISCOVERY_BUDGET_SECONDS}s demo budget.")

        if crawler_future in done:
            try:
                crawler_response = crawler_future.result()
            except Exception as exc:
                crawler_response["diagnostics"] = [{"status": "failed", "error": str(exc)}]
        else:
            crawler_response["diagnostics"] = [{
                "status": "budget_exceeded",
                "error": f"Crawler exceeded {DISCOVERY_BUDGET_SECONDS}s demo budget."
            }]

        # Do NOT wait for slow unfinished workers here. They are cancelled if
        # queued and allowed to finish in the background if already running.
        discovery_executor.shutdown(wait=False, cancel_futures=True)

        # Add concrete Lens social pages as high-value evidence, without
        # launching a second full crawl. This keeps the search bounded and fast.

        crawler_candidates = crawler_response.get("candidates", [])

        crawler_candidates.sort(
            key=lambda item: (
                0 if item.get("is_social") else 1,
                social_priority(item.get("link", "")),
                item.get("crawler_depth", 999),
            )
        )

        source_links = []
        seen_source_links = set()
        for candidate in lens_candidates + profile_candidates + crawler_candidates:
            link = (candidate.get("link") or "").strip()
            if link and link not in seen_source_links:
                seen_source_links.add(link)
                source_links.append({
                    "title": candidate.get("title", "Web result"),
                    "source": candidate.get("source", "Web source"),
                    "link": link,
                    "snippet": candidate.get("snippet", ""),
                    "search_type": candidate.get("discovery_pipeline", "Google Lens")
                })

        for item in crawler_candidates:
            item["discovery_pipeline"] = "Open-Web Crawler"
            item["search_sources"] = ["bounded public-web crawl"]

        candidates = lens_candidates + profile_candidates + crawler_candidates

        # Save each discovery stream separately.
        os.makedirs("data", exist_ok=True)
        with open("data/web_candidates.json", "w", encoding="utf-8") as f:
            json.dump(candidates, f, indent=2, ensure_ascii=False)
        with open("data/crawler_candidates.json", "w", encoding="utf-8") as f:
            json.dump(crawler_candidates, f, indent=2, ensure_ascii=False)

        progress.progress(40)
        set_pipeline_step(3, {1, 2, 3})
        set_pipeline_step(4, {1, 2, 3})

        st.success(
            f" Discovery complete — Google Lens: {len(lens_candidates)} candidates | "
            f"Authorized profile: {len(profile_candidates)} | "
            f"Crawler: {len(crawler_candidates)} candidates | "
            f"Crawler pages: {crawler_response.get('pages_crawled', 0)}"
        )
        st.caption("Social discovery priority: Instagram → Facebook → X → YouTube → Reddit → LinkedIn → Pinterest.")

        # --------------------------------------------------
        # STEP 5 — FACE MATCHING
        # --------------------------------------------------

        status.info(
            f" Comparing faces in {len(candidates)} web candidates..."
        )

        st.caption(
            "Only candidate images where YuNet detects a face are "
            "eligible for SFace similarity scoring."
        )

        match_progress = st.progress(0)
        match_status = st.empty()

        def update_match_progress(current, total, candidate):
            percentage = int((current / total) * 100)

            match_progress.progress(percentage)

            source = candidate.get(
                "source",
                "web source"
            )

            match_status.info(
                f" Comparing candidate "
                f"{current} / {total} — {source}"
            )

        with st.spinner(
            "Please wait — downloading candidate images temporarily "
            "and comparing detected faces..."
        ):
            match_results = compare_faces(
                "data/test.jpg",
                candidates,
                progress_callback=update_match_progress
            )

        lens_links = {
            (c.get("link") or "").strip()
            for c in lens_candidates
            if c.get("link")
        }
        crawler_links = {
            (c.get("link") or "").strip()
            for c in crawler_candidates
            if c.get("link")
        }
        for item in match_results:
            link = (item.get("link") or "").strip()
            if link in crawler_links and link not in lens_links:
                item["discovery_pipeline"] = "Open-Web Crawler"
            else:
                item["discovery_pipeline"] = "Google Lens"

        match_progress.progress(100)

        match_status.success(
            f" Compared {len(candidates)} web candidates."
        )

        with open(
            "data/face_match_results.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                match_results,
                f,
                indent=2,
                ensure_ascii=False
            )

        progress.progress(75)
        set_pipeline_step(4, {1, 2, 3, 4})
        set_pipeline_step(5, {1, 2, 3, 4})

        if not match_results:

            st.error(
                " No candidate faces could be compared."
            )

            st.stop()

        # --------------------------------------------------
        # QUALITY GATE + UNIFIED CANDIDATE RANKING
        # --------------------------------------------------
        # Do not label an unrelated low-similarity image as the best match.
        # The threshold is deliberately conservative for this demo; it is
        # a candidate-quality filter, not an identity guarantee.
        MIN_FACE_SIMILARITY = 0.45
        credible_results = [
            item for item in match_results
            if float(item.get("similarity", 0) or 0) >= MIN_FACE_SIMILARITY
        ]

        for item in match_results:
            item["credible_face_candidate"] = float(item.get("similarity", 0) or 0) >= MIN_FACE_SIMILARITY

        # If nothing clears the gate, retain the results for transparency but
        # do not promote an unrelated image into the identity result.
        ranking_pool = credible_results or []

        # --------------------------------------------------
        # UNIFIED IDENTITY CANDIDATE RANKING
        # --------------------------------------------------

        # A single high SFace score is not enough. Give a small evidence
        # bonus when the same source appears in more than one independent
        # search view, and give social URLs a small discovery bonus.
        # Face similarity remains the dominant signal.
        for item in match_results:
            repeated = len(item.get("search_sources", []))
            platform = social_platform(item.get("link", ""))
            social_bonus = 0.035 if platform == "instagram.com" else (0.025 if platform else 0.0)
            targeted_bonus = 0.015 if item.get("social_targeted") else 0.0
            evidence_bonus = min(0.03, max(0, repeated - 1) * 0.015)
            item["identity_score"] = round(
                float(item.get("similarity", 0) or 0) + social_bonus + targeted_bonus + evidence_bonus,
                4
            )
            item["social_priority"] = social_priority(item.get("link", ""))

        # Display/ranking order is social-first, while the face similarity
        # score remains the primary evidence signal. If at least one social
        # candidate clears the quality gate, prefer the best social candidate
        # rather than letting a shopping page win solely on raw similarity.
        match_results.sort(
            key=lambda x: (
                0 if social_platform(x.get("link", "")) else 1,
                social_priority(x.get("link", "")),
                -float(x.get("similarity", 0) or 0),
                -float(x.get("identity_score", 0) or 0),
            )
        )

        if ranking_pool:
            social_pool = [x for x in ranking_pool if social_platform(x.get("link", ""))]
            if social_pool:
                social_pool.sort(
                    key=lambda x: (
                        social_priority(x.get("link", "")),
                        -float(x.get("similarity", 0) or 0),
                        -float(x.get("identity_score", 0) or 0),
                    )
                )
                best_match = social_pool[0]
            else:
                ranking_pool.sort(
                    key=lambda x: (-float(x.get("similarity", 0) or 0), -float(x.get("identity_score", 0) or 0))
                )
                best_match = ranking_pool[0]
        else:
            best_match = {
                "title": "No credible face match",
                "source": "—",
                "link": "",
                "similarity": 0.0,
                "identity_score": 0.0,
                "faces_detected": 0,
                "image_url": "",
                "discovery_pipeline": "—",
                "credible_face_candidate": False,
            }

        # Restore search-view metadata after face matching.
        candidate_metadata = {}
        for candidate in candidates:
            candidate_link = (candidate.get("link") or "").strip()
            if candidate_link:
                candidate_metadata[candidate_link] = candidate

        for item in match_results:
            metadata = candidate_metadata.get(
                (item.get("link") or "").strip(),
                {}
            )
            item["search_sources"] = metadata.get("search_sources", [])
            item["snippet"] = metadata.get("snippet", "")
            item["discovery_pipeline"] = metadata.get(
                "discovery_pipeline",
                item.get("discovery_pipeline", "Google Lens")
            )

        # No separate Instagram/social winner: the final result can be
        # Instagram, Facebook, X, YouTube, LinkedIn, Reddit, or any other
        # source returned by the search pipeline.

        # --------------------------------------------------
        # STEP 6–8 — FINGERPRINT + BLOCKCHAIN
        # --------------------------------------------------

        blockchain = Blockchain()
        set_pipeline_step(5, {1, 2, 3, 4})

        if best_match.get("credible_face_candidate", False):
            status.info("Creating SHA-256 verification fingerprint...")

            blockchain_match = best_match
            fingerprint = create_fingerprint(blockchain_match)
            progress.progress(85)

            status.info("Recording verification on local blockchain...")

            existing_block = None
            for block in blockchain.chain:
                block_data = block.data
                if block_data.get("fingerprint") == fingerprint:
                    existing_block = block
                    break

            if existing_block:
                verified_block = existing_block
            else:
                verified_block = blockchain.add_block({
                    "type": "verified_face_match",
                    "title": blockchain_match.get("title", "Unknown"),
                    "source": blockchain_match.get("source", "Unknown"),
                    "url": blockchain_match.get("link", ""),
                    "similarity": blockchain_match.get("similarity", 0),
                    "fingerprint": fingerprint
                })

            is_valid, verification_message = blockchain.verify_chain()
            set_pipeline_step(6, {1, 2, 3, 4, 5, 6})
        else:
            # Never write a fake "verified_face_match" block for an unrelated
            # low-similarity candidate. The input was processed, but identity
            # verification was not established.
            fingerprint = ""
            verified_block = None
            is_valid = False
            verification_message = "No credible face candidate cleared the similarity quality gate; nothing was recorded as a verified match."
            progress.progress(90)
            status.warning("No credible face match found. Blockchain verification was not recorded.")

        set_pipeline_step(5, {1, 2, 3, 4})
        progress.progress(90)
        set_pipeline_step(6, {1, 2, 3, 4, 5, 6})
        progress.progress(100)

        status.success("Pipeline completed — candidate discovery, face comparison, and verification finished.")

        # --------------------------------------------------
        # SAVE SESSION RESULT
        # --------------------------------------------------

        st.session_state.pipeline_result = {
            "input_hash": current_hash,
            "faces_detected": len(faces),
            "embedding_shape": str(embedding.shape),
            "lens_count": len(lens_candidates),
            "profile_count": len(profile_candidates),
            "crawler_count": len(crawler_candidates),
            "crawler_pages": crawler_response.get("pages_crawled", 0),
            "crawler_discovery_method": crawler_response.get("discovery_method", "unknown"),
            "search_mode": "Social-first Google Lens + authorized public-profile search + bounded public-web crawler",
            "public_profile_url": public_profile_url,
            "best_match": best_match,
            "social_match": None,
            "source_links": source_links,
            "fingerprint": fingerprint,
            "block": verified_block.to_dict() if verified_block is not None else None,
            "blockchain_valid": is_valid,
            "verification_message": verification_message
        }

    except Exception as e:

        progress.empty()

        st.error(
            f" Pipeline error: {str(e)}"
        )

        st.exception(e)



st.markdown(
    """
    <div class="hh-footer">
        <span>GOA, INDIA · PROJECTAIM</span>
        <span>HACKER HOUSE GOA · 28–31 OCT 2026</span>
        <span>SKOPEO · BUILT FOR DEMONSTRATION</span>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

result = st.session_state.pipeline_result

if result:

    st.divider()

    st.header("Verification Results")

    # --------------------------------------------------
    # PIPELINE SUMMARY
    # --------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Faces Detected", result["faces_detected"])

    with col2:
        st.metric("Lens Matches", result["lens_count"])

    with col3:
        st.metric("Crawler Matches", result.get("crawler_count", 0))

    with col4:
        if result["best_match"].get("credible_face_candidate", False):
            st.metric("Best Similarity", f'{result["best_match"]["similarity"]:.4f}')
        else:
            st.metric("Best Similarity", "—")

    st.caption(f"Crawler pages visited: {result.get('crawler_pages', 0)}")
    st.caption(
        f" Web search method used: **{result.get('search_mode', 'unknown')}**"
    )

    # --------------------------------------------------
    # DISCOVERY TABS
    # --------------------------------------------------

    lens_tab, profile_tab, crawler_tab = st.tabs(["Google Lens", "Authorized Profile", "Open-Web Crawler"])

    lens_results = [
        item for item in all_results if item.get("discovery_pipeline") == "Google Lens"
    ] if False else []

    try:
        with open("data/face_match_results.json", "r", encoding="utf-8") as f:
            tab_matches = json.load(f)
    except Exception:
        tab_matches = []

    with lens_tab:
        lens_tab_results = [
            x for x in tab_matches
            if x.get("discovery_pipeline") == "Google Lens"
        ]
        lens_tab_results.sort(
            key=lambda x: (
                0 if social_platform(x.get("link", "")) else 1,
                social_priority(x.get("link", "")),
                0 if x.get("social_targeted") else 1,
                -float(x.get("similarity", 0)),
            )
        )
        st.metric("Google Lens face matches", len(lens_tab_results))
        if lens_tab_results:
            for i, item in enumerate(lens_tab_results[:10], 1):
                st.markdown(f"**{i}. {item.get('title', 'Untitled')}**")
                st.caption(f"Similarity: {item.get('similarity', 0):.4f} · {item.get('source', 'Web source')}")
                if item.get("link"):
                    st.markdown(f"[ Open source page]({item['link']})")
        else:
            st.info("No Google Lens candidates could be face-matched.")

    with profile_tab:
        profile_tab_results = [
            x for x in tab_matches
            if x.get("discovery_pipeline") == "Public Profile Search"
        ]
        st.metric("Authorized-profile face matches", len(profile_tab_results))
        if result.get("public_profile_url"):
            if profile_tab_results:
                for i, item in enumerate(profile_tab_results[:15], 1):
                    st.markdown(f"**{i}. {item.get('title', 'Public profile result')}**")
                    st.caption(f"Similarity: {item.get('similarity', 0):.4f} · {item.get('source', 'Public profile')}")
                    if item.get("link"):
                        st.markdown(f"[ Open public profile result]({item['link']})")
            else:
                st.info("No face-matched material was returned from the supplied public profile.")
        else:
            st.info("For an ordinary public user, enter the exact public profile URL in the sidebar. Image-only reverse search is not a reliable way to discover an arbitrary non-celebrity account.")

    with crawler_tab:
        crawler_tab_results = [
            x for x in tab_matches
            if x.get("discovery_pipeline") == "Open-Web Crawler"
        ]
        st.metric("Crawler face matches", len(crawler_tab_results))
        st.caption("Crawler priority: Instagram → Facebook → X → YouTube → Reddit → LinkedIn → Pinterest")
        if crawler_tab_results:
            st.caption("The crawler follows public links beyond its search roots within a bounded depth/page budget.")
        else:
            st.info("No crawler candidates could be face-matched from the public search roots.")
        if crawler_tab_results:
            for i, item in enumerate(crawler_tab_results[:10], 1):
                st.markdown(f"**{i}. {item.get('title', 'Untitled')}**")
                st.caption(f"Similarity: {item.get('similarity', 0):.4f} · {item.get('source', 'Web source')}")
                if item.get("link"):
                    st.markdown(f"[ Open crawled page]({item['link']})")
        else:
            st.info("No crawler candidates could be face-matched.")

    st.divider()

    # --------------------------------------------------
    # BEST FACE MATCH
    # --------------------------------------------------

    st.subheader("Highest Face-Similarity Candidate")

    best = result["best_match"]

    if not best.get("credible_face_candidate", True):
        st.warning("No candidate cleared the face-similarity quality gate. The system will not promote a low-confidence image as the identity match.")

    col1, col2 = st.columns([1, 2])

    with col1:

        if best.get("image_url"):

            try:

                st.image(
                    best["image_url"],
                    caption="Candidate Image",
                    width=300
                )

            except Exception:
                st.info("Candidate image unavailable.")

    with col2:

        st.write(
            f"**Title:** {best.get('title', 'N/A')}"
        )

        st.write(
            f"**Source:** {best.get('source', 'N/A')}"
        )

        st.write(
            f"**Face similarity:** "
            f"{best.get('similarity', 0):.4f}"
        )

        st.write(
            f"**Faces detected:** "
            f"{best.get('faces_detected', 0)}"
        )

        if best.get("link"):

            st.markdown(
                f"[ Open source page]({best['link']})"
            )

    # --------------------------------------------------
    # VERIFIED WEB CANDIDATE
    # --------------------------------------------------

    st.subheader("Best Web Candidate")

    best = result["best_match"]
    best_link = best.get("link", "")

    col1, col2 = st.columns([1, 2])

    with col1:
        if best.get("image_url"):
            try:
                st.image(
                    best["image_url"],
                    caption="Best candidate image",
                    width=300
                )
            except Exception:
                st.info("Candidate image unavailable.")

    with col2:
        st.write(f"**Title:** {best.get('title', 'N/A')}")
        st.write(f"**Source:** {best.get('source', 'N/A')}")
        st.write(f"**Face similarity:** {best.get('similarity', 0):.4f}")
        st.write(f"**Identity score:** {best.get('identity_score', best.get('similarity', 0)):.4f}")
        st.write(f"**Faces detected:** {best.get('faces_detected', 0)}")

        if is_social_url(best_link):
            st.success(" This candidate is from a social-media source.")

        if best_link:
            st.markdown(f"[ Open source page]({best_link})")

    st.info(
        "Search results are candidate matches. "
        "Face similarity is used to rank candidates; it is not "
        "a standalone proof of real-world identity."
    )

    # --------------------------------------------------
    # TOP MATCHES
    # --------------------------------------------------

    st.subheader("Top Face-Matching Results")

    table_data = []

    # We only show top 10 here.
    # The complete results remain in data/face_match_results.json.

    try:

        with open(
            "data/face_match_results.json",
            "r",
            encoding="utf-8"
        ) as f:

            all_results = json.load(f)

    except Exception:

        all_results = [best]

    for item in all_results[:10]:

        table_data.append(
            {
                "Rank": item.get("rank"),
                "Source": item.get("source"),
                "Similarity": round(
                    item.get("similarity", 0),
                    4
                ),
                "Identity Score": round(
                    item.get("identity_score", item.get("similarity", 0)),
                    4
                ),
                "Social": "Yes" if is_social_url(item.get("link", "")) else "No",
                "Faces": item.get(
                    "faces_detected",
                    0
                )
            }
        )

    st.dataframe(
        table_data,
        width="stretch",
        hide_index=True
    )

    # --------------------------------------------------
    # VISITABLE MATCH URLS
    # --------------------------------------------------

    st.subheader("Visitable Match URLs")

    st.caption(
        "These source-page URLs come directly from Google Lens. "
        "They are preserved even when a website blocks downloading "
        "its image for face matching."
    )

    # Start with every Lens source link so social/web pages are not
    # lost just because their image could not be downloaded. Then add
    # any face-matched links that were not already present.
    visitable_links = []
    seen_links = set()

    for item in result.get("source_links", []):

        link = item.get("link", "")

        if not link or link in seen_links:
            continue

        seen_links.add(link)
        visitable_links.append(item)

    for item in all_results if 'all_results' in locals() else [best]:

        link = item.get("link", "")

        if not link or link in seen_links:
            continue

        seen_links.add(link)
        visitable_links.append({
            "title": item.get("title", "Face match"),
            "source": item.get("source", "Web source"),
            "link": link,
            "similarity": item.get("similarity"),
            "faces_detected": item.get("faces_detected")
        })

    if visitable_links:

        for i, item in enumerate(visitable_links[:20], start=1):

            st.markdown(
                f"**Match {i} — {item.get('source', 'Web source')}**"
            )

            if item.get("similarity") is not None:

                st.caption(
                    f"Similarity: {item.get('similarity', 0):.4f}  |  "
                    f"Faces detected: {item.get('faces_detected', 0)}"
                )

            elif item.get("title"):

                st.caption(
                    f"{item.get('title')} — source link preserved from "
                    "Google Lens"
                )

            st.markdown(
                f"[ Visit match result]({item['link']})"
            )

            st.code(
                item["link"],
                language="text"
            )

    else:

        st.info(
            "No visitable result URLs were returned."
        )

    # --------------------------------------------------
    # BLOCKCHAIN
    # --------------------------------------------------

    st.header("Blockchain Verification")

    block = result.get("block")

    if block is not None:
        st.write(f"**Block Index:** {block.get('index', '—')}")
        st.write(f"**Previous Hash:** `{block.get('previous_hash', '')}`")
        st.write(f"**Block Hash:** `{block.get('hash', '')}`")
        st.write(f"**SHA-256 Fingerprint:** `{result.get('fingerprint', '')}`")

        if result.get("blockchain_valid", False):
            st.success(
                "Blockchain re-verification successful. The chain is valid and the recorded fingerprint has not been altered."
            )
        else:
            st.error("Blockchain verification failed.")
    else:
        st.warning(
            "No blockchain record was created because no credible face match was established."
        )
        st.caption(
            result.get(
                "verification_message",
                "No verified face match was available for blockchain recording."
            )
        )

    # --------------------------------------------------
    # FINAL PIPELINE
    # --------------------------------------------------

    st.divider()
    st.subheader("Pipeline Status")
    st.markdown(
        "**Face Scan** → **Face Detection** → **Face Encoding** → **Web Search** → "
        "**Face Matching** → **Social/Web Candidate Search** → **SHA-256 Fingerprint** → "
        "**Blockchain Upload** → **Blockchain Re-verification**"
    )

