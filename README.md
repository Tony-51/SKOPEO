# SKOPEO — Face → Web → Blockchain Verifier

SKOPEO is a demonstration application that combines **face detection, face encoding, public-web discovery, face similarity matching, evidence verification, and blockchain-style evidence preservation** in one Streamlit interface.

The project is intended for authorized demonstrations and verification workflows. It is **not** designed to identify arbitrary private individuals or to bypass authentication, CAPTCHAs, paywalls, robots.txt restrictions, or other access controls.

## What the Project Does

The application follows this pipeline:

1. **Face Detection — YuNet**
   - Detects faces in the uploaded image.
   - Selects the detected face for the downstream workflow.

2. **Face Encoding — SFace**
   - Generates a numerical face embedding from the detected face.
   - The embedding is used for similarity comparison rather than storing a person's name.

3. **Web Candidate Search — Google Lens via SerpApi**
   - The uploaded face image is compressed and uploaded to SerpApi once.
   - The resulting image ID is reused for multiple Google Lens searches to reduce redundant uploads and network latency.
   - Social platforms are prioritized in the search strategy:
     **Instagram → Facebook → X → YouTube → Reddit → LinkedIn → Pinterest**.

4. **Public Profile / Web Discovery**
   - A public profile URL can be supplied when verifying an ordinary public user.
   - The public-web crawler can follow links from configured public social search roots and discovered public pages.
   - JavaScript rendering is supported through Playwright when available, which helps with modern social pages whose links are not present in the initial HTML.

5. **Face Matching — Cosine Similarity**
   - Candidate images containing detectable faces are compared against the uploaded face embedding.
   - Only candidates meeting the application's matching criteria are treated as face-match evidence.

6. **Verification — SHA-256**
   - The selected result is converted into a deterministic JSON representation.
   - A SHA-256 fingerprint is generated from that evidence.

7. **Blockchain — Local Simulated Blockchain**
   - The verified result and fingerprint are written to a locally maintained blockchain-like chain.
   - Each block contains an index, timestamp, data, previous block hash, and its own SHA-256 hash.
   - The chain is re-verified by checking both block hashes and previous-hash links.

## Blockchain Used

This project uses a **custom local simulated blockchain implemented in Python**. It is **not connected to Ethereum, Polygon, Solana, Bitcoin, or another public blockchain**.

The blockchain is stored locally in:

```text
data/blockchain.json
```

Each block follows the basic structure:

```text
Block Index
Timestamp
Data
Previous Hash
Current Hash
```

The block hash is calculated using SHA-256 over the serialized block contents. This provides an auditable demonstration of hash chaining and evidence integrity without requiring a wallet, gas fees, RPC provider, or external blockchain network.

## Requirements

- Python 3.10 or newer
- Streamlit
- OpenCV
- NumPy
- Pillow
- Requests
- BeautifulSoup4
- python-dotenv
- SerpApi API key
- Playwright (recommended for JavaScript-heavy public pages)

The existing project `requirements.txt` should be used as the primary dependency list. If Playwright is not already installed, install it separately:

```powershell
pip install playwright
playwright install chromium
```

## Configuration

### Local development

Create or update the `.env` file in the project root:

```env
SERPAPI_KEY=your_serpapi_key_here
```

Do not commit `.env` or expose the API key in source code.

### Streamlit Cloud

For a Streamlit Cloud deployment, add the key as a top-level secret:

```toml
SERPAPI_KEY = "your_serpapi_key_here"
```

The application checks the environment variable first and can also read the Streamlit secret.

## How to Run

### Windows

Open PowerShell in the project directory.

Create a virtual environment if you do not already have one:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

If using the JavaScript crawler:

```powershell
pip install playwright
playwright install chromium
```

Start the application:

```powershell
python -m streamlit run app.py
```

The Streamlit interface will normally open at:

```text
http://localhost:8501
```

The repository also includes `run_project.ps1` for a simplified Windows launch workflow.

## Judge Quick Start

1. Activate the virtual environment.
2. Make sure the SerpApi key is configured.
3. Start the application with `python -m streamlit run app.py`.
4. Upload an authorized image containing a clearly visible face.
5. Click **RUN COMPLETE PIPELINE**.
6. For an ordinary public user, supplying the exact public profile URL is recommended because an image-only reverse search cannot guarantee discovery of an unknown non-celebrity account.
7. Follow the six pipeline stages in the sidebar.
8. Review the matched public-web evidence and the final blockchain verification.

## Search-Speed Design

The discovery branch has been optimized to avoid unnecessary repeated work:

- The image is uploaded to SerpApi once and the returned image ID is reused.
- Multiple Lens searches can run concurrently rather than sequentially.
- Social platforms are ranked before general web results.
- Robots checks for seed domains are performed concurrently.
- Public crawler requests use bounded timeouts and a limited crawl budget.
- Playwright rendering is used selectively for social pages instead of rendering every page.

Actual runtime depends heavily on network speed, SerpApi response time, public-site availability, rate limits, and whether JavaScript rendering is required. The application therefore aims for a substantially faster demo path rather than guaranteeing a fixed response time for every search.

## Known Limitations

### 2. Search results depend on SerpApi / Google Lens

The project does not control Google's index or ranking. Results can vary between images and over time. SerpApi quota limits and API failures can also affect discovery.

### 3. Social sites restrict automated access

Instagram, Facebook, LinkedIn, X, YouTube, Reddit, Pinterest, and other sites may require authentication, return different content to automated clients, rate-limit requests, or block automated crawling.

The crawler does **not** bypass these controls.

### 4. robots.txt is respected

The crawler checks `robots.txt` and does not intentionally ignore explicit crawling restrictions. If a public site denies the crawler, that page may not be available as a candidate.

## Responsible Use

Use SKOPEO only with images and public information that you are authorized to process. Do not use the system to stalk, harass, impersonate, or make high-impact decisions about people. Face similarity results should always be independently reviewed before any consequential decision.

## Project Summary

**SKOPEO** demonstrates how a face-verification workflow can combine:

```text
Face Image
   ↓
YuNet Face Detection
   ↓
SFace Face Encoding
   ↓
Social-First Google Lens / Public Web Discovery
   ↓
Face Similarity Matching
   ↓
Evidence Verification
   ↓
SHA-256 Fingerprint
   ↓
Local Hash-Linked Blockchain
   ↓
Blockchain Re-verification
```

---

### Team Kuromi

**Sanskriti Shukla · Tejas Thorat · Palak Upadhyaya**