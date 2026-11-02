import html
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv
import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TruthLens: Intelligent News Verification System",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "").strip()
GOOGLE_FACTCHECK_API_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "").strip()

MODEL_PATH = BASE_DIR / "calibrated_svm_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"


# ============================================================
# 2. LOAD MODEL AND VECTORIZER
# ============================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


try:
    model, vectorizer = load_artifacts()

except FileNotFoundError as error:
    st.error(
        f"Required model file not found: {error.filename}. "
        "Keep both .pkl files in the same folder as truthlens_app.py."
    )
    st.stop()

except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# ============================================================
# 2A. LIVE NEWS AND FACT-CHECK SEARCH
# ============================================================

def make_search_queries(article_text: str) -> list[str]:
    """Build progressively broader NewsAPI queries so long text is not over-restrictive."""
    text = re.sub(r"\s+", " ", article_text).strip()
    first_part = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", first_part)

    stop_words = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "be", "been", "being", "to", "of", "in", "on", "at", "for", "from",
        "by", "with", "as", "near", "after", "before", "during", "that",
        "this", "it", "its", "has", "have", "had", "successfully"
    }
    keywords = []
    seen = set()
    for token in tokens:
        normalized = token.lower()
        if normalized not in stop_words and normalized not in seen:
            keywords.append(token)
            seen.add(normalized)

    queries = []
    # Try a small set of informative terms first, then broaden if no results.
    if len(keywords) >= 3:
        queries.append(" AND ".join(keywords[:4]))
    if len(keywords) >= 2:
        queries.append(" AND ".join(keywords[:2]))
    if keywords:
        queries.append(keywords[0])
    # If keyword extraction is too aggressive, retain a short headline fallback.
    if tokens:
        queries.append(" ".join(tokens[:6]))

    unique_queries = []
    for query in queries:
        if query and query not in unique_queries:
            unique_queries.append(query)
    return unique_queries


def search_related_news(article_text: str, page_size: int = 5) -> dict:
    if not NEWS_API_KEY:
        return {"status": "not_configured", "items": [], "message": "NEWS_API_KEY is missing from .env."}

    queries = make_search_queries(article_text)
    if not queries:
        return {"status": "empty_query", "items": [], "message": "Could not create a search query."}

    try:
        last_payload = {}
        for query in queries:
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "language": "en",
                    "sortBy": "relevancy",
                    "pageSize": page_size,
                },
                headers={"X-Api-Key": NEWS_API_KEY},
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("status") != "ok":
                return {
                    "status": "error",
                    "items": [],
                    "message": payload.get("message", "NewsAPI request failed."),
                }

            last_payload = payload
            articles = payload.get("articles", [])
            if articles:
                items = []
                for article in articles:
                    items.append({
                        "title": article.get("title") or "Untitled article",
                        "source": (article.get("source") or {}).get("name") or "Unknown source",
                        "url": article.get("url") or "",
                        "publishedAt": article.get("publishedAt") or "",
                        "description": article.get("description") or "",
                    })
                return {"status": "ok", "items": items, "message": "", "query_used": query}

        return {
            "status": "ok",
            "items": [],
            "message": "No articles matched the progressively broadened queries.",
            "queries_tried": queries,
        }
    except requests.RequestException as exc:
        return {"status": "error", "items": [], "message": str(exc)}
    except (ValueError, TypeError) as exc:
        return {"status": "error", "items": [], "message": str(exc)}


def make_search_query(article_text: str, max_words: int = 12) -> str:
    """Create a compact query from informative words in the submitted text."""
    text = re.sub(r"\s+", " ", article_text).strip()
    first_part = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", first_part)
    stop_words = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "be", "been", "being", "to", "of", "in", "on", "at", "for", "from",
        "by", "with", "as", "near", "after", "before", "during", "that",
        "this", "it", "its", "has", "have", "had", "successfully"
    }
    keywords = []
    seen = set()
    for token in tokens:
        normalized = token.lower()
        if normalized not in stop_words and normalized not in seen:
            keywords.append(token)
            seen.add(normalized)
    return " ".join(keywords[:max_words]) or " ".join(tokens[:max_words])


def search_fact_checks(article_text: str, page_size: int = 5) -> dict:
    if not GOOGLE_FACTCHECK_API_KEY:
        return {"status": "not_configured", "items": [], "message": "GOOGLE_FACTCHECK_API_KEY is missing from .env."}
    query = make_search_query(article_text, max_words=18)
    if not query:
        return {"status": "empty_query", "items": [], "message": "Could not create a search query."}
    try:
        response = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={
                "query": query,
                "languageCode": "en",
                "pageSize": page_size,
                "key": GOOGLE_FACTCHECK_API_KEY,
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        items = []
        for claim in payload.get("claims", []):
            for review in claim.get("claimReview", []):
                items.append({
                    "claim": claim.get("text") or "Claim text unavailable",
                    "publisher": review.get("publisher", {}).get("name") or "Unknown publisher",
                    "title": review.get("title") or "Fact-check review",
                    "rating": review.get("textualRating") or "Rating not provided",
                    "url": review.get("url") or "",
                    "date": review.get("reviewDate") or "",
                })
                if len(items) >= page_size:
                    break
            if len(items) >= page_size:
                break
        return {"status": "ok", "items": items, "message": ""}
    except requests.RequestException as exc:
        return {"status": "error", "items": [], "message": str(exc)}
    except (ValueError, TypeError) as exc:
        return {"status": "error", "items": [], "message": str(exc)}


def verify_online(article_text: str) -> dict:
    return {
        "news": search_related_news(article_text),
        "fact_checks": search_fact_checks(article_text),
    }


# ============================================================
# 3. CUSTOM CSS — ULTRA-ADVANCED CYBER-LUX THEME
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700;800&display=swap');

    :root {
        --bg-deep: #020617;
        --panel-bg: rgba(15, 23, 42, 0.82);
        --border-glow: rgba(56, 189, 248, 0.28);
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
        --neon-cyan: #38bdf8;
        --neon-purple: #a855f7;
        --neon-blue: #6366f1;
        --neon-green: #10b981;
        --neon-red: #f43f5e;
        --neon-amber: #f59e0b;
    }

    /* Multi-layered cinematic dark backdrop with glowing mesh highlights */
    .stApp {
        background-color: var(--bg-deep);
        background-image: 
            radial-gradient(circle at 10% 10%, rgba(56, 189, 248, 0.15), transparent 40%),
            radial-gradient(circle at 90% 20%, rgba(168, 85, 247, 0.12), transparent 40%),
            radial-gradient(circle at 50% 85%, rgba(16, 185, 129, 0.08), transparent 45%),
            linear-gradient(145deg, #020617 0%, #090d1f 50%, #02040a 100%);
        color: var(--text-main);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main .block-container {
        max-width: 1360px;
        padding-top: 2.2rem;
        padding-bottom: 5rem;
    }

    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.5px;
    }

    /* Top Navigation Header — Enlarged & Ultra Styled */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 20px;
        padding: 22px 32px;
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 24px;
        background: var(--panel-bg);
        backdrop-filter: blur(30px);
        margin-bottom: 38px;
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }

    .brand-container {
        display: flex;
        align-items: center;
        gap: 20px;
    }

    .brand-logo {
        width: 60px;
        height: 60px;
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
        background: linear-gradient(135deg, #0284c7, #8b5cf6);
        border: 1px solid rgba(56, 189, 248, 0.6);
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.4);
    }

    .brand-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 26px;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.15;
        letter-spacing: -0.5px;
    }

    .brand-title span {
        background: linear-gradient(90deg, #38bdf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .brand-subtitle {
        color: var(--neon-cyan);
        font-size: 12px;
        letter-spacing: 2.2px;
        text-transform: uppercase;
        font-weight: 700;
        margin-top: 4px;
        text-shadow: 0 0 15px rgba(56, 189, 248, 0.3);
    }

    .system-status {
        padding: 10px 18px;
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 30px;
        background: rgba(16, 185, 129, 0.1);
        color: #34d399;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.2px;
        display: flex;
        align-items: center;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        margin-right: 9px;
        box-shadow: 0 0 14px #10b981;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 0.8; }
        50% { transform: scale(1.3); opacity: 1; box-shadow: 0 0 20px #10b981; }
        100% { transform: scale(0.95); opacity: 0.8; }
    }

    /* Hero Section */
    .hero-box {
        text-align: center;
        padding: 35px 10px 45px;
    }

    .hero-badge {
        display: inline-block;
        border: 1px solid rgba(168, 85, 247, 0.4);
        background: rgba(168, 85, 247, 0.08);
        color: #d8b4fe;
        border-radius: 50px;
        padding: 8px 20px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin-bottom: 22px;
        box-shadow: 0 0 25px rgba(168, 85, 247, 0.2);
    }

    .hero-main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(38px, 6vw, 64px);
        font-weight: 800;
        line-height: 1.12;
        letter-spacing: -2px;
        color: #ffffff;
        margin-bottom: 18px;
    }

    .hero-main-title .gradient-text {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 40%, #c084fc 80%, #f43f5e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-desc {
        max-width: 700px;
        margin: 0 auto;
        color: var(--text-muted);
        font-size: 16px;
        line-height: 1.8;
    }

    /* Pipeline Architecture Grid */
    .pipeline-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 18px;
        margin: 35px 0 45px;
    }

    .pipeline-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid var(--border-glow);
        border-radius: 20px;
        padding: 22px 16px;
        text-align: center;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .pipeline-card:hover {
        border-color: rgba(56, 189, 248, 0.7);
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(56, 189, 248, 0.2);
        background: rgba(20, 30, 55, 0.85);
    }

    .pipeline-emoji {
        font-size: 26px;
        margin-bottom: 12px;
    }

    .pipeline-title {
        font-family: 'Space Grotesk', sans-serif;
        color: #f1f5f9;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .pipeline-subtitle {
        color: var(--text-muted);
        font-size: 11px;
    }

    /* Section Headers */
    .sec-tag {
        color: var(--neon-cyan);
        font-size: 11px;
        letter-spacing: 2.2px;
        text-transform: uppercase;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .sec-heading {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 25px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 6px;
    }

    .sec-subtext {
        font-size: 13px;
        color: var(--text-muted);
        line-height: 1.7;
    }

    /* Text Area Customization */
    div[data-testid="stTextArea"] textarea {
        background: #040914 !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-radius: 18px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 14px !important;
        line-height: 1.8 !important;
        padding: 20px !important;
        box-shadow: inset 0 3px 6px rgba(0,0,0,0.6);
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: var(--neon-cyan) !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.3), inset 0 3px 6px rgba(0,0,0,0.6) !important;
    }

    /* Custom Buttons */
    .stButton > button {
        border-radius: 16px;
        border: 1px solid rgba(56, 189, 248, 0.35);
        background: rgba(56, 189, 248, 0.08);
        color: #7dd3fc;
        font-weight: 700;
        transition: all 0.25s ease;
        min-height: 48px;
    }

    .stButton > button:hover {
        border-color: var(--neon-cyan);
        background: rgba(56, 189, 248, 0.2);
        color: #ffffff;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(56, 189, 248, 0.3);
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7, #7c3aed);
        color: white;
        border: none;
        box-shadow: 0 10px 30px rgba(124, 58, 237, 0.4);
    }

    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 14px 40px rgba(56, 189, 248, 0.5);
        background: linear-gradient(135deg, #0369a1, #6d28d9);
    }

    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.8);
        padding: 20px;
        border-radius: 18px;
        border: 1px solid var(--border-glow);
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-muted);
        font-size: 12px;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* ========================================================
       ULTRA-STYLISH GLOWING VERDICT BANNERS (REAL VS FAKE)
       ======================================================== */
    .verdict-banner {
        border-radius: 28px;
        padding: 42px;
        margin: 40px 0;
        position: relative;
        overflow: hidden;
        animation: bannerReveal 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 30px 70px rgba(0, 0, 0, 0.6);
    }

    /* REAL NEWS BANNER */
    .verdict-banner.real {
        background: radial-gradient(circle at 85% 15%, rgba(16, 185, 129, 0.3), transparent 55%),
                    linear-gradient(135deg, rgba(4, 47, 46, 0.94), rgba(2, 30, 26, 0.98));
        border: 1px solid rgba(16, 185, 129, 0.6);
        border-left: 9px solid var(--neon-green);
        box-shadow: 0 0 50px rgba(16, 185, 129, 0.25), inset 0 0 35px rgba(16, 185, 129, 0.06);
    }

    /* FAKE NEWS BANNER */
    .verdict-banner.fake {
        background: radial-gradient(circle at 85% 15%, rgba(244, 63, 94, 0.35), transparent 55%),
                    linear-gradient(135deg, rgba(136, 19, 55, 0.94), rgba(76, 5, 25, 0.98));
        border: 1px solid rgba(244, 63, 94, 0.7);
        border-left: 9px solid var(--neon-red);
        box-shadow: 0 0 50px rgba(244, 63, 94, 0.3), inset 0 0 35px rgba(244, 63, 94, 0.06);
    }

    .verdict-flex {
        display: flex;
        align-items: center;
        gap: 30px;
    }

    .verdict-icon-frame {
        width: 92px;
        height: 92px;
        border-radius: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 46px;
        font-weight: 900;
        flex-shrink: 0;
    }

    .verdict-banner.real .verdict-icon-frame {
        background: rgba(16, 185, 129, 0.18);
        border: 1px solid rgba(16, 185, 129, 0.5);
        color: var(--neon-green);
        box-shadow: 0 0 35px rgba(16, 185, 129, 0.4);
    }

    .verdict-banner.fake .verdict-icon-frame {
        background: rgba(244, 63, 94, 0.18);
        border: 1px solid rgba(244, 63, 94, 0.5);
        color: var(--neon-red);
        box-shadow: 0 0 35px rgba(244, 63, 94, 0.45);
    }

    .verdict-tagline {
        font-size: 11px;
        letter-spacing: 2.8px;
        font-weight: 800;
        margin-bottom: 6px;
        text-transform: uppercase;
    }

    .verdict-banner.real .verdict-tagline { color: #34d399; }
    .verdict-banner.fake .verdict-tagline { color: #fb7185; }

    .verdict-title-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(36px, 5.5vw, 54px);
        font-weight: 800;
        letter-spacing: -1px;
        line-height: 1.1;
        margin: 0;
        text-transform: uppercase;
    }

    .verdict-banner.real .verdict-title-text {
        color: #10b981;
        text-shadow: 0 0 40px rgba(16, 185, 129, 0.6);
    }

    .verdict-banner.fake .verdict-title-text {
        color: #f43f5e;
        text-shadow: 0 0 40px rgba(244, 63, 94, 0.7);
    }

    .verdict-description {
        color: #e2e8f0;
        font-size: 15px;
        line-height: 1.75;
        margin-top: 14px;
        max-width: 750px;
    }

    .verdict-footer-msg {
        margin-top: 24px;
        padding-top: 18px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 12px;
        color: var(--text-muted);
        display: flex;
        align-items: center;
        gap: 10px;
    }

    @keyframes bannerReveal {
        0% { opacity: 0; transform: translateY(20px) scale(0.98); }
        100% { opacity: 1; transform: translateY(0) scale(1); }
    }

    /* Info Mini Cards */
    .info-box-glass {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid var(--border-glow);
        border-radius: 20px;
        padding: 24px;
        height: 100%;
        transition: all 0.25s ease;
    }

    .info-box-glass:hover {
        border-color: rgba(56, 189, 248, 0.5);
        transform: translateY(-2px);
    }

    .info-box-icon {
        font-size: 24px;
        margin-bottom: 12px;
    }

    .info-box-title {
        color: #f1f5f9;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 6px;
        font-family: 'Space Grotesk', sans-serif;
    }

    .info-box-text {
        color: var(--text-muted);
        font-size: 12px;
        line-height: 1.7;
    }

    @media (max-width: 650px) {
        .pipeline-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .verdict-flex { flex-direction: column; align-items: flex-start; gap: 20px; }
        .verdict-banner { padding: 26px; }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 4. HEADER WITH ENLARGED PROJECT NAME
# ============================================================

st.markdown(
    """
    <div class="topbar">
        <div class="brand-container">
            <div class="brand-logo">🔎</div>
            <div>
                <div class="brand-title">Truth<span>Lens</span></div>
                <div class="brand-subtitle">Intelligent News Verification System</div>
            </div>
        </div>
        <div class="system-status">
            <span class="pulse-dot"></span>
            AI ENGINE ACTIVE
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 5. HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-badge">✦ Advanced AI Misinformation Detection Pipeline</div>
        <div class="hero-main-title">
            Beyond the headlines.<br>
            <span class="gradient-text">Closer to the absolute truth.</span>
        </div>
        <div class="hero-desc">
            TruthLens analyzes linguistic markers, structural syntax, and vector embeddings in real-time 
            using high-performance Machine Learning to evaluate news authenticity.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 6. PIPELINE ARCHITECTURE VISUALIZATION
# ============================================================

st.markdown(
    """
    <div class="pipeline-grid">
        <div class="pipeline-card">
            <div class="pipeline-emoji">📰</div>
            <div class="pipeline-title">News Input</div>
            <div class="pipeline-subtitle">Headline or Full Text</div>
        </div>
        <div class="pipeline-card">
            <div class="pipeline-emoji">🔡</div>
            <div class="pipeline-title">TF-IDF Vectorizer</div>
            <div class="pipeline-subtitle">Token Embedding Space</div>
        </div>
        <div class="pipeline-card">
            <div class="pipeline-emoji">🧠</div>
            <div class="pipeline-title">Linear SVM</div>
            <div class="pipeline-subtitle">Probability Calibration</div>
        </div>
        <div class="pipeline-card">
            <div class="pipeline-emoji">📊</div>
            <div class="pipeline-title">Intelligent Verdict</div>
            <div class="pipeline-subtitle">Credibility Assessment</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 7. INTERACTIVE INPUT PANEL & SIDEBAR INFO
# ============================================================

left_col, right_col = st.columns([1.45, 0.75], gap="large")

with left_col:
    st.markdown(
        """
        <div class="sec-tag">01 / Analysis Input</div>
        <div class="sec-heading">Examine article authenticity</div>
        <div class="sec-subtext">
            Input a news headline or complete article text to begin scanning.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    if "news_text" not in st.session_state:
        st.session_state.news_text = ""

    article_text = st.text_area(
        "News article input",
        key="news_text",
        height=260,
        placeholder=(
            "Paste your news headline or full article text here...\n\n"
            "Example: Researchers publish a comprehensive peer-reviewed study detailing "
            "methodology, empirical results, and recognized study constraints."
        ),
        label_visibility="collapsed"
    )

    b1, b2, b3 = st.columns([1.3, 1, 0.85])

    with b1:
        analyze_clicked = st.button("✦ Analyze Story", type="primary", use_container_width=True)

    with b2:
        sample_clicked = st.button("↗ Try Sample", use_container_width=True)

    with b3:
        clear_clicked = st.button("Clear", use_container_width=True)


with right_col:
    st.markdown(
        """
        <div class="sec-tag">02 / System Overview</div>
        <div class="sec-heading">Core Engine Spec</div>
        <div class="sec-subtext">
            High-performance natural language classifier.
        </div>
        <br>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box-glass" style="margin-bottom: 14px;">
            <div class="info-box-icon">⚡</div>
            <div class="info-box-title">High-Speed Vector Mapping</div>
            <div class="info-box-text">Converts textual articles instantly into weighted numeric feature matrices.</div>
        </div>
        <div class="info-box-glass" style="margin-bottom: 14px;">
            <div class="info-box-icon">🎯</div>
            <div class="info-box-title">Calibrated SVM Separation</div>
            <div class="info-box-text">Applies robust margin optimization to compute precise confidence intervals.</div>
        </div>
        <div class="info-box-glass">
            <div class="info-box-icon">🛡️</div>
            <div class="info-box-title">Fact-Checking Shield</div>
            <div class="info-box-text">Serves as a rapid digital screening utility to flag potential misinformation.</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 8. BUTTON ACTIONS
# ============================================================

if sample_clicked:
    st.session_state.news_text = (
        "Researchers published a new peer-reviewed study outlining experimental protocols, "
        "quantitative metrics, and recognized limitations. Experts note that further independent "
        "trials will be necessary to substantiate broader applications."
    )
    st.rerun()

if clear_clicked:
    st.session_state.news_text = ""
    st.session_state.pop("last_prediction", None)
    st.session_state.pop("last_probabilities", None)
    st.session_state.pop("last_feature_count", None)
    st.session_state.pop("online_results", None)
    st.rerun()


# ============================================================
# 9. PREDICTION EXECUTION
# ============================================================

if analyze_clicked:
    cleaned_text = article_text.strip()

    if not cleaned_text:
        st.warning("Please enter a headline or article text before initiating analysis.")
    elif len(cleaned_text.split()) < 3:
        st.warning("Please provide a longer text snippet for accurate feature extraction.")
    else:
        with st.spinner("TruthLens Neural Engine running feature classification..."):
            try:
                transformed_text = vectorizer.transform([cleaned_text])

                probabilities = None
                if hasattr(model, "predict_proba"):
                    raw_probabilities = model.predict_proba(transformed_text)[0]
                    probabilities = {
                        int(label): float(probability)
                        for label, probability in zip(model.classes_, raw_probabilities)
                    }

                    # Conservative FAKE flagging: only label an article FAKE when
                    # the model's class-1 score reaches this threshold. This is
                    # intended to reduce false-positive FAKE labels for genuine
                    # news, but it can allow some fake articles through.
                    FAKE_THRESHOLD = 0.85
                    fake_score = probabilities.get(1, 0.0)
                    prediction = 1 if fake_score >= FAKE_THRESHOLD else 0
                else:
                    # Fallback for classifiers without probability support.
                    prediction = model.predict(transformed_text)[0]

                st.session_state.last_prediction = int(prediction)
                st.session_state.last_probabilities = probabilities
                st.session_state.last_feature_count = int(transformed_text.nnz)

                # Live search runs separately from the ML prediction.
                with st.spinner("Searching live news and fact-check databases..."):
                    st.session_state.online_results = verify_online(cleaned_text)

            except Exception as error:
                st.error(f"Analysis failed. Ensure model artifacts match. Error: {error}")


# ============================================================
# 10. STYLISH VERDICT DISPLAY (REAL VS FAKE)

st.info("Conservative screening mode: an article is flagged FAKE only when its model score is at least 85%. This reduces some false alarms but may miss fake stories; it is not factual verification.")
# ============================================================

if "last_prediction" in st.session_state:
    prediction = st.session_state.last_prediction
    probabilities = st.session_state.last_probabilities
    feature_count = st.session_state.last_feature_count

    st.markdown("---")

    st.markdown(
        """
        <div style="text-align: center; margin-top: 15px;">
            <div class="sec-tag">03 / Output Verdict</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Assuming class 0 = REAL and class 1 = FAKE
    if prediction == 0:
        st.markdown(
            """
            <div class="verdict-banner real">
                <div class="verdict-flex">
                    <div class="verdict-icon-frame">✓</div>
                    <div>
                        <div class="verdict-tagline">✦ TruthLens Verification Result</div>
                        <div class="verdict-title-text">Real News</div>
                        <div class="verdict-description">
                            The model score did not cross the conservative threshold for flagging FAKE. This is a model prediction, not independent proof that every claim is accurate.
                        </div>
                    </div>
                </div>
                <div class="verdict-footer-msg">
                    <span>🟢 Model classification: REAL-like text pattern. Cross-check important claims with reliable independent sources.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif prediction == 1:
        st.markdown(
            """
            <div class="verdict-banner fake">
                <div class="verdict-flex">
                    <div class="verdict-icon-frame">!</div>
                    <div>
                        <div class="verdict-tagline">✦ TruthLens Verification Result</div>
                        <div class="verdict-title-text">Fake News</div>
                        <div class="verdict-description">
                            The model assigned a high FAKE-class score that crossed the conservative flagging threshold. This can still be wrong; verify important claims independently.
                        </div>
                    </div>
                </div>
                <div class="verdict-footer-msg">
                    <span>🔴 Model classification: FAKE-like text pattern. This is not a factual verdict; verify the claims using primary and independent sources.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(f"Unexpected class label returned: {prediction}. Check classification mapping.")

    # ========================================================
    # 11. METRICS DASHBOARD ROW
    # ========================================================

    word_count = len(article_text.split())
    character_count = len(article_text)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Word Count", f"{word_count:,}")
    with m2:
        st.metric("Character Count", f"{character_count:,}")
    with m3:
        st.metric("Active Features", f"{feature_count:,}")
    with m4:
        if probabilities is not None:
            pred_prob = probabilities.get(prediction, 0.0)
            st.metric("Model Confidence", f"{pred_prob * 100:.1f}%")
        else:
            st.metric("Predicted Class", str(prediction))

    st.write("")

    # ========================================================
    # 12. PROBABILITY DISTRIBUTION CHART & BREAKDOWN
    # ========================================================

    chart_left, chart_right = st.columns([1.2, 0.8], gap="large")

    with chart_left:
        st.markdown(
            """
            <div class="sec-heading">Probability Distribution</div>
            <div class="sec-subtext">Model scores for each training label; these are not probabilities that the story is factually true or false.</div>
            """,
            unsafe_allow_html=True
        )

        if probabilities is not None:
            real_score = probabilities.get(0, 0.0) * 100
            fake_score = probabilities.get(1, 0.0) * 100

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=[real_score],
                    y=["REAL NEWS"],
                    orientation="h",
                    name="Real News",
                    marker=dict(
                        color="#10b981",
                        line=dict(color="#34d399", width=2)
                    ),
                    text=[f"{real_score:.1f}%"],
                    textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(color="#022c22", size=14, family="Space Grotesk"),
                    hovertemplate="Real News Score: %{x:.2f}%<extra></extra>"
                )
            )

            fig.add_trace(
                go.Bar(
                    x=[fake_score],
                    y=["FAKE NEWS"],
                    orientation="h",
                    name="Fake News",
                    marker=dict(
                        color="#f43f5e",
                        line=dict(color="#fb7185", width=2)
                    ),
                    text=[f"{fake_score:.1f}%"],
                    textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(color="#4c0519", size=14, family="Space Grotesk"),
                    hovertemplate="Fake News Score: %{x:.2f}%<extra></extra>"
                )
            )

            fig.update_layout(
                barmode="group",
                height=230,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans, sans-serif", color="#cbd5e1"),
                xaxis=dict(
                    range=[0, 100],
                    ticksuffix="%",
                    showgrid=True,
                    gridcolor="rgba(148,163,184,0.1)",
                    zeroline=False
                ),
                yaxis=dict(showgrid=False, zeroline=False),
                showlegend=False,
                bargap=0.4
            )

            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Probability estimation is not enabled for this classifier model.")

    with chart_right:
        st.markdown(
            """
            <div class="sec-heading">Metric Breakdown</div>
            <div class="sec-subtext">Understanding confidence indicators.</div>
            <br>
            <div class="info-box-glass">
                <div class="info-box-title"><span style="color:#10b981;">●</span> Real News Index</div>
                <div class="info-box-text">Score for the class labelled REAL in the training dataset; not proof of factual accuracy.</div>
                <br>
                <div class="info-box-title"><span style="color:#f43f5e;">●</span> Fake News Index</div>
                <div class="info-box-text">Score for the class labelled FAKE in the training dataset; may include false positives.</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 13. LIVE EVIDENCE SEARCH (SEPARATE FROM MODEL PREDICTION)
# ============================================================

if "online_results" in st.session_state:
    results = st.session_state.online_results
    news_result = results.get("news", {})
    fact_result = results.get("fact_checks", {})

    st.markdown("---")
    st.markdown('<div class="sec-tag">04 / Live Evidence Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-heading">Online reports and existing fact-checks</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-subtext">These search results provide context, not automatic proof that the submitted article is true or false.</div>',
        unsafe_allow_html=True,
    )

    if news_result.get("status") == "not_configured":
        st.info(
            "Live news search is not configured. Add NEWS_API_KEY to the .env file "
            "in the same folder as truthlens_app.py, then restart Streamlit."
        )
        st.code("NEWS_API_KEY=your_newsapi_key_here", language="text")
        st.caption("Keep .env private and never commit it to a public GitHub repository.")

    news_col, fact_col = st.columns(2, gap="large")
    with news_col:
        st.markdown("#### 📰 Related news")
        if news_result.get("status") == "ok":
            if news_result.get("items"):
                for item in news_result["items"]:
                    st.markdown(f"**{item['title']}**")
                    st.caption(f"{item['source']} · {item['publishedAt'] or 'Date unavailable'}")
                    if item.get("description"):
                        st.write(item["description"])
                    if item.get("url"):
                        st.markdown(f"[Open original report]({item['url']})")
                    st.markdown("---")
            else:
                st.info("No related articles were returned for this search query. This does not mean the story is false.")
        elif news_result.get("status") != "not_configured":
            st.warning(f"News search could not be completed: {news_result.get('message', 'Unknown error')}")

    with fact_col:
        st.markdown("#### 🧾 Existing fact-checks")
        if fact_result.get("status") == "ok":
            if fact_result.get("items"):
                for item in fact_result["items"]:
                    st.markdown(f"**{item['title']}**")
                    st.caption(f"Publisher: {item['publisher']} · Rating: {item['rating']}")
                    st.write(f"Claim reviewed: {item['claim']}")
                    if item.get("date"):
                        st.caption(f"Review date: {item['date']}")
                    if item.get("url"):
                        st.markdown(f"[Read fact-check]({item['url']})")
                    st.markdown("---")
            else:
                st.info("No matching fact-checks were found. The claim remains unverified by this search.")
        elif fact_result.get("status") == "not_configured":
            st.info("Optional Google Fact Check search is not configured. Related news search can still work without it.")
        else:
            st.warning(f"Fact-check search could not be completed: {fact_result.get('message', 'Unknown error')}")

    st.warning(
        "Evidence status: UNVERIFIED BY AUTOMATION. Review the source, publication date, claim wording, and independent "
        "corroboration yourself. Related coverage alone is not enough to mark a claim as supported or refuted."
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.markdown(
    """
    <div style="text-align: center; border-top: 1px solid rgba(148,163,184,0.15); padding-top: 35px; margin-top: 60px; color: #64748b; font-size: 12px;">
        <span style="color: #38bdf8; font-family: 'Space Grotesk', sans-serif; font-size: 18px; font-weight: 800; letter-spacing: -0.5px;">TruthLens: Intelligent News Verification System</span><br><br>
        Powered by Natural Language Processing & Linear Support Vector Machine Classifiers
    </div>
    """,
    unsafe_allow_html=True
)