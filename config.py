"""
config.py — Central configuration for pylib_finder.
Edit these values to tune behaviour without touching core logic.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env if present; env vars already set in the shell take priority
except ImportError:
    pass  # python-dotenv not installed — fall back to shell environment variables

# ── LLM ──────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LLM_MODEL = "claude-opus-4-6"          # model used for expansion + scoring
LLM_MAX_TOKENS = 1024

# ── Search ────────────────────────────────────────────────────────────────────
MAX_CANDIDATES_PER_SOURCE = 8         # how many hits to pull per scraper
MAX_TOTAL_CANDIDATES = 32              # hard cap before validation

# ── Validation thresholds ─────────────────────────────────────────────────────
MIN_FIT_SCORE = 3.0                    # below this → excluded from display
DAYS_SINCE_RELEASE_WARN = 730          # warn if package not updated in 2 years
MIN_MONTHLY_DOWNLOADS = 50            # below this → flagged as low-traction

# ── Parallel validation ───────────────────────────────────────────────────────
MAX_VALIDATION_WORKERS = 8             # ThreadPoolExecutor workers for Stage 4

# ── Output ────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "pylib_results")

# ── Scraper timeouts (seconds) ────────────────────────────────────────────────
HTTP_TIMEOUT = 10

# ── GitHub auth (optional — increases API rate limit from 10 to 30 req/min) ───
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

# ── Safety ────────────────────────────────────────────────────────────────────
# AST patterns that trigger a safety warning in package source
SUSPICIOUS_AST_PATTERNS = [
    "exec", "eval", "compile",
    "__import__",
    "subprocess", "os.system",
    "socket", "urllib.request", "requests.get",  # network in __init__
]
