"""
config.py — Central configuration for pylib_finder.
Edit these values to tune behavior without touching core logic.
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
LLM_MAX_TOKENS = 2048

# ── Search ────────────────────────────────────────────────────────────────────
MAX_CANDIDATES_PER_SOURCE = 4         # how many hits to pull per scraper
MAX_TOTAL_CANDIDATES = 16              # hard cap before validation

# ── Validation thresholds ─────────────────────────────────────────────────────
MIN_FIT_SCORE = 3.0                    # below this → excluded from display
DAYS_SINCE_RELEASE_WARN = 730          # warn if package not updated in 2 years
MIN_MONTHLY_DOWNLOADS = 50             # below this → flagged as low-traction
DAYS_NEW_PACKAGE_WARN = 90             # warn if package first released within this many days
PRE_RELEASE_WARN = True                # show info note for version 0.x packages

# ── Parallel validation ───────────────────────────────────────────────────────
MAX_VALIDATION_WORKERS = 8             # ThreadPoolExecutor workers for Stage 4

# ── Output ────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "pylib_results")

# ── Scraper search depth ─────────────────────────────────────────────────────
# Number of expanded search terms sent to each source.
# Raising these finds more candidates but increases wall-clock time linearly.
PYPI_MAX_TERMS        = 3   # PyPI tolerates a few more (no strict rate limit)
SCRAPER_MAX_TERMS     = 2   # GitHub, GitHub Topics, SO, Web, Papers With Code
REDDIT_MAX_TERMS      = 2   # terms sent per subreddit
REDDIT_MAX_SUBREDDITS = 4   # subreddits searched (top N from ordered list)

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
    "getenv",                                     # os.getenv — credential harvesting
    "ctypes", "CDLL", "WinDLL", "cdll",          # native code execution
]

# Credential-related path fragments that should never appear in a library __init__
SUSPICIOUS_PATHS = [
    ".ssh", ".aws/credentials", ".aws/config",
    "etc/passwd", "etc/shadow",
    ".config/gcloud", ".kube/config",
    "AppData/Roaming",
]

# Licenses that warrant a warning for potentially restrictive commercial use.
# These are flagged as ⚠ (informational warning, does not affect safety_passed).
RESTRICTIVE_LICENSES = [
    "GPL", "AGPL", "EUPL", "SSPL", "CPAL", "OSL",
    "Affero", "Reciprocal",
]
