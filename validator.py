"""
validator.py — Validation pipeline for each candidate.

Stages (in order):
  1. Authenticity  — is it on PyPI? Populates version, license, URLs.
  2. Safety        — OSV CVE check + AST scan of __init__.py
  3. Health        — download counts, release age
  4. Exclusion     — hard exclusion for PyPI-absent or CVE-flagged packages
  5. Fit scoring   — batch LLM scoring (one call for all non-excluded candidates)

Stages 1–4 run in parallel across all candidates using ThreadPoolExecutor.
Stage 5 is a single batched LLM call, issued after all parallel work completes.
"""

import ast
import re
import json
import datetime
import urllib.request
import urllib.error
import concurrent.futures
from typing import List, Optional

from config import (
    HTTP_TIMEOUT, DAYS_SINCE_RELEASE_WARN,
    MIN_MONTHLY_DOWNLOADS, SUSPICIOUS_AST_PATTERNS, MAX_VALIDATION_WORKERS,
)
from models import Candidate, RunContext, SearchResult
from scrapers import fetch_pypi_details, _get_json


# ── Build Candidate from SearchResult ────────────────────────────────────────

def build_candidate(sr: SearchResult) -> Candidate:
    """Convert a merged SearchResult into a Candidate for validation."""
    sources = [s.strip() for s in sr.source.split(",")]
    c = Candidate(name=sr.name, sources=sources, description=sr.description)
    if sr.url:
        if "pypi.org" in sr.url:
            c.pypi_url = sr.url
        elif "github.com" in sr.url:
            c.github_url = sr.url

    # Extract embedded github_stars tag written by scrape_github/scrape_github_topics
    stars_match = re.search(r'\[github_stars:(\d+)\]', sr.description)
    if stars_match:
        c.github_stars = int(stars_match.group(1))
        # Strip the tag from the visible description
        c.description = re.sub(r'\s*\[github_stars:\d+\]', '', c.description).strip()

    return c


# ── Stage 1: Authenticity ─────────────────────────────────────────────────────

def check_authenticity(c: Candidate) -> None:
    """
    Verifies the package exists on PyPI and populates basic metadata.
    Sets c.pypi_verified and populates version, urls, license.
    """
    data = fetch_pypi_details(c.name)
    if not data:
        c.pypi_verified = False
        c.safety_notes.append("Not found on PyPI")
        return

    c.pypi_verified = True
    info = data.get("info", {})

    c.latest_version = info.get("version", "")
    c.license = info.get("license", "") or ""
    c.pypi_url = info.get("package_url", f"https://pypi.org/project/{c.name}")
    c.docs_url = info.get("docs_url", "") or info.get("home_page", "") or ""

    classifiers = info.get("classifiers", [])
    c.python_versions = [
        cl.split(" :: ")[-1]
        for cl in classifiers
        if "Programming Language :: Python :: 3" in cl
        and cl.count("::") == 2
    ]

    project_urls = info.get("project_urls") or {}
    for key, url in project_urls.items():
        if "github.com" in (url or ""):
            c.github_url = url
            break

    pypi_summary = info.get("summary", "")
    if pypi_summary and len(pypi_summary) > len(c.description):
        c.description = pypi_summary

    releases = data.get("releases", {})
    if c.latest_version and c.latest_version in releases:
        release_files = releases[c.latest_version]
        if release_files:
            upload_time = release_files[-1].get("upload_time", "")
            if upload_time:
                try:
                    dt = datetime.datetime.fromisoformat(upload_time)
                    c.days_since_release = (datetime.datetime.utcnow() - dt).days
                except ValueError:
                    pass


# ── Stage 2: Safety ───────────────────────────────────────────────────────────

def check_safety(c: Candidate) -> None:
    """
    Two checks:
      a) OSV vulnerability database — any known CVEs for this package
      b) AST scan of PyPI-published source (top-level __init__.py or setup.py)
    """
    _check_osv(c)
    _check_ast(c)

    critical_flags = [
        n for n in c.safety_notes
        if "CVE" in n or "malicious" in n.lower() or "suspicious" in n.lower()
    ]
    c.safety_passed = len(critical_flags) == 0


def _check_osv(c: Candidate) -> None:
    """Query OSV.dev for known vulnerabilities."""
    url = "https://api.osv.dev/v1/query"
    payload = json.dumps({
        "package": {"name": c.name, "ecosystem": "PyPI"}
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
            vulns = data.get("vulns", [])
            if vulns:
                ids = [v.get("id", "?") for v in vulns[:3]]
                c.safety_notes.append(f"⚠ Known CVEs: {', '.join(ids)}")
                c.cve_clean = False
            else:
                c.cve_clean = True
    except Exception:
        c.safety_notes.append("CVE check skipped (OSV unreachable)")
        c.cve_clean = None


def _check_ast(c: Candidate) -> None:
    """
    Download the source distribution from PyPI and AST-scan __init__.py
    for suspicious patterns.
    """
    if not c.pypi_verified:
        return

    data = fetch_pypi_details(c.name)
    if not data:
        return

    urls = data.get("urls", [])
    source_url = None
    for u in urls:
        if u.get("packagetype") == "sdist" and u.get("url", "").endswith(".tar.gz"):
            source_url = u["url"]
            break

    if not source_url:
        return

    try:
        import io
        import tarfile

        req = urllib.request.Request(source_url)
        req.add_header("User-Agent", "pylib-finder/0.1")
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()

        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
            init_source = None
            for member in tar.getmembers():
                if member.name.endswith("__init__.py") and member.name.count("/") <= 2:
                    f = tar.extractfile(member)
                    if f:
                        init_source = f.read().decode("utf-8", errors="replace")
                        break

            if init_source is None:
                return

            try:
                tree = ast.parse(init_source)
            except SyntaxError:
                c.safety_notes.append("⚠ AST parse failed (possibly obfuscated)")
                return

            flags = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func_name = ""
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr
                    if func_name in SUSPICIOUS_AST_PATTERNS:
                        flags.append(func_name)

                if isinstance(node, ast.Assign):
                    for val in ast.walk(node):
                        if isinstance(val, ast.Constant) and isinstance(val.value, str):
                            if len(val.value) > 100 and re.match(r'^[A-Za-z0-9+/=]+$', val.value):
                                flags.append("possible_base64_obfuscation")

            if flags:
                unique_flags = list(dict.fromkeys(flags))
                c.safety_notes.append(
                    f"⚠ Suspicious AST patterns: {', '.join(unique_flags)}"
                )

    except Exception:
        pass


# ── Stage 3: Health signals ───────────────────────────────────────────────────

def check_health(c: Candidate) -> None:
    """Check download stats from PyPI Stats API and flag stale packages."""
    if not c.pypi_verified:
        return

    data = _get_json(f"https://pypistats.org/api/packages/{c.name}/recent")
    if data and "data" in data:
        c.monthly_downloads = data["data"].get("last_month")

    if c.days_since_release is not None and c.days_since_release > DAYS_SINCE_RELEASE_WARN:
        c.safety_notes.append(
            f"⚠ Not updated in {c.days_since_release // 365}+ year(s)"
        )
    if c.monthly_downloads is not None and c.monthly_downloads < MIN_MONTHLY_DOWNLOADS:
        c.safety_notes.append(
            f"⚠ Low download traction ({c.monthly_downloads:,}/month)"
        )


# ── Stage 4: Exclusion decision ───────────────────────────────────────────────

def decide_exclusion(c: Candidate) -> None:
    """
    Exclude candidates that fail hard safety/authenticity requirements.
    Soft warnings (stale, low downloads) are noted but don't exclude.
    """
    if not c.pypi_verified:
        c.excluded = True
        c.exclusion_reason = "Not found on PyPI"
        return

    if c.cve_clean is False:
        c.excluded = True
        c.exclusion_reason = "Known CVEs detected"
        return

    critical = [
        n for n in c.safety_notes
        if "malicious" in n.lower() or "obfuscat" in n.lower()
    ]
    if critical:
        c.excluded = True
        c.exclusion_reason = "; ".join(critical)


# ── Single-candidate pipeline (for parallel execution) ───────────────────────

def _validate_one(
    name: str,
    sr: SearchResult,
    progress_callback,
) -> Candidate:
    """Run stages 1–4 for a single candidate. Safe to call from a thread."""
    c = build_candidate(sr)

    if progress_callback:
        progress_callback(name, "authenticity")
    check_authenticity(c)

    if progress_callback:
        progress_callback(name, "safety")
    check_safety(c)
    check_health(c)
    decide_exclusion(c)

    return c


# ── Full pipeline ─────────────────────────────────────────────────────────────

def validate_candidates(
    search_results: dict,
    ctx: RunContext,
    progress_callback=None,
) -> List[Candidate]:
    """
    Run all validation stages for every candidate.

    Stages 1–4 (all network-bound) run in parallel via ThreadPoolExecutor.
    Stage 5 (fit scoring) runs as a single batched LLM call after all parallel
    work completes, falling back to per-candidate calls if the batch fails.

    search_results: {name: SearchResult} from scrapers.merge_results()
    ctx: RunContext with intent for fit scoring
    progress_callback: optional fn(name, stage) for CLI progress display
    """
    from llm import score_fit_batch, score_fit  # avoid circular import at module level

    # ── Stages 1–4: parallel network validation ───────────────────────────────
    candidates: List[Candidate] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_VALIDATION_WORKERS) as executor:
        future_to_name = {
            executor.submit(_validate_one, name, sr, progress_callback): name
            for name, sr in search_results.items()
        }
        for future in concurrent.futures.as_completed(future_to_name):
            try:
                candidates.append(future.result())
            except Exception as e:
                name = future_to_name[future]
                # Create a stub excluded candidate so the run doesn't silently lose it
                stub = Candidate(name=name)
                stub.excluded = True
                stub.exclusion_reason = f"Validation error: {e}"
                candidates.append(stub)

    # ── Stage 5: batch fit scoring ────────────────────────────────────────────
    non_excluded = [c for c in candidates if not c.excluded]

    if non_excluded:
        if progress_callback:
            progress_callback(f"{len(non_excluded)} candidates", "batch fit scoring")
        try:
            score_fit_batch(non_excluded, ctx)
        except Exception as batch_err:
            # Fall back to individual scoring
            if progress_callback:
                progress_callback("batch failed", f"falling back to per-candidate scoring ({batch_err})")
            for c in non_excluded:
                try:
                    score_fit(c, ctx)
                except Exception as e:
                    c.fit_notes = f"Scoring failed: {e}"
                    c.fit_score = 0.0

    candidates.sort(key=lambda c: (c.excluded, -c.fit_score))
    return candidates
