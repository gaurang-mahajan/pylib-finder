"""
scrapers.py — One function per source, all return List[SearchResult].

Each scraper is intentionally fault-tolerant:
  - Never raises — returns empty list on any error
  - Logs nothing directly; the caller handles errors

Sources
-------
  scrape_pypi            — PyPI search HTML (classifier-filtered)
  scrape_package_hints   — Direct PyPI JSON lookup for LLM-suggested name hints
  scrape_github          — GitHub repo search, stars-sorted, with PyPI name extraction
  scrape_github_topics   — GitHub topic-tagged repos (curated community signal)
  scrape_stackoverflow   — StackExchange API, question titles + answer bodies
  scrape_reddit          — Reddit JSON search across multiple Python subreddits
  scrape_web             — DuckDuckGo HTML, pip-install/import pattern extraction
  scrape_papers_with_code — Papers With Code API, ML/AI paper implementations
"""

import re
import time
import urllib.parse
import urllib.request
import urllib.error
import json
from typing import List, Optional

from config import HTTP_TIMEOUT, MAX_CANDIDATES_PER_SOURCE, GITHUB_TOKEN
from models import SearchResult


# ── Shared HTTP helper ────────────────────────────────────────────────────────

def _get(url: str, headers: dict = None) -> str:
    """Simple GET, returns response text or '' on failure."""
    req = urllib.request.Request(url, headers=headers or {})
    req.add_header(
        "User-Agent",
        "pylib-finder/0.1 (research tool; contact via GitHub)"
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _get_json(url: str, headers: dict = None):
    text = _get(url, headers)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _github_headers() -> dict:
    """Build GitHub API headers, including auth token when available."""
    h = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"token {GITHUB_TOKEN}"
    return h


# ── Extract PyPI-friendly package name from a GitHub repo item ────────────────

def _extract_pypi_name_from_repo(item: dict) -> Optional[str]:
    """
    Tries to derive a PyPI package name from a GitHub repo API item.

    Priority order:
      1. Explicit `pip install <name>` in the repo description
      2. Explicit `import <name>` in the repo description
      3. Repo name normalised to underscores (fallback)
    """
    description = (item.get("description") or "").lower()

    pip_match = re.search(r'pip install ([a-z][a-z0-9_-]{2,40})', description)
    if pip_match:
        return pip_match.group(1).replace("-", "_")

    import_match = re.search(r'\bimport ([a-z][a-z0-9_]{2,40})\b', description)
    if import_match:
        return import_match.group(1)

    return item.get("name", "").lower().replace("-", "_").replace(" ", "_") or None


# ── 1. PyPI search ────────────────────────────────────────────────────────────

def scrape_pypi(search_terms: List[str]) -> List[SearchResult]:
    """
    PyPI search HTML endpoint — same as pip search.
    Queries up to 3 terms without a classifier filter so non-science libs
    are also discovered.
    """
    results: List[SearchResult] = []
    seen: set = set()

    for term in search_terms[:3]:
        encoded = urllib.parse.quote_plus(term)
        url = f"https://pypi.org/search/?q={encoded}&o=-zscore"

        html = _get(url)
        if not html:
            continue

        names = re.findall(
            r'class="package-snippet__name">\s*([^<]+)\s*</span>', html
        )
        descriptions = re.findall(
            r'class="package-snippet__description">\s*([^<]+)\s*</span>', html
        )

        for i, name in enumerate(names[:MAX_CANDIDATES_PER_SOURCE]):
            name = name.strip().lower()
            if name and name not in seen:
                seen.add(name)
                desc = descriptions[i].strip() if i < len(descriptions) else ""
                results.append(SearchResult(
                    name=name,
                    source="pypi",
                    description=desc,
                    url=f"https://pypi.org/project/{name}",
                ))

        time.sleep(0.5)

    return results


# ── 2. PyPI JSON API — fetch details for a known package name ─────────────────

def fetch_pypi_details(name: str) -> dict:
    """
    Returns raw PyPI JSON for a specific package, or {} on failure.
    Used by the validator, not the discovery phase.
    """
    data = _get_json(f"https://pypi.org/pypi/{name}/json")
    return data or {}


# ── 3. Package hints — direct PyPI lookup for LLM-suggested name fragments ───

def scrape_package_hints(hints: List[str]) -> List[SearchResult]:
    """
    Direct PyPI JSON API lookup for each name hint produced by query expansion.
    These are high-confidence candidates — the LLM already believes they exist.
    Packages that don't exist on PyPI are simply skipped (validator confirms later).
    """
    results: List[SearchResult] = []

    for hint in hints:
        name = hint.strip().lower().replace("-", "_")
        if not name or len(name) < 2:
            continue

        data = _get_json(f"https://pypi.org/pypi/{name}/json")
        if data and "info" in data:
            info = data["info"]
            results.append(SearchResult(
                name=name,
                source="hints",
                description=info.get("summary", ""),
                url=info.get("package_url", f"https://pypi.org/project/{name}"),
            ))

        time.sleep(0.3)

    return results


# ── 4. GitHub repo search ─────────────────────────────────────────────────────

def scrape_github(search_terms: List[str]) -> List[SearchResult]:
    """
    GitHub repository search sorted by stars.
    Extracts PyPI-friendly package name from repo description before
    falling back to the repo name itself.
    Includes star count in a structured side-channel via the description
    (picked up by the validator to populate github_stars).
    """
    results: List[SearchResult] = []
    seen: set = set()
    headers = _github_headers()

    for term in search_terms[:2]:
        query = urllib.parse.quote_plus(f"{term} language:python")
        url = (
            f"https://api.github.com/search/repositories"
            f"?q={query}&sort=stars&per_page={MAX_CANDIDATES_PER_SOURCE}"
        )

        data = _get_json(url, headers=headers)
        if not data or "items" not in data:
            continue

        for item in data["items"][:MAX_CANDIDATES_PER_SOURCE]:
            name = _extract_pypi_name_from_repo(item)
            if not name or name in seen:
                continue
            seen.add(name)

            stars = item.get("stargazers_count", 0)
            desc = item.get("description", "") or ""
            # Embed stars in a parseable tag so the validator can extract it
            # without changing the SearchResult schema.
            desc_with_stars = f"{desc} [github_stars:{stars}]".strip()

            results.append(SearchResult(
                name=name,
                source="github",
                description=desc_with_stars,
                url=item.get("html_url", ""),
            ))

        time.sleep(1.0)

    return results


# ── 5. GitHub topics search ───────────────────────────────────────────────────

def scrape_github_topics(search_terms: List[str]) -> List[SearchResult]:
    """
    GitHub repository search restricted to repos whose topic tags match the
    search term slug.  Topic-tagged repos are typically more purpose-built and
    discoverable than repos found by plain keyword matching.
    """
    results: List[SearchResult] = []
    seen: set = set()
    headers = _github_headers()

    for term in search_terms[:2]:
        topic_slug = re.sub(r'[^a-z0-9]+', '-', term.lower()).strip('-')
        url = (
            f"https://api.github.com/search/repositories"
            f"?q=topic:{topic_slug}+language:python"
            f"&sort=stars&per_page={MAX_CANDIDATES_PER_SOURCE}"
        )

        data = _get_json(url, headers=headers)
        if not data or "items" not in data:
            continue

        for item in data["items"][:MAX_CANDIDATES_PER_SOURCE]:
            name = _extract_pypi_name_from_repo(item)
            if not name or name in seen:
                continue
            seen.add(name)

            stars = item.get("stargazers_count", 0)
            desc = item.get("description", "") or ""
            desc_with_stars = f"{desc} [github_stars:{stars}]".strip()

            results.append(SearchResult(
                name=name,
                source="github_topics",
                description=desc_with_stars,
                url=item.get("html_url", ""),
            ))

        time.sleep(1.0)

    return results


# ── 6. Stack Overflow search ──────────────────────────────────────────────────

def scrape_stackoverflow(search_terms: List[str]) -> List[SearchResult]:
    """
    StackExchange API — searches questions tagged python for relevant terms.

    Two tiers of signal:
      1. Package names extracted from question titles (fast, broad).
      2. Package names extracted from the body of the top accepted/voted answer
         for each question (slower, but much higher precision).
    """
    results: List[SearchResult] = []
    seen: set = set()

    for term in search_terms[:2]:
        encoded = urllib.parse.quote_plus(term)

        # Fetch questions with bodies included
        url = (
            f"https://api.stackexchange.com/2.3/search/advanced"
            f"?q={encoded}&tagged=python&site=stackoverflow"
            f"&sort=relevance&pagesize=10&filter=withbody"
        )

        data = _get_json(url)
        if not data or "items" not in data:
            continue

        question_ids = []
        stopwords = {
            "the", "for", "with", "using", "how", "find", "get", "python",
            "data", "from", "and", "not", "that", "this", "can", "what",
            "when", "where", "why", "best", "way", "list", "does", "use",
            "code", "error", "output", "input", "value", "type", "need",
        }

        for item in data["items"][:MAX_CANDIDATES_PER_SOURCE]:
            title = item.get("title", "")
            body = item.get("body", "")

            combined = (title + " " + body[:600]).lower()
            combined_clean = re.sub(r'<[^>]+>', ' ', combined)

            # High-signal: explicit pip install or import mentions
            explicit = re.findall(
                r'(?:pip install|import)\s+([a-z][a-z0-9_-]{2,30})',
                combined_clean,
            )
            # Fallback: word tokens that look like package names
            word_tokens = re.findall(r'\b([a-z][a-z0-9_-]{2,30})\b', title.lower())

            for pkg in explicit + word_tokens:
                pkg = pkg.replace("-", "_")
                if pkg not in stopwords and pkg not in seen:
                    seen.add(pkg)
                    results.append(SearchResult(
                        name=pkg,
                        source="stackoverflow",
                        description=title,
                        url=item.get("link", ""),
                    ))

            if item.get("question_id"):
                question_ids.append(str(item["question_id"]))

        # Fetch top answers for first 5 questions (second-tier signal)
        if question_ids:
            ids_str = ";".join(question_ids[:5])
            ans_url = (
                f"https://api.stackexchange.com/2.3/questions/{ids_str}/answers"
                f"?site=stackoverflow&sort=votes&pagesize=3&filter=withbody"
            )
            ans_data = _get_json(ans_url)
            if ans_data and "items" in ans_data:
                for ans in ans_data["items"]:
                    body = re.sub(r'<[^>]+>', ' ', ans.get("body", "")).lower()
                    for pkg in re.findall(
                        r'(?:pip install|import)\s+([a-z][a-z0-9_-]{2,30})',
                        body,
                    ):
                        pkg = pkg.replace("-", "_")
                        if pkg not in stopwords and pkg not in seen:
                            seen.add(pkg)
                            results.append(SearchResult(
                                name=pkg,
                                source="stackoverflow",
                                description=f"Mentioned in answer for: {term}",
                                url="",
                            ))

        time.sleep(0.5)

    return results


# ── 7. Reddit search ──────────────────────────────────────────────────────────

def scrape_reddit(search_terms: List[str]) -> List[SearchResult]:
    """
    Reddit JSON search across Python-related subreddits.
    Extracts package names from post titles and bodies using explicit
    import/pip-install patterns plus inline code blocks.
    """
    results: List[SearchResult] = []
    seen: set = set()
    subreddits = [
        "datascience", "learnpython", "MachineLearning", "Python",
        "LanguageTechnology", "artificial", "bioinformatics", "genomics",
        "statistics"
    ]

    for term in search_terms[:2]:
        for sub in subreddits[:4]:
            encoded = urllib.parse.quote_plus(term)
            url = (
                f"https://www.reddit.com/r/{sub}/search.json"
                f"?q={encoded}&restrict_sr=1&sort=relevance&limit=10"
            )
            data = _get_json(url, headers={"Accept": "application/json"})
            if not data:
                continue

            posts = data.get("data", {}).get("children", [])
            for post in posts[:MAX_CANDIDATES_PER_SOURCE]:
                post_data = post.get("data", {})
                title = post_data.get("title", "")
                selftext = post_data.get("selftext", "")

                combined = (title + " " + selftext[:800]).lower()

                # Backtick-wrapped names, pip install, import patterns
                pkg_candidates = re.findall(r'`([a-z][a-z0-9_-]{2,30})`', combined)
                pkg_candidates += re.findall(
                    r'(?:import|pip install|pip3 install|library|package)\s+'
                    r'([a-z][a-z0-9_-]{2,30})',
                    combined,
                )

                for pkg in pkg_candidates:
                    pkg = pkg.replace("-", "_")
                    if pkg not in seen:
                        seen.add(pkg)
                        results.append(SearchResult(
                            name=pkg,
                            source="reddit",
                            description=title,
                            url=f"https://reddit.com{post_data.get('permalink', '')}",
                        ))

            time.sleep(1.0)

    return results


# ── 8. Web search (via DuckDuckGo HTML) ───────────────────────────────────────

def scrape_web(search_terms: List[str]) -> List[SearchResult]:
    """
    DuckDuckGo HTML search — no API key needed.
    Extracts package-looking names from result snippets.
    """
    results: List[SearchResult] = []
    seen: set = set()

    for term in search_terms[:2]:
        query = urllib.parse.quote_plus(f"python library {term}")
        url = f"https://html.duckduckgo.com/html/?q={query}"

        html = _get(url, headers={"Accept-Language": "en-US"})
        if not html:
            continue

        snippets = re.findall(r'class="result__snippet">(.*?)</a>', html, re.DOTALL)
        titles = re.findall(r'class="result__title".*?>(.*?)</a>', html, re.DOTALL)

        combined_text = " ".join(snippets + titles)
        combined_text = re.sub(r'<[^>]+>', ' ', combined_text).lower()

        pip_pkgs = re.findall(r'pip install ([a-z][a-z0-9_-]{2,30})', combined_text)
        import_pkgs = re.findall(r'import ([a-z][a-z0-9_]{2,30})', combined_text)

        for pkg in pip_pkgs + import_pkgs:
            pkg = pkg.replace("-", "_")
            if pkg not in seen:
                seen.add(pkg)
                results.append(SearchResult(
                    name=pkg,
                    source="web",
                    description=f"Found in web search for: {term}",
                    url="",
                ))

        time.sleep(1.0)

    return results


# ── 9. Papers With Code ───────────────────────────────────────────────────────

def scrape_papers_with_code(search_terms: List[str]) -> List[SearchResult]:
    """
    Papers With Code API — finds academic ML/AI papers that have public code.
    Extracts GitHub repository names as candidate package names, which are then
    verified against PyPI in the validation stage.

    Particularly strong for cutting-edge ML/scientific computing queries where
    the best tool may be a research repo rather than a polished PyPI package.
    """
    results: List[SearchResult] = []
    seen: set = set()

    for term in search_terms[:2]:
        encoded = urllib.parse.quote_plus(term)
        url = (
            f"https://paperswithcode.com/api/v1/papers/"
            f"?q={encoded}&page=1&items_per_page={MAX_CANDIDATES_PER_SOURCE}"
        )

        data = _get_json(url)
        if not data or "results" not in data:
            continue

        for paper in data["results"][:MAX_CANDIDATES_PER_SOURCE]:
            repos = paper.get("repositories", [])
            # Sort by stars descending so we pick the most popular implementation
            repos_sorted = sorted(repos, key=lambda r: r.get("stars", 0), reverse=True)

            for repo in repos_sorted[:2]:
                repo_url = repo.get("url", "")
                if "github.com" not in repo_url:
                    continue

                parts = repo_url.rstrip("/").split("/")
                if len(parts) < 2:
                    continue

                pkg_name = parts[-1].lower().replace("-", "_")
                if not pkg_name or len(pkg_name) < 2 or pkg_name in seen:
                    continue

                seen.add(pkg_name)
                stars = repo.get("stars", 0)
                title = paper.get("title", "")
                results.append(SearchResult(
                    name=pkg_name,
                    source="paperswithcode",
                    description=f"Implementation of: {title} [github_stars:{stars}]",
                    url=repo_url,
                ))

        time.sleep(0.5)

    return results


# ── Merge + deduplicate all search results ────────────────────────────────────

def merge_results(all_results: List[SearchResult]) -> dict:
    """
    Merges results by package name.
    Returns dict: {name: SearchResult} with combined sources.
    The description from the highest-priority source is kept.

    Source priority: hints > llm > pypi > github > github_topics >
                     paperswithcode > stackoverflow > reddit > web
    """
    priority = {
        "hints": 0,
        "llm": 1,
        "pypi": 2,
        "github": 3,
        "github_topics": 4,
        "paperswithcode": 5,
        "stackoverflow": 6,
        "reddit": 7,
        "web": 8,
    }

    merged: dict = {}

    for r in all_results:
        name = r.name.strip().lower().replace("-", "_")
        if not name or len(name) < 2:
            continue

        if name not in merged:
            merged[name] = {"result": r, "sources": [r.source]}
        else:
            merged[name]["sources"].append(r.source)
            existing_priority = priority.get(merged[name]["result"].source, 99)
            new_priority = priority.get(r.source, 99)
            if new_priority < existing_priority:
                merged[name]["result"] = r

    output = {}
    for name, data in merged.items():
        r = data["result"]
        r.source = ", ".join(sorted(set(data["sources"]), key=lambda s: priority.get(s, 99)))
        output[name] = r

    return output
