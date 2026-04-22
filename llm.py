"""
llm.py — All LLM interactions:
  1. expand_query()      → structured search terms from raw user input
  2. llm_prior()         → Claude's own candidate suggestions
  3. score_fit_batch()   → fit scoring for all candidates in a single API call
  4. score_fit()         → single-candidate fallback (used if batch fails)
"""

import json
import time
import anthropic

from config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_MAX_TOKENS
from models import RunContext, SearchResult

_client = None

# ── Retry logic ───────────────────────────────────────────────────────────────
_RETRYABLE_STATUS = {529, 500, 502, 503, 504}
_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 2.0   # seconds; doubles each attempt (2 → 4 → 8 → 16)


def _call_with_retry(**kwargs) -> object:
    """
    Wrapper around client.messages.create with exponential backoff.
    Retries on transient server errors (overloaded, 5xx).
    Raises immediately on client errors (4xx except 429) and auth failures.
    """
    client = _get_client()
    delay = _RETRY_BASE_DELAY

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return client.messages.create(**kwargs)
        except anthropic.APIStatusError as e:
            status = e.status_code
            if status == 429 or status in _RETRYABLE_STATUS:
                if attempt == _MAX_RETRIES:
                    raise
                # Honour Retry-After header when present
                retry_after = e.response.headers.get("retry-after")
                wait = float(retry_after) if retry_after else delay
                time.sleep(wait)
                delay *= 2
            else:
                raise  # 400, 401, 403 etc. — not worth retrying

# ── Token usage tracking ──────────────────────────────────────────────────────
_token_usage = {"input": 0, "output": 0, "calls": 0}


def get_token_usage() -> dict:
    """Return a copy of accumulated token usage for this session."""
    return dict(_token_usage)


def reset_token_usage() -> None:
    """Reset counters — call at the start of each run if reusing the module."""
    _token_usage["input"] = 0
    _token_usage["output"] = 0
    _token_usage["calls"] = 0


def _track(response) -> None:
    """Increment counters from an API response object."""
    _token_usage["input"] += response.usage.input_tokens
    _token_usage["output"] += response.usage.output_tokens
    _token_usage["calls"] += 1


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Export it with: export ANTHROPIC_API_KEY=sk-..."
            )
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


# ── 1. Query Expansion ────────────────────────────────────────────────────────

EXPAND_PROMPT = """\
You are a Python ecosystem expert. A user has described a computational task and \
wants to find existing Python libraries for it.

User query: "{query}"

Return a JSON object with exactly these keys:
{{
  "intent": "<one clear sentence describing the technical goal>",
  "search_terms": ["<3-6 short search phrases for web/PyPI/GitHub search>"],
  "package_hints": ["<2-4 likely partial package name fragments, lowercase>"]
}}

Only return the JSON object, no commentary.
"""


def expand_query(raw_query: str) -> RunContext:
    """Call Claude to expand the raw query into structured search inputs."""
    prompt = EXPAND_PROMPT.format(query=raw_query)

    response = _call_with_retry(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    _track(response)

    data = json.loads(_strip_fences(response.content[0].text))

    ctx = RunContext(raw_query=raw_query)
    ctx.intent = data.get("intent", raw_query)
    ctx.search_terms = data.get("search_terms", [raw_query])
    ctx.package_hints = data.get("package_hints", [])
    return ctx


# ── 2. LLM Prior ─────────────────────────────────────────────────────────────

PRIOR_PROMPT = """\
You are a Python ecosystem expert. Based purely on your training knowledge, \
suggest Python libraries that could help with the following task.

Task intent: "{intent}"

Return a JSON array of up to 8 candidates, including both well-known and niche \
libraries. Each item:
{{
  "name": "<exact PyPI package name, lowercase>",
  "description": "<one sentence on what it does and why it fits>",
  "suggested_functions": ["<specific class or function name if known>"],
  "confidence": "<high|medium|low>"
}}

Only return the JSON array, no commentary. Only suggest packages you are \
reasonably confident exist on PyPI.
"""


def llm_prior(ctx: RunContext) -> list[SearchResult]:
    """Ask Claude for its own candidate suggestions based on the intent."""
    prompt = PRIOR_PROMPT.format(intent=ctx.intent)

    response = _call_with_retry(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    _track(response)

    items = json.loads(_strip_fences(response.content[0].text))

    results = []
    for item in items:
        r = SearchResult(
            name=item.get("name", "").strip().lower(),
            source="llm",
            description=item.get("description", ""),
            url=f"https://pypi.org/project/{item.get('name', '')}",
        )
        if item.get("suggested_functions"):
            r.description += (
                " | functions: " + ", ".join(item["suggested_functions"])
            )
        if r.name:
            results.append(r)
    return results


# ── 3. Batch Fit Scoring ──────────────────────────────────────────────────────

BATCH_SCORE_PROMPT = """\
You are evaluating Python libraries to find the best fit for a user's task.

User intent: "{intent}"

Score each library below on a scale of 0.0 to 10.0 for how well it addresses \
the intent. Also provide a one-sentence explanation and suggest specific \
function or class names where applicable.

Each library entry may include these maturity signals — use them to apply light
penalisation (−0.5 to −1.5 points) where appropriate:
  - days_since_last_update: penalise heavily stale packages (>730 days)
  - days_since_first_release: penalise very new packages (<90 days) slightly
  - pre_release: True if version is 0.x — note API instability
  - monthly_downloads: low counts (<50/month) suggest limited real-world use
  - github_stars: very low stars suggest niche or abandoned projects

Libraries to evaluate:
{libraries_json}

Return a JSON array in the SAME ORDER as the input, one object per library:
[
  {{
    "name": "<package name, unchanged>",
    "fit_score": <float 0.0-10.0>,
    "fit_notes": "<one sentence explaining the score and any maturity concerns>",
    "suggested_functions": ["<specific function/class names if applicable>"]
  }},
  ...
]

Only return the JSON array. Include every library, even low-scoring ones.
"""


def score_fit_batch(candidates: list, ctx: RunContext) -> None:
    """
    Score all candidates in a single LLM call. Mutates each candidate in-place.

    Falls back gracefully: if parsing fails or a candidate is missing from the
    response, that candidate keeps fit_score=0.0 and gets a fallback note.
    """
    if not candidates:
        return

    libraries = [
        {
            "name": c.name,
            "description": c.description,
            "pypi_verified": c.pypi_verified,
            "version": c.latest_version or "unknown",
            "days_since_last_update": c.days_since_release,
            "days_since_first_release": c.days_since_first_release,
            "pre_release": c.latest_version.startswith("0.") if c.latest_version else None,
            "monthly_downloads": c.monthly_downloads,
            "github_stars": c.github_stars,
        }
        for c in candidates
    ]

    prompt = BATCH_SCORE_PROMPT.format(
        intent=ctx.intent,
        libraries_json=json.dumps(libraries, indent=2),
    )

    # Allow ~350 output tokens per candidate; cap at model context ceiling
    max_out = min(4096, 350 * len(candidates))

    response = _call_with_retry(
        model=LLM_MODEL,
        max_tokens=max_out,
        messages=[{"role": "user", "content": prompt}],
    )
    _track(response)

    results = json.loads(_strip_fences(response.content[0].text))

    # Map by name for O(1) lookup regardless of ordering
    result_map = {item.get("name", "").lower().replace("-", "_"): item for item in results}

    for c in candidates:
        item = result_map.get(c.name.lower().replace("-", "_"), {})
        if item:
            c.fit_score = float(item.get("fit_score", 0.0))
            c.fit_notes = item.get("fit_notes", "")
            c.suggested_functions = item.get("suggested_functions", [])
        else:
            c.fit_score = 0.0
            c.fit_notes = "Score unavailable (not returned by batch scorer)"


# ── 4. Single Fit Score (fallback) ────────────────────────────────────────────

SCORE_PROMPT = """\
You are evaluating whether a Python library is a good fit for a user's task.

User intent: "{intent}"

Library: {name}
Description: {description}
PyPI verified: {pypi_verified}
Latest version: {version}
Days since last update: {days_since_last_update}
Days since first release (package age): {days_since_first_release}
Pre-release (0.x): {pre_release}
Monthly downloads: {monthly_downloads}
GitHub stars: {github_stars}

Apply light penalisation (−0.5 to −1.5 pts) for packages that are heavily stale
(>730 days since update), very new (<90 days old), pre-1.0, low downloads, or low stars.

Rate the fit on a scale of 0.0 to 10.0.

Return a JSON object:
{{
  "fit_score": <float 0-10>,
  "fit_notes": "<one sentence explaining the score and any maturity concerns>",
  "suggested_functions": ["<specific function/class names if applicable>"]
}}

Only return the JSON object.
"""


def score_fit(candidate, ctx: RunContext) -> None:
    """Mutates candidate in-place with fit_score, fit_notes, suggested_functions."""
    pre_release = candidate.latest_version.startswith("0.") if candidate.latest_version else None
    prompt = SCORE_PROMPT.format(
        intent=ctx.intent,
        name=candidate.name,
        description=candidate.description,
        pypi_verified=candidate.pypi_verified,
        version=candidate.latest_version or "unknown",
        days_since_last_update=candidate.days_since_release,
        days_since_first_release=candidate.days_since_first_release,
        pre_release=pre_release,
        monthly_downloads=candidate.monthly_downloads,
        github_stars=candidate.github_stars,
    )

    response = _call_with_retry(
        model=LLM_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    _track(response)

    data = json.loads(_strip_fences(response.content[0].text))
    candidate.fit_score = float(data.get("fit_score", 0.0))
    candidate.fit_notes = data.get("fit_notes", "")
    candidate.suggested_functions = data.get("suggested_functions", [])


# ── 5. Query Disambiguation ───────────────────────────────────────────────────

_DISAMBIGUATE_PROMPT = """\
A user has submitted this query to a Python library discovery tool:

"{query}"

Is this query ambiguous — could it describe two or more meaningfully different
technical use-cases that would lead to entirely different library recommendations?

Examples of ambiguous queries:
- "graph library"  → graph visualisation  OR  graph algorithms / networks
- "parser"         → HTML parser  OR  argument parser  OR  grammar/AST parser

If ambiguous, list the distinct interpretations (2–4 max).
If unambiguous, return an empty list.

Return JSON:
{{
  "ambiguous": true/false,
  "interpretations": [
    {{"label": "<short label>", "intent": "<one clear sentence>"}},
    ...
  ]
}}

Only return the JSON object.
"""


def disambiguate_query(raw_query: str) -> list:
    """
    Returns a list of interpretation dicts if the query is ambiguous, else [].
    Each dict: {"label": str, "intent": str}
    """
    try:
        response = _call_with_retry(
            model=LLM_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": _DISAMBIGUATE_PROMPT.format(query=raw_query)}],
        )
        _track(response)
        data = json.loads(_strip_fences(response.content[0].text))
        if data.get("ambiguous"):
            return data.get("interpretations", [])
    except Exception:
        pass
    return []


# ── 6. Relevance Pre-filter ───────────────────────────────────────────────────

_PREFILTER_PROMPT = """\
You are filtering Python library candidates before expensive validation.

User intent: "{intent}"

Candidate libraries discovered so far (name + description):
{candidates_json}

For each candidate, decide if it is OBVIOUSLY IRRELEVANT to the intent —
meaning a Python developer would immediately rule it out without further
investigation.

Be conservative: only remove clear mismatches. When in doubt, keep it.

Return a JSON array of the names to KEEP:
["name1", "name2", ...]

Only return the JSON array.
"""


def filter_irrelevant(candidates: dict, ctx: RunContext) -> dict:
    """
    Remove obviously irrelevant candidates before validation (saves API calls).
    Returns a (possibly smaller) {name: SearchResult} dict.
    Conservative: if the call fails or filters everything, returns the original.
    """
    if not candidates:
        return candidates

    items = [
        {"name": name, "description": sr.description[:200]}
        for name, sr in candidates.items()
    ]
    prompt = _PREFILTER_PROMPT.format(
        intent=ctx.intent,
        candidates_json=json.dumps(items, indent=2),
    )
    try:
        response = _call_with_retry(
            model=LLM_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        _track(response)
        keep = {
            n.lower().replace("-", "_")
            for n in json.loads(_strip_fences(response.content[0].text))
        }
        filtered = {k: v for k, v in candidates.items() if k in keep}
        return filtered if filtered else candidates  # safety net
    except Exception:
        return candidates


# ── 7. AST Flag Interpretation ────────────────────────────────────────────────

_AST_INTERPRET_PROMPT = """\
A static AST scan of the Python package "{name}" found these flags in its \
__init__.py:

{flags}

Package description: {description}

In 1–2 sentences give a plain-English risk assessment. Is this likely malicious,
incidentally suspicious (e.g. optional telemetry, build tooling), or a false
positive? Be direct and concise.
"""


def interpret_ast_flags(c) -> str:
    """Return a 1–2 sentence LLM risk assessment for the AST flags on candidate c."""
    ast_notes = [n for n in c.safety_notes if "AST" in n]
    if not ast_notes:
        return ""
    try:
        prompt = _AST_INTERPRET_PROMPT.format(
            name=c.name,
            flags="\n".join(ast_notes),
            description=c.description[:300],
        )
        response = _call_with_retry(
            model=LLM_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        _track(response)
        return response.content[0].text.strip()
    except Exception:
        return ""


# ── 8. CVE Contextualisation ─────────────────────────────────────────────────

_CVE_CONTEXT_PROMPT = """\
The Python package "{name}" has these known CVEs according to OSV.dev:

{cve_notes}

Package description: {description}

In 1–2 sentences: are these CVEs likely to affect typical programmatic use of
this library (importing and calling its API)? Or are they scoped to specific
usage patterns (CLI, server deployment, optional features)? Be direct.
"""


def contextualise_cves(c) -> str:
    """Return a 1–2 sentence LLM context note for the CVEs found on candidate c."""
    cve_notes = [n for n in c.safety_notes if "CVE" in n]
    if not cve_notes:
        return ""
    try:
        prompt = _CVE_CONTEXT_PROMPT.format(
            name=c.name,
            cve_notes="\n".join(cve_notes),
            description=c.description[:300],
        )
        response = _call_with_retry(
            model=LLM_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        _track(response)
        return response.content[0].text.strip()
    except Exception:
        return ""


# ── 9. Usage Snippet Generation ───────────────────────────────────────────────

_SNIPPET_PROMPT = """\
You are generating minimal, correct Python usage examples for library \
recommendations.

User task: "{intent}"

For each library below, write a short (4–8 line) Python snippet showing exactly
how to use it for the user's specific task. Use the suggested functions/classes
as a guide. The snippet should be immediately runnable after a standard install.

{libraries_json}

Return a JSON array:
[
  {{
    "name": "<package name, unchanged>",
    "snippet": "<4–8 lines of Python, no markdown fences>"
  }},
  ...
]

Only return the JSON array.
"""


def generate_usage_snippets(candidates: list, ctx: RunContext) -> None:
    """
    Generate usage snippets for the given candidates in a single LLM call.
    Mutates each candidate's usage_snippet field in-place.
    """
    if not candidates:
        return

    libraries = [
        {
            "name": c.name,
            "description": c.description,
            "suggested_functions": c.suggested_functions,
        }
        for c in candidates
    ]
    prompt = _SNIPPET_PROMPT.format(
        intent=ctx.intent,
        libraries_json=json.dumps(libraries, indent=2),
    )
    try:
        response = _call_with_retry(
            model=LLM_MODEL,
            max_tokens=min(4096, 600 * len(candidates)),
            messages=[{"role": "user", "content": prompt}],
        )
        _track(response)
        results = json.loads(_strip_fences(response.content[0].text))
        result_map = {
            item["name"].lower().replace("-", "_"): item for item in results
        }
        for c in candidates:
            item = result_map.get(c.name.lower().replace("-", "_"), {})
            c.usage_snippet = item.get("snippet", "")
    except Exception:
        pass


# ── 10. Top-N Comparison Narrative ────────────────────────────────────────────

_COMPARE_PROMPT = """\
You are helping a Python user choose between library recommendations.

User task: "{intent}"

Top candidates (ranked by fit score):
{candidates_json}

Write a tight 3–4 sentence paragraph comparing these options. Mention the single
most important trade-off between them, then end with a direct recommendation.
Be concise — no padding, no repeating the task description.

Return plain text only — no markdown headers, no bullet points.
"""


def compare_top_candidates(candidates: list, ctx: RunContext) -> str:
    """
    Generate a plain-text comparison narrative for the top candidates.
    Returns a string, or "" on failure.
    """
    if len(candidates) < 2:
        return ""

    items = [
        {
            "name": c.name,
            "fit_score": c.fit_score,
            "fit_notes": c.fit_notes,
            "version": c.latest_version,
            "days_since_update": c.days_since_release,
            "monthly_downloads": c.monthly_downloads,
            "github_stars": c.github_stars,
        }
        for c in candidates
    ]
    try:
        response = _call_with_retry(
            model=LLM_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": _COMPARE_PROMPT.format(
                intent=ctx.intent,
                candidates_json=json.dumps(items, indent=2),
            )}],
        )
        _track(response)
        return response.content[0].text.strip()
    except Exception:
        return ""


# ── 11. LLM-assisted package extraction from free text ────────────────────────

_EXTRACT_PROMPT = """\
You are helping identify Python library recommendations from community posts.

User's task: "{intent}"

Below are text snippets from {source} posts/answers. Each snippet may mention
Python libraries in natural language — e.g. "I used X for this", "check out X",
"X works well here" — without any explicit pip/import syntax.

Extract any Python package names that are genuinely recommended or used for a
task similar to the user's intent. Ignore generic words, concepts, and non-package
names. Only include packages you are reasonably confident exist on PyPI.

Snippets:
{snippets}

Return a JSON array:
[
  {{"name": "<pypi package name, lowercase>", "reason": "<one short phrase>"}},
  ...
]

Return [] if no confident matches. Only return the JSON array.
"""


def extract_packages_from_text(texts: list, intent: str, source: str) -> list:
    """
    Extract PyPI package names from free-text snippets using the LLM.
    Additive — failures silently return [].

    Args:
        texts:  list of raw text strings (post bodies, answer bodies, etc.)
        intent: the user's search intent
        source: label for the source, e.g. "Reddit" or "Stack Overflow"

    Returns:
        list of {"name": str, "reason": str}
    """
    if not texts:
        return []

    snippets = "\n---\n".join(t[:400] for t in texts if t.strip())
    if not snippets:
        return []

    try:
        response = _call_with_retry(
            model=LLM_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": _EXTRACT_PROMPT.format(
                intent=intent,
                source=source,
                snippets=snippets,
            )}],
        )
        _track(response)
        items = json.loads(_strip_fences(response.content[0].text))
        return [
            {
                "name": item["name"].strip().lower().replace("-", "_"),
                "reason": item.get("reason", ""),
            }
            for item in items
            if item.get("name", "").strip()
        ]
    except Exception:
        return []
