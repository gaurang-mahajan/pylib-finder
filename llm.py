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

Libraries to evaluate:
{libraries_json}

Return a JSON array in the SAME ORDER as the input, one object per library:
[
  {{
    "name": "<package name, unchanged>",
    "fit_score": <float 0.0-10.0>,
    "fit_notes": "<one sentence explaining the score>",
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

Rate the fit on a scale of 0.0 to 10.0.

Return a JSON object:
{{
  "fit_score": <float 0-10>,
  "fit_notes": "<one sentence explaining the score>",
  "suggested_functions": ["<specific function/class names if applicable>"]
}}

Only return the JSON object.
"""


def score_fit(candidate, ctx: RunContext) -> None:
    """Mutates candidate in-place with fit_score, fit_notes, suggested_functions."""
    prompt = SCORE_PROMPT.format(
        intent=ctx.intent,
        name=candidate.name,
        description=candidate.description,
        pypi_verified=candidate.pypi_verified,
        version=candidate.latest_version or "unknown",
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
