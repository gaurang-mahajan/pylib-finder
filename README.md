# pylib_finder

AI-assisted Python library discovery with safety validation. Describe a data science task in plain English and get back ranked, validated Python library candidates — with safety checks, health signals, and optional machine-readable skill cards.

---

## Setup

```bash
# 1. Clone and install dependencies
git clone https://github.com/gaurang-mahajan/pylib-finder.git
cd pylib-finder
pip install -r requirements.txt

# 2. Add your API key
cp .env.example .env
# Open .env and set ANTHROPIC_API_KEY (required) and GITHUB_TOKEN (optional).
# The tool loads .env automatically — no shell exports needed.

# 3. Start
python main.py
# then at the prompt:
#   search "find python function to locate knee in PCA plot"
```

---

## Usage

Start the shell, then fire commands as needed:

```bash
python main.py
```

```
pylib> search "time series anomaly detection"
pylib> search "image augmentation" --show-excluded
pylib> search "dimensionality reduction" --no-skill-cards
pylib> generate-skill ./pylib_results/2026-02-27_143022_knee-pca/results.json
pylib> help
pylib> quit
```

| Flag | Effect |
|------|--------|
| `--show-excluded` | Also display packages that failed safety/authenticity checks |
| `--no-skill-cards` | Skip the skill card approval prompt at the end |

---

## Output

Each search run creates a timestamped folder under `pylib_results/`:

```
pylib_results/
  └── 2026-02-27_143022_locate-knee-in-pca-plot/
        ├── report.md        ← human-readable ranked summary
        ├── results.json     ← full structured data (all candidates + token usage)
        └── skills/          ← skill cards, only written after user approval
              └── kneed.yaml
```

Token usage for the run is printed at the end of every search and saved in `results.json` under the `token_usage` key.

---

## Workflow

The pipeline has seven stages. Each is labelled as **LLM** (Claude API call) or **deterministic** (no AI, just APIs and code).

### Stage 1 — Query Expansion `[LLM]`
Claude takes the raw user query and produces: a one-sentence intent description, 3–6 search phrases for use across sources, and 2–4 likely package name fragments. This structured output drives all downstream searches and is used as the reference for fit scoring later.

### Stage 1b — Query Disambiguation `[LLM]`
Immediately after expansion, Claude checks whether the query is ambiguous enough to produce meaningfully different library recommendations under different interpretations (e.g. "graph library" → visualisation vs. algorithms). If multiple interpretations are detected, the user is shown a numbered list and asked to pick one. The chosen intent then drives a fresh search-term expansion before scraping begins.

### Stage 2 — LLM Prior `[LLM]`
Before any external search runs, Claude suggests packages it already knows from training (up to 8 candidates). These "seeded candidates" surface niche-but-correct packages that don't rank well in keyword search. They enter the pipeline tagged `source: llm` and must pass all the same validation checks as any other candidate.

### Stage 3 — Multi-Source Scraping `[deterministic]`
Eight scrapers run in parallel via `ThreadPoolExecutor`, each returning package name candidates:

- **Package Hints** — direct PyPI JSON API lookup for the name fragments produced in Stage 1. These are the highest-confidence candidates — the LLM already believes they exist on PyPI.
- **PyPI** — searches pypi.org using the same endpoint as `pip search`, without a topic classifier restriction so non-scientific packages are also discovered.
- **GitHub** — searches repositories via the GitHub REST API filtered to Python, sorted by stars. Extracts the PyPI package name from `pip install` or `import` mentions in the repo description before falling back to the repo name itself. Embeds star count as a health signal.
- **GitHub Topics** — searches GitHub repos whose *topic tags* match the search term slug (e.g. `topic:anomaly-detection`). Topic-tagged repos are curated by their maintainers and tend to be more purpose-built than plain keyword matches.
- **Stack Overflow** — queries the StackExchange API for relevant Python questions. Extracts package names from question titles and top-voted answer bodies using `pip install` / `import` regex patterns, then runs a second **LLM pass** over the answer bodies to catch natural-language recommendations (e.g. *"I'd use statsmodels for this"*) that regex misses.
- **Reddit** — searches r/datascience, r/learnpython, r/MachineLearning, r/Python, r/LanguageTechnology, r/bioinformatics and r/genomics. Applies the same two-step approach: regex on post bodies first, then an **LLM pass** over collected post texts to extract conversational package mentions.
- **Web** — DuckDuckGo HTML search, extracts `pip install <pkg>` and `import <pkg>` patterns from snippets.
- **Papers With Code** — queries the [Papers With Code API](https://paperswithcode.com/api/v1/) for academic ML/AI papers that have public code. Extracts GitHub repository names as candidate package names, sorted by star count. Particularly strong for cutting-edge ML queries.

Results from all sources are merged and deduplicated by package name. A candidate found by multiple sources gets a combined `sources` list.

Source priority for description selection (highest to lowest): `hints > llm > pypi > github > github_topics > paperswithcode > stackoverflow > reddit > web`.

### Stage 3b — Relevance Pre-filter `[LLM]`
After scraping and merging, a single LLM call reviews all candidate names and descriptions and removes obvious mismatches before the expensive validation stage. This is intentionally conservative — only clear mismatches are dropped. The call typically saves several PyPI/OSV/pypistats API calls per run.

### Stage 4 — Validation `[deterministic + LLM]`
Each unique candidate goes through four sub-stages:

1. **Authenticity** `[deterministic]` — calls the PyPI JSON API (`pypi.org/pypi/{name}/json`) to confirm the package exists. Populates version, license, GitHub URL, Python version support, and days since last release. Packages not on PyPI are excluded immediately.

   Authenticity, safety, and health checks all run **in parallel** via `ThreadPoolExecutor` (up to 8 workers), cutting Stage 4 wall-clock time by roughly the number of candidates.

2. **Safety** `[deterministic]` — four checks:
   - *OSV vulnerability scan*: queries [osv.dev](https://osv.dev) for known CVEs. A confirmed CVE is grounds for exclusion.
   - *License check*: flags packages with no declared license (`ℹ`) and packages under restrictive licenses — GPL, AGPL, EUPL, SSPL, etc. — that may limit commercial or proprietary use (`⚠`). Configurable via `RESTRICTIVE_LICENSES` in `config.py`. Does not cause exclusion.
   - *PyPI metadata signals*: flags yanked latest releases, packages with only a single version ever published, no project URLs, and wheel-only packages where source cannot be inspected.
   - *AST scan*: downloads the source tarball from PyPI, extracts `__init__.py`, and walks the AST for: suspicious calls (`exec`, `eval`, `compile`, `subprocess`, `ctypes`/`CDLL`, `getenv`, network calls); `os.environ` attribute access; hardcoded IPv4 addresses; and sensitive credential paths (`.ssh`, `.aws/credentials`, etc.). Also scans **every string constant in the entire AST** for large base64-like blobs, catching inline patterns like `exec(base64.b64decode("..."))`. Best-effort; no code is executed.

3. **Health** `[deterministic]` — fetches download stats from [pypistats.org](https://pypistats.org). Also surfaces GitHub star count (if scraped). Issues warnings (never exclusions) for:
   - Not updated in 2+ years (`DAYS_SINCE_RELEASE_WARN`)
   - Fewer downloads than threshold (`MIN_MONTHLY_DOWNLOADS`)
   - First released within 90 days (`DAYS_NEW_PACKAGE_WARN`) — flagged as a safety warning given higher risk of instability
   - Version 0.x (`PRE_RELEASE_WARN`) — flagged as informational

4. **Fit scoring** `[LLM]` — Claude scores **all non-excluded candidates in a single API call** (batch scoring), returning a 0–10 score, a one-sentence explanation, and suggested function/class names. The model is explicitly given maturity signals — `days_since_last_update`, `days_since_first_release`, `pre_release`, `monthly_downloads`, and `github_stars` — and instructed to apply light penalisation for stale, new, or pre-release packages. Falls back to per-candidate calls if the batch call fails.

5. **AST flag interpretation** `[LLM]` — for any candidate whose AST scan raised flags, a short LLM call produces a 1–2 sentence plain-English risk assessment (likely malicious / incidentally suspicious / false positive). Stored as `ast_risk_summary` and shown in the details section.

6. **CVE contextualisation** `[LLM]` — for any candidate with known CVEs, a short LLM call assesses whether the CVEs affect typical programmatic (API) use of the library or are scoped to CLI, server deployment, or optional features. Stored as `cve_summary`.

All LLM calls use **exponential backoff** (up to 4 retries, 2→4→8→16 s delays) on transient server errors (429 rate-limit, 529 overloaded, 5xx). A `Retry-After` header is honoured when present.

### Stage 4b — Usage Snippets `[LLM]`
After validation, a single LLM call generates a 4–8 line Python usage example for each of the top 5 candidates, tailored to the user's specific task using the `suggested_functions` already returned by the scorer. Snippets appear in the terminal detail view, the markdown report, and the YAML skill cards.

### Stage 5 — Display `[deterministic]`
Results are printed to the terminal as a ranked table (via `rich`) with a details section for the top 5. Each entry shows fit notes, suggested functions, health metrics, safety notes, AST risk summary (if any), CVE context (if any), and usage snippet. Excluded candidates are summarised as a count; `--show-excluded` reveals the reasons.

A **comparison narrative** is printed after the table — a 4–6 sentence paragraph comparing the top 3 candidates, covering trade-offs in maturity, API style, maintenance, and fit, ending with a direct recommendation.

### Stage 6 — Save outputs `[deterministic]`
`report.md` and `results.json` are written to the timestamped output folder. Both include `usage_snippet`, `ast_risk_summary`, `cve_summary`, and the top-3 `comparison` narrative. `results.json` also includes all candidate data and the token usage summary for the run.

### Stage 7 — Skill card approval gate `[deterministic]`
The user is prompted to select which candidates to generate skill cards for. Only approved candidates get a `.yaml` file written under `skills/`. This step can be re-run at any time on a past `results.json` without making any new API calls:

```bash
# from the interactive shell:
#   generate-skill ./pylib_results/<run-folder>/results.json
#
# or one-shot:
python main.py generate-skill ./pylib_results/<run-folder>/results.json
```

---

## Tests

```bash
# Run all unit tests (no API key required, no network calls)
python tests.py

# Verbose output
python tests.py -v

# Integration tests run automatically when ANTHROPIC_API_KEY is set
ANTHROPIC_API_KEY=sk-ant-... python tests.py -v
```

The test suite has 71 tests covering:
- Config values and new settings
- Model dataclass fields (`github_stars`)
- `merge_results` deduplication, source priority, and normalisation
- `_extract_pypi_name_from_repo` — pip/import extraction and fallbacks
- All new scrapers (`scrape_package_hints`, `scrape_github_topics`, `scrape_papers_with_code`) with mocked HTTP
- `build_candidate` — `github_stars` extraction from embedded tags
- `check_authenticity`, `check_health`, `decide_exclusion` with mocked PyPI/pypistats
- `score_fit_batch` — mutation, empty input, and missing-candidate tolerance
- `_strip_fences` — markdown fence removal
- Parallel validation correctness
- JSON output structure and `github_stars` field
- Integration tests for `expand_query`, `llm_prior`, `score_fit_batch` (skipped when no API key)

---

## File structure

```
pylib_finder/
  main.py           ← CLI entry point, pipeline orchestration, subcommands
  config.py         ← tuneable settings (thresholds, timeouts, model, workers)
  models.py         ← shared dataclasses (Candidate, SearchResult, RunContext)
  llm.py            ← Claude calls: query expansion, LLM prior, batch fit scoring
                      also tracks and exposes token usage for the session
  scrapers.py       ← 8 scrapers + merger: PyPI, hints, GitHub, GitHub Topics,
                      Stack Overflow, Reddit, Web, Papers With Code
  validator.py      ← parallel authenticity (PyPI), safety (OSV + AST), health,
                      and batch fit scoring orchestration
  output.py         ← markdown report, results.json, skill card YAML writer
  tests.py          ← 71 unit + integration tests
  requirements.txt  ← anthropic, rich (all other deps are stdlib)
  .env.example      ← template for required environment variables
  .gitignore        ← excludes pylib_results/, .env, __pycache__, venvs
  ABOUT.md          ← plain-English overview of the tool
  pylib_results/    ← timestamped run output (gitignored, kept via .gitkeep)
```

---

## Configuration

Key settings in `config.py`:

| Setting | Default | Effect |
|---|---|---|
| `LLM_MODEL` | `claude-opus-4-6` | Model used for all LLM stages |
| `MAX_CANDIDATES_PER_SOURCE` | `8` | Hits to pull per scraper |
| `MAX_TOTAL_CANDIDATES` | `32` | Hard cap on candidates entering validation |
| `MAX_VALIDATION_WORKERS` | `8` | ThreadPoolExecutor workers for parallel validation |
| `PYPI_MAX_TERMS` | `3` | Search terms sent to PyPI scraper |
| `SCRAPER_MAX_TERMS` | `2` | Search terms sent to GitHub, SO, Web, Papers With Code |
| `REDDIT_MAX_TERMS` | `2` | Search terms sent per subreddit |
| `REDDIT_MAX_SUBREDDITS` | `4` | Subreddits searched (top N from ordered list) |
| `DAYS_SINCE_RELEASE_WARN` | `730` | Warn if package not updated in this many days |
| `DAYS_NEW_PACKAGE_WARN` | `90` | Warn if package first released within this many days |
| `PRE_RELEASE_WARN` | `True` | Show info note for version 0.x packages |
| `MIN_MONTHLY_DOWNLOADS` | `50` | Warn if below this monthly download threshold |
| `HTTP_TIMEOUT` | `10` | Seconds before a scraper request times out |
| `GITHUB_TOKEN` | `.env` / env var | Optional GitHub PAT — raises API rate limit |
| `SUSPICIOUS_AST_PATTERNS` | see file | Function names that trigger AST safety warnings |
