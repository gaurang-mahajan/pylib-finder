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

# 3. Run
python main.py "find python function to locate knee in PCA plot"
```

---

## Usage

```bash
# Full search (subcommand form)
python main.py search "time series anomaly detection"

# Bare query also works — shorthand for 'search'
python main.py "NLP tokenizer fast"

# Show packages that were excluded (failed safety/authenticity checks)
python main.py search "image augmentation" --show-excluded

# Skip skill card prompt at the end
python main.py search "dimensionality reduction" --no-skill-cards

# Generate skill cards from a previous run (no API calls needed)
python main.py generate-skill ./pylib_results/2026-02-27_143022_knee-pca/results.json
```

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

### Stage 2 — LLM Prior `[LLM]`
Before any external search runs, Claude suggests packages it already knows from training (up to 8 candidates). These "seeded candidates" surface niche-but-correct packages that don't rank well in keyword search. They enter the pipeline tagged `source: llm` and must pass all the same validation checks as any other candidate.

### Stage 3 — Multi-Source Scraping `[deterministic]`
Eight scrapers run in parallel via `ThreadPoolExecutor`, each returning package name candidates:

- **Package Hints** — direct PyPI JSON API lookup for the name fragments produced in Stage 1. These are the highest-confidence candidates — the LLM already believes they exist on PyPI.
- **PyPI** — searches pypi.org using the same endpoint as `pip search`, without a topic classifier restriction so non-scientific packages are also discovered.
- **GitHub** — searches repositories via the GitHub REST API filtered to Python, sorted by stars. Extracts the PyPI package name from `pip install` or `import` mentions in the repo description before falling back to the repo name itself. Embeds star count as a health signal.
- **GitHub Topics** — searches GitHub repos whose *topic tags* match the search term slug (e.g. `topic:anomaly-detection`). Topic-tagged repos are curated by their maintainers and tend to be more purpose-built than plain keyword matches.
- **Stack Overflow** — queries the StackExchange API for relevant Python questions. Extracts package names from both question titles and the body of top-voted answers, using explicit `pip install` / `import` patterns for higher precision.
- **Reddit** — searches r/datascience, r/learnpython, r/MachineLearning, r/Python, r/LanguageTechnology, r/bioinformatics and r/genomics. Covers 2 search terms and up to 800 characters of each post body.
- **Web** — DuckDuckGo HTML search, extracts `pip install <pkg>` and `import <pkg>` patterns from snippets.
- **Papers With Code** — queries the [Papers With Code API](https://paperswithcode.com/api/v1/) for academic ML/AI papers that have public code. Extracts GitHub repository names as candidate package names, sorted by star count. Particularly strong for cutting-edge ML queries.

Results from all sources are merged and deduplicated by package name. A candidate found by multiple sources gets a combined `sources` list.

Source priority for description selection (highest to lowest): `hints > llm > pypi > github > github_topics > paperswithcode > stackoverflow > reddit > web`.

### Stage 4 — Validation `[deterministic + LLM]`
Each unique candidate goes through four sub-stages:

1. **Authenticity** `[deterministic]` — calls the PyPI JSON API (`pypi.org/pypi/{name}/json`) to confirm the package exists. Populates version, license, GitHub URL, Python version support, and days since last release. Packages not on PyPI are excluded immediately.

   Authenticity, safety, and health checks all run **in parallel** via `ThreadPoolExecutor` (up to 8 workers), cutting Stage 4 wall-clock time by roughly the number of candidates.

2. **Safety** `[deterministic]` — two checks:
   - *OSV vulnerability scan*: queries [osv.dev](https://osv.dev) for known CVEs. A confirmed CVE is grounds for exclusion.
   - *AST scan*: downloads the source tarball from PyPI, extracts `__init__.py`, and walks the AST for suspicious patterns (`exec`, `eval`, `subprocess`, network calls, base64 obfuscation). Best-effort; no code is executed.

3. **Health** `[deterministic]` — fetches download stats from [pypistats.org](https://pypistats.org). Also surfaces GitHub star count (if scraped). Flags packages not updated in 2+ years or with fewer than 100 downloads/month as warnings, not exclusions.

4. **Fit scoring** `[LLM]` — Claude scores **all non-excluded candidates in a single API call** (batch scoring), returning a 0–10 score, a one-sentence explanation, and suggested function/class names. Falls back to per-candidate calls if the batch call fails. This replaces the original N-calls-per-run approach and cuts LLM cost for Stage 4 by 10–20×.

### Stage 5 — Display `[deterministic]`
Results are printed to the terminal as a ranked table (via `rich`) with a detail section for the top 5. GitHub stars are shown alongside download counts in the health line. Excluded candidates are summarised as a count; `--show-excluded` reveals the reasons.

### Stage 6 — Save outputs `[deterministic]`
`report.md` and `results.json` are written to the timestamped output folder. `results.json` includes all candidate data (including `github_stars`) plus the token usage summary for the run.

### Stage 7 — Skill card approval gate `[deterministic]`
The user is prompted to select which candidates to generate skill cards for. Only approved candidates get a `.yaml` file written under `skills/`. This step can be re-run at any time on a past `results.json` without making any new API calls:

```bash
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
| `DAYS_SINCE_RELEASE_WARN` | `730` | Flag packages not updated in this many days |
| `MIN_MONTHLY_DOWNLOADS` | `50` | Flag packages below this download threshold |
| `HTTP_TIMEOUT` | `10` | Seconds before a scraper request times out |
| `GITHUB_TOKEN` | `.env` / env var | Optional GitHub PAT — raises API rate limit |
| `SUSPICIOUS_AST_PATTERNS` | see file | Function names that trigger AST safety warnings |
