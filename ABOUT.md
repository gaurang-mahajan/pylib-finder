# Pylib-Finder — What It Is and Why It Exists

## The problem it solves

You're working on something specific — fitting a pharmacokinetic curve, simulating a biological pathway, locating the knee in a PCA plot — and you wonder: *has someone already built a Python package for this I could re-use?*

The obvious place to start is a web search. You skim Stack Overflow threads, find a GitHub repo with a promising name, then spend another 15 minutes checking whether it's maintained, whether anyone's flagged it as broken or malicious, and whether you've even found the best-fit option. Sometimes you might miss the right package entirely because it didn't surface in the first few results.

**pylib_finder** automates that whole process. You describe what you need in plain English, it searches multiple sources at once, filters out anything sketchy or abandoned, and returns a short ranked list of real installable packages — with the specific functions to try first.

---

## Why not just ask an LLM directly?

You could ask an AI directly. But LLMs could hallucinate package names, recommend things that were abandoned years ago, and have no real-time signal on security vulnerabilities, download trends, or maintenance status.

pylib_finder uses LLM reasoning where it's strong — understanding your intent, knowing the Python ecosystem, evaluating relevance — and uses live APIs for everything that needs to be factual and current.

---

## Is it "agentic"?

It's a structured **multi-step workflow** rather than a fully autonomous agent. The same stages run in order every time; there's no loop where the tool decides what to do next. LLM calls happen at seven specific points: disambiguating ambiguous queries, interpreting your query into search terms, seeding an initial candidate list from the model's own knowledge, pre-filtering obviously irrelevant scraped candidates, rating each validated candidate for relevance, interpreting any suspicious AST flags, contextualising CVE findings, generating usage snippets, and producing a comparison narrative for the top picks. Everything else is deterministic code. So, it's more of an **LLM-guided search pipeline**.

---

## How it works

**1. Query understanding** — An LLM rephrases your query into a clean intent statement and generates search phrases. It also guesses likely package name fragments for direct high-confidence lookups. If the query is ambiguous (e.g. "graph library" could mean visualisation or algorithms), it surfaces the distinct interpretations and lets you pick before any search begins.

**2. LLM prior seeding** — Before any external search, the LLM is asked what packages it already knows for this task based on its training knowledge. This surfaces niche tools that wouldn't show up in keyword search.

**3. Multi-source parallel search** — PyPI, GitHub (by stars), GitHub Topics (maintainer-curated topic tags), Stack Overflow (question titles and answer bodies), Reddit (r/datascience, r/MachineLearning, and others), web search (DuckDuckGo), Papers With Code (academic ML implementations), and direct PyPI name lookups. For Reddit and Stack Overflow, a regex pass extracts explicit `pip install` / `import` mentions first, then a second **LLM pass** reads the raw post and answer text to catch natural-language recommendations like *"I've been using kneed for this"* that regex misses entirely. Results are merged and deduplicated — a package found in five places is a stronger signal than one found in one. A further LLM pre-filter then removes obvious mismatches before the expensive validation stage.

**4. Validation** — Every candidate is checked in parallel against live sources:

- **Authenticity** — PyPI confirms the package exists and returns its version, license, and Python version support.
- **CVE scan** — OSV.dev is queried for known vulnerabilities. Confirmed CVEs cause the package to be excluded, with an LLM-generated note explaining whether they affect typical API use.
- **Health** — pypistats returns monthly downloads and days since last update. Warnings are issued for packages not updated in 2+ years, very new packages (under 90 days old), and pre-1.0 versions.
- **License** — packages with no declared license get an info note; copyleft licenses (GPL, AGPL, EUPL, SSPL, etc.) get a warning in case they conflict with proprietary or commercial use.
- **PyPI metadata red flags** — yanked latest releases, packages with only one version ever published, missing project URLs, and wheel-only packages where source code cannot be inspected.
- **AST scan** — the source tarball is downloaded into memory, `__init__.py` is parsed, and the AST is walked for suspicious calls (`exec`, `eval`, `ctypes`, `getenv`, network calls), `os.environ` access, hardcoded IP addresses, sensitive credential paths (`.ssh`, `.aws/credentials`, etc.), and large base64-like string blobs anywhere in the tree — catching inline obfuscation like `exec(base64.b64decode("..."))`. Nothing is written to disk and no code is ever executed. Where flags are found, an LLM call produces a plain-English risk assessment.

**5. Relevance scoring** — All validated candidates are scored in a single LLM call (0–10) with a one-sentence explanation and suggested function names. The model is explicitly given maturity signals — days since last update, package age, pre-release status, download count, and GitHub stars — and instructed to factor them into the score. The `fit_notes` in the output will mention maturity concerns where relevant.

**6. Output** — A ranked table in the terminal, followed by a comparison paragraph that weighs the top 3 picks against each other and ends with a direct recommendation. Each top candidate also gets a short usage snippet showing how to use it for your specific task. Everything is saved — a markdown report, a full JSON file, and optionally YAML skill cards — to a timestamped folder.

---

## The value

A ranked shortlist of real, safe, installable packages — not guesses — with specific function names, health signals, and a saved record of the search. Most useful for **niche or domain-specific needs** where the best package might not be one with 10k stars and a casual web search might either miss it or bury it.
