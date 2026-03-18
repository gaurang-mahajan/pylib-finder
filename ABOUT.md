# Pylib-Finder — What It Is and Why It Exists

## The problem it solves

You're working on something specific — fitting a pharmacokinetic curve, simulating a biological pathway, locating the knee in a PCA plot — and you wonder: *has someone already built a Python package for this I could re-use?*

The obvious move is a web search. You skim Stack Overflow threads, find a GitHub repo with a promising name, then spend another 15 minutes checking whether it's maintained, whether anyone's flagged it as broken or malicious, and whether you've even found the best-fit option. Sometimes you might miss the right package entirely because it didn't surface in the first few results.

**pylib_finder** automates that whole process. You describe what you need in plain English, it searches multiple sources at once, filters out anything sketchy or abandoned, and returns a short ranked list of real installable packages — with the specific functions to try first.

---

## Why not just ask an LLM directly?

You could ask an AI directly. The problem is that LLMs could hallucinate package names, recommend things that were abandoned years ago, and have no real-time signal on security vulnerabilities, download trends, or maintenance status.

pylib_finder uses LLM reasoning where it's strong — understanding your intent, knowing the Python ecosystem, evaluating relevance — and uses live APIs for everything that needs to be factual and current.

---

## Is it "agentic"?

It's a structured **multi-step workflow** rather than a fully autonomous agent. The same seven stages run in order every time; there's no loop where the tool decides what to do next. LLM calls happen at three specific points: interpreting your query, seeding an initial candidate list from the model's own knowledge, and rating each validated candidate for relevance. Everything in between is deterministic code. So, it's more of an **LLM-guided search pipeline**.

---

## How it works

**1. Query understanding** — An LLM rephrases your query into a clean intent statement and generates search phrases. It also guesses likely package name fragments for direct high-confidence lookups.

**2. LLM prior seeding** — Before any external search, the LLM is asked what packages it already knows for this task based on its training knowledge. This surfaces niche tools that wouldn't show up in keyword search.

**3. Multi-source parallel search** — PyPI, GitHub (by stars), GitHub Topics (maintainer-curated topic tags), Stack Overflow (question titles and answer bodies), Reddit (r/datascience, r/MachineLearning, and others), web search (DuckDuckGo), Papers With Code (academic ML implementations), and direct PyPI name lookups. Results are merged and deduplicated — a package found in five places is a stronger signal than one found in one.

**4. Validation** — Every candidate is checked in parallel against live sources: PyPI confirms it exists and returns version, license, and Python support; OSV checks for known CVEs (packages with confirmed vulnerabilities are dropped); pypistats returns monthly downloads and last-update age. A static AST scan also inspects the published source for suspicious patterns like `eval`, `exec`, or network calls at import time — nothing is executed.

**5. Relevance scoring** — All validated candidates are scored in a single LLM call (0–10) with a one-sentence explanation and suggested function names. The model sees each package's description, download count, and GitHub stars alongside your original intent.

**6. Output** — A ranked table in the terminal, a markdown report, and a full JSON file saved to a timestamped folder. You can optionally generate YAML skill cards for the top picks — compact records of how to install, import, and call each package, useful as personal notes or as context for other AI tools.

---

## The value

A ranked shortlist of real, safe, installable packages — not guesses — with specific function names, health signals, and a saved record of the search. Most useful for **niche or domain-specific needs** where the best package might not be one with 10k stars and a casual web search might either miss it or bury it.
