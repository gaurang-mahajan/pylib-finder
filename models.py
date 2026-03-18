"""
models.py — Shared dataclasses used throughout pylib_finder.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Candidate:
    """A single library candidate moving through the pipeline."""
    name: str                               # PyPI / package name
    sources: List[str] = field(default_factory=list)  # e.g. ["llm", "pypi", "so"]
    description: str = ""                  # short description from discovery
    pypi_url: str = ""
    github_url: str = ""
    docs_url: str = ""

    # Validation fields — populated by validation pipeline
    pypi_verified: bool = False            # exists on PyPI
    latest_version: str = ""
    days_since_release: Optional[int] = None
    monthly_downloads: Optional[int] = None
    license: str = ""
    python_versions: List[str] = field(default_factory=list)

    github_stars: Optional[int] = None     # GitHub star count, if available

    safety_passed: bool = False
    safety_notes: List[str] = field(default_factory=list)
    cve_clean: Optional[bool] = None       # None = not checked

    fit_score: float = 0.0                 # 0–10, LLM-assigned
    fit_notes: str = ""                    # why this score
    suggested_functions: List[str] = field(default_factory=list)

    excluded: bool = False
    exclusion_reason: str = ""


@dataclass
class SearchResult:
    """Thin wrapper returned by each scraper before merging."""
    name: str
    source: str                            # "pypi" | "github" | "so" | "reddit" | "web" | "llm"
    description: str = ""
    url: str = ""


@dataclass
class RunContext:
    """Carries the user query and its expanded forms through the whole run."""
    raw_query: str
    intent: str = ""                       # 1-sentence intent, LLM-generated
    search_terms: List[str] = field(default_factory=list)
    package_hints: List[str] = field(default_factory=list)  # name fragments
    timestamp: str = ""
    output_dir: str = ""
