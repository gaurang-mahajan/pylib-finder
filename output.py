"""
output.py — Write human-readable and machine-readable results to disk.
"""

import json
import os
import datetime
import re
from typing import List

from models import Candidate, RunContext


# ── Directory setup ───────────────────────────────────────────────────────────

def make_output_dir(ctx: RunContext) -> str:
    """Create a timestamped output folder and return its path."""
    slug = re.sub(r'[^a-z0-9]+', '-', ctx.raw_query.lower())[:40].strip('-')
    folder_name = f"{ctx.timestamp}_{slug}"
    path = os.path.join(ctx.output_dir, folder_name)
    os.makedirs(path, exist_ok=True)
    ctx.output_dir = path
    return path


# ── Markdown report ───────────────────────────────────────────────────────────

def _safety_icon(c: Candidate) -> str:
    if not c.pypi_verified:
        return "❌"
    if c.cve_clean is False:
        return "❌"
    critical = [n for n in c.safety_notes if "⚠" in n and "CVE" in n]
    if critical:
        return "⚠️ "
    return "✅"


def _health_str(c: Candidate) -> str:
    parts = []
    if c.latest_version:
        parts.append(f"v{c.latest_version}")
    if c.days_since_release is not None:
        if c.days_since_release < 30:
            parts.append("updated recently")
        elif c.days_since_release < 365:
            parts.append(f"updated {c.days_since_release}d ago")
        else:
            parts.append(f"updated {c.days_since_release // 365}y ago")
    if c.monthly_downloads is not None:
        dl = c.monthly_downloads
        parts.append(f"{dl:,} dl/month" if dl < 1_000_000 else f"{dl/1e6:.1f}M dl/month")
    if c.github_stars is not None:
        parts.append(f"⭐ {c.github_stars:,}")
    if c.license:
        parts.append(c.license)
    return " | ".join(parts) if parts else "—"


def write_markdown_report(candidates: List[Candidate], ctx: RunContext, comparison: str = "") -> str:
    """Write report.md and return file path."""
    lines = []

    lines.append(f"# PyLib Finder Report\n")
    lines.append(f"**Query:** {ctx.raw_query}  ")
    lines.append(f"**Intent:** {ctx.intent}  ")
    lines.append(f"**Date:** {ctx.timestamp}  \n")
    lines.append("---\n")

    valid = [c for c in candidates if not c.excluded]
    excluded = [c for c in candidates if c.excluded]

    lines.append(f"## Results ({len(valid)} candidates, {len(excluded)} excluded)\n")

    for i, c in enumerate(valid, 1):
        rank_label = "⭐ RECOMMENDED" if i == 1 else f"#{i}"
        lines.append(f"### {i}. `{c.name}` — {rank_label}")
        lines.append(f"**Fit score:** {c.fit_score:.1f}/10  ")
        lines.append(f"**Safety:** {_safety_icon(c)}  ")
        lines.append(f"**Sources:** {', '.join(c.sources)}  \n")

        lines.append(f"{c.description}\n")

        if c.fit_notes:
            lines.append(f"> {c.fit_notes}\n")

        if c.suggested_functions:
            lines.append(f"**Key functions/classes:** `{'`, `'.join(c.suggested_functions)}`  ")

        lines.append(f"**Health:** {_health_str(c)}  ")

        if c.pypi_url:
            lines.append(f"**PyPI:** {c.pypi_url}  ")
        if c.github_url:
            lines.append(f"**GitHub:** {c.github_url}  ")
        if c.docs_url:
            lines.append(f"**Docs:** {c.docs_url}  ")

        if c.safety_notes:
            lines.append(f"\n**Notes:**")
            for note in c.safety_notes:
                lines.append(f"- {note}")

        if c.ast_risk_summary:
            lines.append(f"\n**AST risk assessment:** {c.ast_risk_summary}")
        if c.cve_summary:
            lines.append(f"\n**CVE context:** {c.cve_summary}")

        if c.usage_snippet:
            lines.append(f"\n**Usage example:**")
            lines.append(f"```python\n{c.usage_snippet}\n```")

        lines.append("\n---\n")

    if comparison:
        lines.append(f"\n## Comparison\n")
        lines.append(comparison)
        lines.append("")

    if excluded:
        lines.append(f"\n## Excluded Candidates ({len(excluded)})\n")
        lines.append("| Package | Reason |")
        lines.append("|---------|--------|")
        for c in excluded:
            lines.append(f"| `{c.name}` | {c.exclusion_reason} |")
        # Surface CVE/AST context for excluded candidates where available
        for c in excluded:
            if c.cve_summary or c.ast_risk_summary:
                lines.append(f"\n**`{c.name}`**")
                if c.cve_summary:
                    lines.append(f"- CVE context: {c.cve_summary}")
                if c.ast_risk_summary:
                    lines.append(f"- AST risk: {c.ast_risk_summary}")

    content = "\n".join(lines)
    path = os.path.join(ctx.output_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ── JSON results ──────────────────────────────────────────────────────────────

def write_json_results(
    candidates: List[Candidate],
    ctx: RunContext,
    token_usage: dict = None,
    comparison: str = "",
) -> str:
    """Write results.json and return file path."""
    data = {
        "query": ctx.raw_query,
        "intent": ctx.intent,
        "search_terms": ctx.search_terms,
        "timestamp": ctx.timestamp,
        "token_usage": token_usage or {},
        "comparison": comparison,
        "candidates": [],
    }

    for c in candidates:
        data["candidates"].append({
            "name": c.name,
            "excluded": c.excluded,
            "exclusion_reason": c.exclusion_reason,
            "sources": c.sources,
            "description": c.description,
            "fit_score": c.fit_score,
            "fit_notes": c.fit_notes,
            "suggested_functions": c.suggested_functions,
            "usage_snippet": c.usage_snippet,
            "ast_risk_summary": c.ast_risk_summary,
            "cve_summary": c.cve_summary,
            "pypi_verified": c.pypi_verified,
            "latest_version": c.latest_version,
            "days_since_release": c.days_since_release,
            "days_since_first_release": c.days_since_first_release,
            "monthly_downloads": c.monthly_downloads,
            "license": c.license,
            "safety_passed": c.safety_passed,
            "cve_clean": c.cve_clean,
            "safety_notes": c.safety_notes,
            "github_stars": c.github_stars,
            "pypi_url": c.pypi_url,
            "github_url": c.github_url,
            "docs_url": c.docs_url,
        })

    path = os.path.join(ctx.output_dir, "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


# ── Skill card writer ─────────────────────────────────────────────────────────

def write_skill_card(c: Candidate, output_dir: str) -> str:
    """Write a YAML skill card for a single approved candidate. Returns file path."""
    import_name = c.name.replace("-", "_")

    lines = [
        f"name: {c.name}",
        f"version: \"{c.latest_version}\"",
        f"install: pip install {c.name}",
        f"imports: [{import_name}]",
        f"primary_use: \"{c.description[:250]}\"",
    ]

    if c.suggested_functions:
        lines.append(f"key_functions:")
        for fn in c.suggested_functions:
            lines.append(f"  - {fn}")

    if c.usage_snippet:
        # Indent snippet lines for YAML literal block scalar
        indented = "\n".join(f"    {ln}" for ln in c.usage_snippet.splitlines())
        lines.append(f"usage_example: |")
        lines.append(indented)

    lines.append(f"safety:")
    lines.append(f"  pypi_verified: {str(c.pypi_verified).lower()}")
    lines.append(f"  cve_clean: {str(c.cve_clean).lower()}")
    lines.append(f"  safety_passed: {str(c.safety_passed).lower()}")

    lines.append(f"fit_score: {c.fit_score}")
    lines.append(f"license: \"{c.license}\"")

    if c.pypi_url:
        lines.append(f"pypi_url: {c.pypi_url}")
    if c.github_url:
        lines.append(f"github_url: {c.github_url}")
    if c.docs_url:
        lines.append(f"docs_url: {c.docs_url}")

    skills_dir = os.path.join(output_dir, "skills")
    os.makedirs(skills_dir, exist_ok=True)

    path = os.path.join(skills_dir, f"{c.name}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path
