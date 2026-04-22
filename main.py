"""
main.py — CLI entry point for pylib_finder.

Usage:
    python main.py "find python function to identify knee in PCA plot"
    python main.py "time series anomaly detection library" --no-skill-cards
    python main.py "NLP tokenizer fast" --show-excluded
"""

import asyncio
import datetime
import shlex
import sys
import os
import concurrent.futures
from typing import List

# ── Rich for terminal display ─────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
    from rich.prompt import Prompt
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from config import RESULTS_DIR, MAX_TOTAL_CANDIDATES
from models import RunContext, Candidate
from llm import (
    expand_query, llm_prior, get_token_usage, reset_token_usage,
    disambiguate_query, filter_irrelevant,
    generate_usage_snippets, compare_top_candidates,
)
from scrapers import (
    scrape_pypi, scrape_package_hints,
    scrape_github, scrape_github_topics,
    scrape_stackoverflow, scrape_reddit,
    scrape_web, scrape_papers_with_code,
    merge_results,
)
from validator import validate_candidates
from output import make_output_dir, write_markdown_report, write_json_results, write_skill_card

console = Console() if HAS_RICH else None


def print_header():
    if HAS_RICH:
        console.print(Panel.fit(
            "[bold cyan]PyLib-Finder[/bold cyan]\n"
            "[dim]AI-assisted Python library discovery with safety validation[/dim]",
            border_style="cyan"
        ))
    else:
        print("=" * 50)
        print("  PyLib Finder — Python Library Discovery")
        print("=" * 50)


def cprint(msg: str, style: str = ""):
    if HAS_RICH:
        console.print(msg)
    else:
        # Strip rich markup for plain output
        import re
        clean = re.sub(r'\[.*?\]', '', msg)
        print(clean)


def run_scrapers(ctx: RunContext) -> dict:
    """Run all scrapers in parallel using ThreadPoolExecutor."""
    cprint("\n[bold]🌐 Searching sources...[/bold]")

    scraper_fns = [
        ("PyPI",             lambda: scrape_pypi(ctx.search_terms)),
        ("Package Hints",    lambda: scrape_package_hints(ctx.package_hints)),
        ("GitHub",           lambda: scrape_github(ctx.search_terms)),
        ("GitHub Topics",    lambda: scrape_github_topics(ctx.search_terms)),
        ("Stack Overflow",   lambda: scrape_stackoverflow(ctx.search_terms, ctx.intent)),
        ("Reddit",           lambda: scrape_reddit(ctx.search_terms, ctx.intent)),
        ("Web",              lambda: scrape_web(ctx.search_terms)),
        ("Papers With Code", lambda: scrape_papers_with_code(ctx.search_terms)),
    ]

    all_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fn): name for name, fn in scraper_fns}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                results = future.result()
                cprint(f"  [green]✓[/green] {name}: {len(results)} candidates")
                all_results.extend(results)
            except Exception as e:
                cprint(f"  [yellow]⚠[/yellow] {name}: failed ({e})")

    return merge_results(all_results)


def display_results(candidates, show_excluded: bool = False):
    """Render the validation results as a rich table."""
    valid = [c for c in candidates if not c.excluded]
    excluded = [c for c in candidates if c.excluded]

    if not valid:
        cprint("\n[red]No valid candidates found.[/red]")
        return

    cprint(f"\n[bold cyan]━━━ RESULTS ({len(valid)} candidates)[/bold cyan]")

    if HAS_RICH:
        table = Table(show_header=True, header_style="bold", border_style="dim")
        table.add_column("#", width=3)
        table.add_column("Package", style="bold cyan", width=18)
        table.add_column("Fit", width=5)
        table.add_column("Safety", width=7)
        table.add_column("Sources", width=22)
        table.add_column("Description", width=55)

        for i, c in enumerate(valid, 1):
            safety = "✅" if c.safety_passed else "⚠️ "
            fit_color = "green" if c.fit_score >= 7 else "yellow" if c.fit_score >= 4 else "red"
            table.add_row(
                str(i),
                c.name,
                f"[{fit_color}]{c.fit_score:.1f}[/{fit_color}]",
                safety,
                ", ".join(c.sources),
                c.description[:80] + ("…" if len(c.description) > 80 else ""),
            )
        console.print(table)
    else:
        for i, c in enumerate(valid, 1):
            safety = "✅" if c.safety_passed else "⚠ "
            print(f"\n  {i}. {c.name:<20} fit: {c.fit_score:.1f}  safety: {safety}")
            print(f"     sources: {', '.join(c.sources)}")
            print(f"     {c.description[:80]}")

    # Show detail for top 3
    cprint("\n[bold]── Details ──────────────────────────────────────────────────[/bold]")
    for i, c in enumerate(valid[:5], 1):
        label = " ⭐ RECOMMENDED" if i == 1 else ""
        cprint(f"\n[bold cyan]{i}. {c.name}[/bold cyan]{label}")
        if c.fit_notes:
            cprint(f"   [dim]{c.fit_notes}[/dim]")
        if c.suggested_functions:
            cprint(f"   [yellow]→[/yellow] {', '.join(c.suggested_functions)}")
        health_parts = []
        if c.latest_version:
            health_parts.append(f"v{c.latest_version}")
        if c.days_since_release is not None:
            health_parts.append(f"{c.days_since_release}d since release")
        if c.monthly_downloads:
            health_parts.append(f"{c.monthly_downloads:,} dl/month")
        if c.github_stars is not None:
            health_parts.append(f"⭐ {c.github_stars:,}")
        if c.license:
            health_parts.append(c.license)
        if health_parts:
            cprint(f"   [dim]{' | '.join(health_parts)}[/dim]")
        if c.pypi_url:
            cprint(f"   [blue]{c.pypi_url}[/blue]")
        if c.safety_notes:
            for note in c.safety_notes:
                cprint(f"   {note}")
        if c.ast_risk_summary:
            cprint(f"   [dim]AST risk: {c.ast_risk_summary}[/dim]")
        if c.cve_summary:
            cprint(f"   [dim]CVE context: {c.cve_summary}[/dim]")

    if excluded:
        cprint(f"\n[dim]⚠ {len(excluded)} candidates excluded (failed safety/authenticity). "
               f"Use --show-excluded to inspect.[/dim]")
        if show_excluded:
            cprint("\n[bold]── Excluded ──[/bold]")
            for c in excluded:
                cprint(f"  [red]✗[/red] {c.name}: {c.exclusion_reason}")


def approval_gate(candidates) -> List:
    """Interactive prompt: which results to generate skill cards for."""
    valid = [c for c in candidates if not c.excluded]
    if not valid:
        return []

    cprint("\n[bold]━━━ SKILL CARDS[/bold]")
    cprint("Generate skill cards for which results?")

    options = "  [A] All  "
    for i, c in enumerate(valid, 1):
        options += f"[{i}] {c.name}  "
    options += "[N] None"
    cprint(options)

    if HAS_RICH:
        choice = Prompt.ask("> ", default="N")
    else:
        choice = input("> ").strip()

    choice = choice.strip().upper()
    if choice == "N" or choice == "":
        return []
    if choice == "A":
        return valid

    # Parse individual numbers
    selected = []
    for part in choice.replace(",", " ").split():
        try:
            idx = int(part)
            if 1 <= idx <= len(valid):
                selected.append(valid[idx - 1])
        except ValueError:
            pass
    return selected


def replay_skill_cards(json_path: str):
    """Load a previous results.json and re-run the skill card approval gate."""
    import json as _json

    if not os.path.exists(json_path):
        cprint(f"[red]File not found: {json_path}[/red]")
        return

    with open(json_path) as f:
        data = _json.load(f)

    output_dir = os.path.dirname(os.path.abspath(json_path))

    # Reconstruct Candidate objects from saved JSON
    candidates = []
    for item in data.get("candidates", []):
        if item.get("excluded"):
            continue
        c = Candidate(name=item["name"])
        for key, val in item.items():
            if hasattr(c, key):
                setattr(c, key, val)
        candidates.append(c)

    if not candidates:
        cprint("[yellow]No valid (non-excluded) candidates found in this results file.[/yellow]")
        return

    cprint(f"\n[dim]Loaded {len(candidates)} candidates from:[/dim]")
    cprint(f"[blue]{json_path}[/blue]")

    # Show a quick summary table so the user knows what they're choosing from
    cprint(f"\n[bold cyan]━━━ CANDIDATES[/bold cyan]")
    for i, c in enumerate(candidates, 1):
        safety = "✅" if c.safety_passed else "⚠️ "
        fit_color = "green" if c.fit_score >= 7 else "yellow" if c.fit_score >= 4 else "red"
        if HAS_RICH:
            console.print(
                f"  [bold]{i}.[/bold] [cyan]{c.name:<20}[/cyan] "
                f"fit: [{fit_color}]{c.fit_score:.1f}[/{fit_color}]  {safety}  "
                f"[dim]{c.description[:60]}[/dim]"
            )
        else:
            print(f"  {i}. {c.name:<20} fit: {c.fit_score:.1f}  {c.description[:60]}")

    approved = approval_gate(candidates)
    if approved:
        for c in approved:
            path = write_skill_card(c, output_dir)
            cprint(f"  [green]✓[/green] Skill card → [blue]{path}[/blue]")
        cprint(f"\n[green]✅ {len(approved)} skill card(s) written.[/green]")
    else:
        cprint("\n[dim]No skill cards generated.[/dim]")


def run_search(query: str, show_excluded: bool = False, no_skill_cards: bool = False):
    """Run the full library discovery pipeline for a single query."""
    reset_token_usage()

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    ctx = RunContext(raw_query=query, timestamp=ts, output_dir=RESULTS_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Stage 1: Query expansion ──────────────────────────────────────────────
    cprint("\n[bold]🔍 Expanding query...[/bold]")
    try:
        ctx = expand_query(query)
        ctx.timestamp = ts
        ctx.output_dir = RESULTS_DIR
        cprint(f"  [dim]Intent: {ctx.intent}[/dim]")
        cprint(f"  [dim]Search terms: {', '.join(ctx.search_terms)}[/dim]")
    except Exception as e:
        cprint(f"  [red]Query expansion failed: {e}[/red]")
        cprint("  [dim]Proceeding with raw query...[/dim]")
        ctx.intent = query
        ctx.search_terms = [query]

    # ── Stage 1b: Disambiguation ──────────────────────────────────────────────
    try:
        interpretations = disambiguate_query(query)
        if interpretations:
            cprint("\n[bold yellow]⚠ Ambiguous query — multiple interpretations detected:[/bold yellow]")
            for i, interp in enumerate(interpretations, 1):
                cprint(f"  [{i}] [cyan]{interp['label']}[/cyan]: {interp['intent']}")
            cprint(f"  [0] Use original intent (proceed as-is)")
            if HAS_RICH:
                choice_str = Prompt.ask("  Select interpretation", default="0")
            else:
                choice_str = input("  Select interpretation [0]: ").strip() or "0"
            try:
                choice = int(choice_str)
            except ValueError:
                choice = 0
            if 1 <= choice <= len(interpretations):
                chosen = interpretations[choice - 1]
                cprint(f"  [dim]Using: {chosen['intent']}[/dim]")
                ctx.intent = chosen["intent"]
                try:
                    ctx_refined = expand_query(chosen["intent"])
                    ctx.search_terms = ctx_refined.search_terms
                    ctx.package_hints = ctx_refined.package_hints
                except Exception:
                    pass
    except Exception:
        pass

    # ── Stage 2: LLM prior ────────────────────────────────────────────────────
    cprint("\n[bold]🤖 Seeding candidates from LLM prior...[/bold]")
    llm_results = []
    try:
        llm_results = llm_prior(ctx)
        cprint(f"  [green]✓[/green] LLM suggested {len(llm_results)} candidates")
    except Exception as e:
        cprint(f"  [yellow]⚠[/yellow] LLM prior failed: {e}")

    # ── Stage 3: Multi-source scraping ────────────────────────────────────────
    scraped = run_scrapers(ctx)

    all_raw = llm_results + list(scraped.values())
    all_results = {}
    for r in all_raw:
        name = r.name.strip().lower().replace("-", "_")
        if not name:
            continue
        if name not in all_results:
            all_results[name] = r
        else:
            existing_sources = set(all_results[name].source.split(", "))
            existing_sources.add(r.source)
            all_results[name].source = ", ".join(sorted(existing_sources))

    if len(all_results) > MAX_TOTAL_CANDIDATES:
        all_results = dict(list(all_results.items())[:MAX_TOTAL_CANDIDATES])

    cprint(f"\n[bold]  Total unique candidates:[/bold] {len(all_results)}")

    # ── Stage 3b: Relevance pre-filter ───────────────────────────────────────
    cprint("\n[bold]🔎 Pre-filtering for relevance...[/bold]")
    try:
        keys_before = set(all_results.keys())
        all_results = filter_irrelevant(all_results, ctx)
        dropped_keys = sorted(keys_before - set(all_results.keys()))
        if dropped_keys:
            cprint(f"  [dim]Dropped {len(dropped_keys)} clearly irrelevant candidates "
                   f"({len(all_results)} remaining):[/dim]")
            for name in dropped_keys:
                cprint(f"    [dim]✗ {name}[/dim]")
        else:
            cprint(f"  [dim]All {len(all_results)} candidates kept[/dim]")
    except Exception as e:
        cprint(f"  [yellow]⚠[/yellow] Pre-filter skipped ({e})")

    # ── Stage 4: Validation ───────────────────────────────────────────────────
    cprint(f"\n[bold]🛡️  Validating {len(all_results)} candidates...[/bold]")

    validated_count = [0]
    def progress_cb(name, stage):
        validated_count[0] += 1
        cprint(f"  [dim][{validated_count[0]}/{len(all_results)}] {name}: {stage}[/dim]")

    candidates = validate_candidates(all_results, ctx, progress_callback=progress_cb)

    # ── Stage 4b: Comparison narrative ───────────────────────────────────────
    top_candidates = [c for c in candidates if not c.excluded][:5]
    comparison = ""
    if len(top_candidates) >= 2:
        cprint("\n[bold]🤖 Generating comparison narrative...[/bold]")
        try:
            comparison = compare_top_candidates(top_candidates[:3], ctx)
        except Exception as e:
            cprint(f"  [yellow]⚠[/yellow] Comparison skipped ({e})")

    # ── Stage 5: Display results ──────────────────────────────────────────────
    display_results(candidates, show_excluded=show_excluded)

    if comparison:
        cprint("\n[bold]── Comparison ───────────────────────────────────────────────[/bold]")
        cprint(f"[dim]{comparison}[/dim]")

    # ── Token usage summary ───────────────────────────────────────────────────
    usage = get_token_usage()
    total_tokens = usage["input"] + usage["output"]
    cprint(
        f"\n[dim]🔢 Token usage — "
        f"input: {usage['input']:,}  "
        f"output: {usage['output']:,}  "
        f"total: {total_tokens:,}  "
        f"({usage['calls']} LLM calls)[/dim]"
    )

    # ── Stage 6: Save outputs ─────────────────────────────────────────────────
    output_dir = make_output_dir(ctx)
    write_markdown_report(candidates, ctx, comparison=comparison)
    write_json_results(candidates, ctx, token_usage=usage, comparison=comparison)
    cprint(f"\n[bold]📁 Results saved →[/bold] [blue]{output_dir}[/blue]")
    cprint(f"   [dim]report.md, results.json[/dim]")

    # ── Stage 7: Skill card approval gate ─────────────────────────────────────
    if not no_skill_cards:
        approved = approval_gate(candidates)
        if approved:
            cprint("\n[bold]✍  Generating usage snippets for approved candidates...[/bold]")
            try:
                generate_usage_snippets(approved, ctx)
            except Exception as e:
                cprint(f"  [yellow]⚠[/yellow] Snippet generation skipped ({e})")
            for c in approved:
                path = write_skill_card(c, output_dir)
                cprint(f"  [green]✓[/green] Skill card → [blue]{path}[/blue]")
            cprint(f"\n[green]✅ Done! {len(approved)} skill card(s) written.[/green]")
        else:
            cprint("\n[dim]No skill cards generated.[/dim]")
    else:
        cprint("\n[dim]Skill card generation skipped.[/dim]")


def _show_repl_help():
    cprint("\n[bold]Commands:[/bold]")
    cprint("  [cyan]search \"<query>\"[/cyan]                             — find libraries for a task")
    cprint("  [cyan]search \"<query>\" --show-excluded[/cyan]             — also show excluded packages")
    cprint("  [cyan]search \"<query>\" --no-skill-cards[/cyan]            — skip skill card prompt")
    cprint("  [cyan]generate-skill <results.json>[/cyan]                 — write skill cards from a past run")
    cprint("  [cyan]help[/cyan]                                          — show this message")
    cprint("  [cyan]quit[/cyan]                                          — exit\n")


def repl():
    """Interactive shell — runs until the user types quit/exit."""
    print_header()
    cprint("[dim]Interactive mode. Type [bold]help[/bold] for available commands.[/dim]")

    while True:
        try:
            if HAS_RICH:
                raw = Prompt.ask("\n[bold cyan]pylib>[/bold cyan]")
            else:
                raw = input("\npylib> ").strip()
        except (KeyboardInterrupt, EOFError):
            cprint("\n[dim]Goodbye![/dim]")
            break

        raw = raw.strip()
        if not raw:
            continue

        if raw.lower() in ("quit", "exit", "q"):
            cprint("[dim]Goodbye![/dim]")
            break

        if raw.lower() in ("help", "?"):
            _show_repl_help()
            continue

        try:
            parts = shlex.split(raw)
        except ValueError as e:
            cprint(f"[red]Parse error: {e}[/red]")
            continue

        cmd = parts[0].lower()

        if cmd == "generate-skill":
            if len(parts) < 2:
                cprint("[red]Usage: generate-skill <path/to/results.json>[/red]")
                continue
            replay_skill_cards(parts[1])

        elif cmd == "search":
            # Extract flags, collect remaining tokens as the query
            flags = {p for p in parts[1:] if p.startswith("--")}
            query_parts = [p for p in parts[1:] if not p.startswith("--")]
            if not query_parts:
                cprint("[red]Usage: search \"<query>\" [--show-excluded] [--no-skill-cards][/red]")
                continue
            query = " ".join(query_parts)
            run_search(
                query,
                show_excluded="--show-excluded" in flags,
                no_skill_cards="--no-skill-cards" in flags,
            )

        else:
            cprint(f"[red]Unknown command '{parts[0]}'.[/red]  Type [bold]help[/bold] for available commands.")


def main():
    repl()


if __name__ == "__main__":
    main()
