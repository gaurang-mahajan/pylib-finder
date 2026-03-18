"""
tests.py — Unit and integration tests for pylib_finder.

Run all tests (no network required for unit tests):
    python tests.py

Run with verbose output:
    python tests.py -v

Integration tests (require ANTHROPIC_API_KEY) are skipped automatically
when the key is absent.
"""

import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from typing import List

# ── Ensure project root is on the path ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from models import Candidate, SearchResult, RunContext
from scrapers import (
    merge_results,
    _extract_pypi_name_from_repo,
    scrape_package_hints,
    scrape_papers_with_code,
    scrape_github_topics,
    _get_json,
)
from validator import build_candidate, check_authenticity, check_health, decide_exclusion
from output import _health_str, write_json_results, write_markdown_report
import config


# ─────────────────────────────────────────────────────────────────────────────
# Helper factories
# ─────────────────────────────────────────────────────────────────────────────

def make_sr(name, source="pypi", description="", url="") -> SearchResult:
    return SearchResult(name=name, source=source, description=description, url=url)


def make_candidate(name="testpkg", excluded=False, fit_score=7.0) -> Candidate:
    c = Candidate(name=name)
    c.excluded = excluded
    c.fit_score = fit_score
    c.pypi_verified = not excluded
    c.safety_passed = not excluded
    c.latest_version = "1.0.0"
    c.monthly_downloads = 5000
    return c


# ─────────────────────────────────────────────────────────────────────────────
# config.py
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig(unittest.TestCase):
    def test_candidate_cap_raised(self):
        # Must be higher than the original default of 5
        self.assertGreater(config.MAX_CANDIDATES_PER_SOURCE, 5)

    def test_total_candidates_raised(self):
        # Must be higher than the original default of 20
        self.assertGreater(config.MAX_TOTAL_CANDIDATES, 20)

    def test_validation_workers_set(self):
        self.assertGreaterEqual(config.MAX_VALIDATION_WORKERS, 4)

    def test_github_token_attr_exists(self):
        self.assertTrue(hasattr(config, "GITHUB_TOKEN"))


# ─────────────────────────────────────────────────────────────────────────────
# models.py
# ─────────────────────────────────────────────────────────────────────────────

class TestModels(unittest.TestCase):
    def test_candidate_has_github_stars(self):
        c = Candidate(name="foo")
        self.assertIsNone(c.github_stars)

    def test_candidate_defaults(self):
        c = Candidate(name="foo")
        self.assertFalse(c.excluded)
        self.assertFalse(c.pypi_verified)
        self.assertEqual(c.fit_score, 0.0)
        self.assertEqual(c.sources, [])

    def test_run_context_defaults(self):
        ctx = RunContext(raw_query="test query")
        self.assertEqual(ctx.search_terms, [])
        self.assertEqual(ctx.package_hints, [])


# ─────────────────────────────────────────────────────────────────────────────
# scrapers.py — merge_results
# ─────────────────────────────────────────────────────────────────────────────

class TestMergeResults(unittest.TestCase):
    def test_deduplication_by_name(self):
        results = [
            make_sr("numpy", "pypi"),
            make_sr("numpy", "github"),
        ]
        merged = merge_results(results)
        self.assertEqual(len(merged), 1)
        self.assertIn("numpy", merged)

    def test_sources_combined(self):
        results = [
            make_sr("pandas", "pypi"),
            make_sr("pandas", "github"),
            make_sr("pandas", "reddit"),
        ]
        merged = merge_results(results)
        sources = merged["pandas"].source
        self.assertIn("pypi", sources)
        self.assertIn("github", sources)
        self.assertIn("reddit", sources)

    def test_source_priority_hints_over_llm(self):
        results = [
            make_sr("kneed", "llm", description="from llm"),
            make_sr("kneed", "hints", description="from hints"),
        ]
        merged = merge_results(results)
        # hints has priority 0, llm has priority 1 → hints description kept
        self.assertEqual(merged["kneed"].description, "from hints")

    def test_source_priority_pypi_over_github(self):
        results = [
            make_sr("scipy", "github", description="github desc"),
            make_sr("scipy", "pypi", description="pypi desc"),
        ]
        merged = merge_results(results)
        self.assertEqual(merged["scipy"].description, "pypi desc")

    def test_hyphen_to_underscore_normalisation(self):
        results = [
            make_sr("scikit-learn", "pypi"),
            make_sr("scikit_learn", "github"),
        ]
        merged = merge_results(results)
        self.assertEqual(len(merged), 1)

    def test_empty_names_skipped(self):
        results = [make_sr("", "pypi"), make_sr("  ", "github")]
        merged = merge_results(results)
        self.assertEqual(len(merged), 0)

    def test_short_names_skipped(self):
        results = [make_sr("a", "pypi")]
        merged = merge_results(results)
        self.assertEqual(len(merged), 0)

    def test_new_sources_in_priority(self):
        """hints, github_topics, paperswithcode must be in priority map."""
        from scrapers import merge_results as _mr
        # Test by checking the priority ordering holds
        results = [
            make_sr("pkg", "paperswithcode", description="pwc"),
            make_sr("pkg", "hints", description="hint"),
        ]
        merged = _mr(results)
        self.assertEqual(merged["pkg"].description, "hint")


# ─────────────────────────────────────────────────────────────────────────────
# scrapers.py — _extract_pypi_name_from_repo
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractPypiName(unittest.TestCase):
    def test_pip_install_in_description(self):
        item = {"name": "my-repo", "description": "Install via pip install mypackage"}
        self.assertEqual(_extract_pypi_name_from_repo(item), "mypackage")

    def test_import_in_description(self):
        item = {"name": "my-repo", "description": "Use import coollib in your code"}
        self.assertEqual(_extract_pypi_name_from_repo(item), "coollib")

    def test_fallback_to_repo_name(self):
        item = {"name": "anomaly-detection", "description": "A great tool"}
        self.assertEqual(_extract_pypi_name_from_repo(item), "anomaly_detection")

    def test_hyphens_normalised(self):
        item = {"name": "time-series-lib", "description": ""}
        result = _extract_pypi_name_from_repo(item)
        self.assertNotIn("-", result)

    def test_empty_name(self):
        item = {"name": "", "description": ""}
        result = _extract_pypi_name_from_repo(item)
        self.assertFalse(result)  # None or "" — both are falsy


# ─────────────────────────────────────────────────────────────────────────────
# scrapers.py — scrape_package_hints (mocked HTTP)
# ─────────────────────────────────────────────────────────────────────────────

class TestScrapePackageHints(unittest.TestCase):
    def _mock_pypi_response(self, name):
        return {
            "info": {
                "name": name,
                "summary": f"Test package {name}",
                "package_url": f"https://pypi.org/project/{name}/",
            }
        }

    @patch("scrapers._get_json")
    @patch("scrapers.time")
    def test_returns_results_for_valid_hints(self, mock_time, mock_get_json):
        mock_get_json.return_value = self._mock_pypi_response("kneed")
        results = scrape_package_hints(["kneed"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "kneed")
        self.assertEqual(results[0].source, "hints")

    @patch("scrapers._get_json")
    @patch("scrapers.time")
    def test_skips_missing_packages(self, mock_time, mock_get_json):
        mock_get_json.return_value = None
        results = scrape_package_hints(["nonexistentxyz123"])
        self.assertEqual(len(results), 0)

    @patch("scrapers._get_json")
    @patch("scrapers.time")
    def test_empty_hints_returns_empty(self, mock_time, mock_get_json):
        results = scrape_package_hints([])
        self.assertEqual(results, [])
        mock_get_json.assert_not_called()

    @patch("scrapers._get_json")
    @patch("scrapers.time")
    def test_normalises_hyphens(self, mock_time, mock_get_json):
        mock_get_json.return_value = self._mock_pypi_response("scikit_learn")
        results = scrape_package_hints(["scikit-learn"])
        self.assertEqual(results[0].name, "scikit_learn")


# ─────────────────────────────────────────────────────────────────────────────
# scrapers.py — scrape_papers_with_code (mocked HTTP)
# ─────────────────────────────────────────────────────────────────────────────

class TestScrapePapersWithCode(unittest.TestCase):
    def _mock_pwc_response(self):
        return {
            "results": [
                {
                    "title": "Deep Anomaly Detection",
                    "repositories": [
                        {"url": "https://github.com/user/deep-anomaly", "stars": 500},
                        {"url": "https://github.com/user/another-impl", "stars": 100},
                    ],
                }
            ]
        }

    @patch("scrapers._get_json")
    @patch("scrapers.time")
    def test_returns_search_results(self, mock_time, mock_get_json):
        mock_get_json.return_value = self._mock_pwc_response()
        results = scrape_papers_with_code(["anomaly detection"])
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].source, "paperswithcode")

    @patch("scrapers._get_json")
    @patch("scrapers.time")
    def test_normalises_repo_name(self, mock_time, mock_get_json):
        mock_get_json.return_value = self._mock_pwc_response()
        results = scrape_papers_with_code(["anomaly detection"])
        for r in results:
            self.assertNotIn("-", r.name)

    @patch("scrapers._get_json")
    @patch("scrapers.time")
    def test_embeds_stars_in_description(self, mock_time, mock_get_json):
        mock_get_json.return_value = self._mock_pwc_response()
        results = scrape_papers_with_code(["anomaly detection"])
        first = next((r for r in results if "deep_anomaly" in r.name), None)
        self.assertIsNotNone(first)
        self.assertIn("github_stars:500", first.description)

    @patch("scrapers._get_json")
    @patch("scrapers.time")
    def test_handles_no_results(self, mock_time, mock_get_json):
        mock_get_json.return_value = {"results": []}
        results = scrape_papers_with_code(["obscure query xyz"])
        self.assertEqual(results, [])

    @patch("scrapers._get_json")
    @patch("scrapers.time")
    def test_handles_api_failure(self, mock_time, mock_get_json):
        mock_get_json.return_value = None
        results = scrape_papers_with_code(["some query"])
        self.assertEqual(results, [])


# ─────────────────────────────────────────────────────────────────────────────
# scrapers.py — scrape_github_topics (mocked HTTP)
# ─────────────────────────────────────────────────────────────────────────────

class TestScrapeGithubTopics(unittest.TestCase):
    def _mock_github_response(self):
        return {
            "items": [
                {
                    "name": "anomaly-detector",
                    "description": "pip install anomalydetector — fast anomaly detection",
                    "html_url": "https://github.com/user/anomaly-detector",
                    "stargazers_count": 1200,
                }
            ]
        }

    @patch("scrapers._get_json")
    @patch("scrapers.time")
    def test_returns_results(self, mock_time, mock_get_json):
        mock_get_json.return_value = self._mock_github_response()
        results = scrape_github_topics(["anomaly detection"])
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].source, "github_topics")

    @patch("scrapers._get_json")
    @patch("scrapers.time")
    def test_extracts_pip_install_name(self, mock_time, mock_get_json):
        mock_get_json.return_value = self._mock_github_response()
        results = scrape_github_topics(["anomaly detection"])
        # Description has "pip install anomalydetector" so name should be extracted
        self.assertEqual(results[0].name, "anomalydetector")

    @patch("scrapers._get_json")
    @patch("scrapers.time")
    def test_stars_embedded_in_description(self, mock_time, mock_get_json):
        mock_get_json.return_value = self._mock_github_response()
        results = scrape_github_topics(["anomaly detection"])
        self.assertIn("github_stars:1200", results[0].description)

    @patch("scrapers._get_json")
    @patch("scrapers.time")
    def test_handles_empty_response(self, mock_time, mock_get_json):
        mock_get_json.return_value = None
        results = scrape_github_topics(["some term"])
        self.assertEqual(results, [])


# ─────────────────────────────────────────────────────────────────────────────
# validator.py — build_candidate
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildCandidate(unittest.TestCase):
    def test_basic_fields(self):
        sr = make_sr("pandas", "pypi", "A data analysis library", "https://pypi.org/project/pandas")
        c = build_candidate(sr)
        self.assertEqual(c.name, "pandas")
        self.assertIn("pypi", c.sources)
        self.assertEqual(c.pypi_url, "https://pypi.org/project/pandas")

    def test_github_url_assigned(self):
        sr = make_sr("mylib", "github", url="https://github.com/user/mylib")
        c = build_candidate(sr)
        self.assertEqual(c.github_url, "https://github.com/user/mylib")

    def test_github_stars_extracted_from_description(self):
        sr = make_sr("mylib", "github", description="A tool [github_stars:2500]")
        c = build_candidate(sr)
        self.assertEqual(c.github_stars, 2500)
        self.assertNotIn("github_stars", c.description)

    def test_stars_tag_stripped_from_description(self):
        sr = make_sr("mylib", "github", description="Fast library [github_stars:100]")
        c = build_candidate(sr)
        self.assertEqual(c.description, "Fast library")

    def test_multiple_sources_split(self):
        sr = make_sr("pkg", "pypi, github, reddit")
        c = build_candidate(sr)
        self.assertIn("pypi", c.sources)
        self.assertIn("github", c.sources)
        self.assertIn("reddit", c.sources)


# ─────────────────────────────────────────────────────────────────────────────
# validator.py — check_authenticity (mocked PyPI)
# ─────────────────────────────────────────────────────────────────────────────

MOCK_PYPI_DATA = {
    "info": {
        "name": "requests",
        "version": "2.31.0",
        "summary": "Python HTTP for Humans.",
        "license": "Apache-2.0",
        "package_url": "https://pypi.org/project/requests/",
        "docs_url": None,
        "home_page": "https://requests.readthedocs.io",
        "classifiers": [
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.7",
        ],
        "project_urls": {
            "Source": "https://github.com/psf/requests",
        },
    },
    "releases": {
        "2.31.0": [{"upload_time": "2023-05-22T14:00:00", "packagetype": "sdist"}]
    },
    "urls": [],
}


class TestCheckAuthenticity(unittest.TestCase):
    @patch("validator.fetch_pypi_details", return_value=MOCK_PYPI_DATA)
    def test_sets_pypi_verified(self, mock_fetch):
        c = Candidate(name="requests")
        check_authenticity(c)
        self.assertTrue(c.pypi_verified)

    @patch("validator.fetch_pypi_details", return_value=MOCK_PYPI_DATA)
    def test_populates_version(self, mock_fetch):
        c = Candidate(name="requests")
        check_authenticity(c)
        self.assertEqual(c.latest_version, "2.31.0")

    @patch("validator.fetch_pypi_details", return_value=MOCK_PYPI_DATA)
    def test_populates_license(self, mock_fetch):
        c = Candidate(name="requests")
        check_authenticity(c)
        self.assertEqual(c.license, "Apache-2.0")

    @patch("validator.fetch_pypi_details", return_value=MOCK_PYPI_DATA)
    def test_populates_github_url(self, mock_fetch):
        c = Candidate(name="requests")
        check_authenticity(c)
        self.assertIn("github.com", c.github_url)

    @patch("validator.fetch_pypi_details", return_value={})
    def test_marks_unverified_when_not_on_pypi(self, mock_fetch):
        c = Candidate(name="nonexistentxyz999")
        check_authenticity(c)
        self.assertFalse(c.pypi_verified)

    @patch("validator.fetch_pypi_details", return_value=MOCK_PYPI_DATA)
    def test_populates_days_since_release(self, mock_fetch):
        c = Candidate(name="requests")
        check_authenticity(c)
        self.assertIsNotNone(c.days_since_release)
        self.assertGreater(c.days_since_release, 0)


# ─────────────────────────────────────────────────────────────────────────────
# validator.py — decide_exclusion
# ─────────────────────────────────────────────────────────────────────────────

class TestDecideExclusion(unittest.TestCase):
    def test_excludes_not_on_pypi(self):
        c = Candidate(name="fakepkg")
        c.pypi_verified = False
        decide_exclusion(c)
        self.assertTrue(c.excluded)
        self.assertIn("PyPI", c.exclusion_reason)

    def test_excludes_cve_packages(self):
        c = Candidate(name="vulnpkg")
        c.pypi_verified = True
        c.cve_clean = False
        decide_exclusion(c)
        self.assertTrue(c.excluded)
        self.assertIn("CVE", c.exclusion_reason)

    def test_passes_clean_package(self):
        c = Candidate(name="cleanpkg")
        c.pypi_verified = True
        c.cve_clean = True
        decide_exclusion(c)
        self.assertFalse(c.excluded)

    def test_excludes_obfuscated_package(self):
        c = Candidate(name="suspicious")
        c.pypi_verified = True
        c.cve_clean = True
        c.safety_notes = ["⚠ Suspicious AST patterns: possible_base64_obfuscation"]
        decide_exclusion(c)
        self.assertTrue(c.excluded)


# ─────────────────────────────────────────────────────────────────────────────
# validator.py — check_health (mocked pypistats)
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckHealth(unittest.TestCase):
    @patch("validator._get_json", return_value={"data": {"last_month": 250000}})
    def test_populates_monthly_downloads(self, mock_get):
        c = Candidate(name="requests")
        c.pypi_verified = True
        check_health(c)
        self.assertEqual(c.monthly_downloads, 250000)

    @patch("validator._get_json", return_value={"data": {"last_month": 10}})
    def test_flags_low_traction(self, mock_get):
        c = Candidate(name="obscurepkg")
        c.pypi_verified = True
        c.days_since_release = 100
        check_health(c)
        self.assertTrue(any("Low download" in n for n in c.safety_notes))

    @patch("validator._get_json", return_value={"data": {"last_month": 5000}})
    def test_flags_stale_package(self, mock_get):
        c = Candidate(name="oldpkg")
        c.pypi_verified = True
        c.days_since_release = 800
        check_health(c)
        self.assertTrue(any("Not updated" in n for n in c.safety_notes))

    def test_skips_unverified(self):
        c = Candidate(name="fakepkg")
        c.pypi_verified = False
        check_health(c)
        self.assertIsNone(c.monthly_downloads)


# ─────────────────────────────────────────────────────────────────────────────
# llm.py — score_fit_batch (mocked Anthropic client)
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreFitBatch(unittest.TestCase):
    def _make_mock_response(self, candidates):
        scores = [
            {
                "name": c.name,
                "fit_score": 8.5,
                "fit_notes": "Excellent match for the task.",
                "suggested_functions": ["KneeLocator"],
            }
            for c in candidates
        ]
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=json.dumps(scores))]
        mock_resp.usage = MagicMock(input_tokens=100, output_tokens=200)
        return mock_resp

    @patch("llm._get_client")
    def test_mutates_candidates_in_place(self, mock_get_client):
        candidates = [make_candidate("kneed"), make_candidate("scipy")]
        ctx = RunContext(raw_query="find knee in curve", intent="detect knee point")

        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_response(candidates)
        mock_get_client.return_value = mock_client

        from llm import score_fit_batch
        score_fit_batch(candidates, ctx)

        self.assertEqual(candidates[0].fit_score, 8.5)
        self.assertEqual(candidates[1].fit_score, 8.5)
        self.assertEqual(candidates[0].fit_notes, "Excellent match for the task.")
        self.assertIn("KneeLocator", candidates[0].suggested_functions)

    @patch("llm._get_client")
    def test_handles_empty_list(self, mock_get_client):
        ctx = RunContext(raw_query="test", intent="test")
        from llm import score_fit_batch
        score_fit_batch([], ctx)
        mock_get_client.assert_not_called()

    @patch("llm._get_client")
    def test_tolerates_missing_candidate_in_response(self, mock_get_client):
        """If LLM omits a candidate, that candidate should get a fallback note."""
        c1 = make_candidate("kneed")
        c2 = make_candidate("scipy", fit_score=0.0)  # start at 0 to verify it stays 0
        ctx = RunContext(raw_query="test", intent="test")

        # Response only includes c1, not c2
        partial_scores = [{"name": "kneed", "fit_score": 9.0, "fit_notes": "Great", "suggested_functions": []}]
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=json.dumps(partial_scores))]
        mock_resp.usage = MagicMock(input_tokens=50, output_tokens=80)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp
        mock_get_client.return_value = mock_client

        from llm import score_fit_batch
        score_fit_batch([c1, c2], ctx)

        self.assertEqual(c1.fit_score, 9.0)
        self.assertEqual(c2.fit_score, 0.0)
        self.assertIn("unavailable", c2.fit_notes)


# ─────────────────────────────────────────────────────────────────────────────
# llm.py — _strip_fences
# ─────────────────────────────────────────────────────────────────────────────

class TestStripFences(unittest.TestCase):
    def test_strips_json_fence(self):
        from llm import _strip_fences
        text = "```json\n{\"key\": 1}\n```"
        self.assertEqual(_strip_fences(text), '{"key": 1}')

    def test_strips_plain_fence(self):
        from llm import _strip_fences
        text = "```\n[1, 2, 3]\n```"
        self.assertEqual(_strip_fences(text), "[1, 2, 3]")

    def test_passthrough_plain_json(self):
        from llm import _strip_fences
        text = '{"hello": "world"}'
        self.assertEqual(_strip_fences(text), '{"hello": "world"}')


# ─────────────────────────────────────────────────────────────────────────────
# output.py — _health_str
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthStr(unittest.TestCase):
    def test_includes_version(self):
        c = make_candidate()
        c.latest_version = "2.0.0"
        self.assertIn("v2.0.0", _health_str(c))

    def test_includes_downloads(self):
        c = make_candidate()
        c.monthly_downloads = 123456
        self.assertIn("123,456", _health_str(c))

    def test_includes_million_downloads(self):
        c = make_candidate()
        c.monthly_downloads = 2_500_000
        self.assertIn("2.5M", _health_str(c))

    def test_includes_github_stars(self):
        c = make_candidate()
        c.github_stars = 3200
        self.assertIn("3,200", _health_str(c))
        self.assertIn("⭐", _health_str(c))

    def test_includes_license(self):
        c = make_candidate()
        c.license = "MIT"
        self.assertIn("MIT", _health_str(c))

    def test_returns_dash_when_empty(self):
        c = Candidate(name="empty")
        self.assertEqual(_health_str(c), "—")


# ─────────────────────────────────────────────────────────────────────────────
# output.py — write_json_results includes github_stars
# ─────────────────────────────────────────────────────────────────────────────

class TestWriteJsonResults(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def test_github_stars_in_json(self):
        import tempfile, json as _json

        c = make_candidate("testpkg")
        c.github_stars = 999
        ctx = RunContext(raw_query="test", intent="test")
        ctx.output_dir = self.tmpdir
        ctx.timestamp = "2026-01-01_000000"

        write_json_results([c], ctx)

        path = os.path.join(self.tmpdir, "results.json")
        with open(path) as f:
            data = _json.load(f)

        self.assertEqual(data["candidates"][0]["github_stars"], 999)

    def test_json_structure(self):
        import tempfile, json as _json

        c = make_candidate("pkg1")
        ctx = RunContext(raw_query="test query", intent="test intent")
        ctx.output_dir = self.tmpdir
        ctx.timestamp = "2026-01-01_000000"

        write_json_results([c], ctx)

        path = os.path.join(self.tmpdir, "results.json")
        with open(path) as f:
            data = _json.load(f)

        self.assertIn("query", data)
        self.assertIn("candidates", data)
        self.assertIn("token_usage", data)


# ─────────────────────────────────────────────────────────────────────────────
# validator.py — parallel vs sequential equivalence (mocked network)
# ─────────────────────────────────────────────────────────────────────────────

class TestParallelValidation(unittest.TestCase):
    @patch("validator.fetch_pypi_details", return_value=MOCK_PYPI_DATA)
    @patch("validator._get_json", return_value={"data": {"last_month": 50000}})
    @patch("validator._check_osv")
    @patch("validator._check_ast")
    def test_all_candidates_processed(self, mock_ast, mock_osv, mock_stats, mock_pypi):
        """Parallel validation returns one Candidate per input."""
        from validator import validate_candidates

        search_results = {
            "pkg_a": make_sr("pkg_a", "pypi"),
            "pkg_b": make_sr("pkg_b", "github"),
            "pkg_c": make_sr("pkg_c", "reddit"),
        }
        ctx = RunContext(raw_query="test", intent="test")

        # Patch score_fit_batch to avoid real LLM calls
        with patch("llm.score_fit_batch") as mock_batch:
            mock_batch.return_value = None
            candidates = validate_candidates(search_results, ctx)

        self.assertEqual(len(candidates), 3)

    @patch("validator.fetch_pypi_details", return_value={})
    @patch("validator._check_osv")
    @patch("validator._check_ast")
    def test_failed_pypi_auth_excluded(self, mock_ast, mock_osv, mock_pypi):
        """Packages not on PyPI must be excluded."""
        from validator import validate_candidates

        search_results = {"fakepkg": make_sr("fakepkg", "web")}
        ctx = RunContext(raw_query="test", intent="test")

        with patch("llm.score_fit_batch"):
            candidates = validate_candidates(search_results, ctx)

        self.assertTrue(candidates[0].excluded)


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests (require ANTHROPIC_API_KEY — skipped if absent)
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY not set")
class TestIntegration(unittest.TestCase):
    """Live tests that make real API calls. Run only when key is available."""

    def test_expand_query_returns_context(self):
        from llm import expand_query, reset_token_usage
        reset_token_usage()
        ctx = expand_query("find python library for time series anomaly detection")
        self.assertIsInstance(ctx.intent, str)
        self.assertGreater(len(ctx.search_terms), 0)
        self.assertGreater(len(ctx.package_hints), 0)

    def test_llm_prior_returns_results(self):
        from llm import llm_prior, reset_token_usage
        reset_token_usage()
        ctx = RunContext(raw_query="test", intent="detect anomalies in time series data")
        results = llm_prior(ctx)
        self.assertGreater(len(results), 0)
        self.assertTrue(all(r.source == "llm" for r in results))

    def test_score_fit_batch_real(self):
        from llm import score_fit_batch, reset_token_usage
        reset_token_usage()
        ctx = RunContext(raw_query="test", intent="detect anomalies in time series data")
        c1 = make_candidate("pyod")
        c1.description = "Python Outlier Detection toolkit"
        c1.pypi_verified = True
        c1.latest_version = "1.1.3"
        score_fit_batch([c1], ctx)
        self.assertGreater(c1.fit_score, 0.0)
        self.assertIsInstance(c1.fit_notes, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
