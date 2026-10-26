"""Tests for Step 7d: round dates inferred from the article, reading the newest pages, traction words, GDELT
switched off. (The page-date tests of 7d moved to test_7e.py when the text-date rule was narrowed.)
Fake servers only (same style as test_research.py)."""
from datetime import date

import pytest

from research import news as news_mod
from research.pipeline import research
from research.scoring import CompanyRecord, compute_signals, figure_is_stale
from research.verify import Fact, Figure, Page, check_fact, infer_round_date, merge_all, merge_field, traction_level
from scripts.run_research import build_parser, cell, month
from test_research import SITE, Site, SmartLlama, rig  # noqa: F401  (rig is a fixture)
from test_research import html as make_html

TODAY = date(2026, 10, 24)
URL = "https://tracxn.com/d/companies/stripe"


# ------------------------------------------------------------------ the Stripe case from the real run
STRIPE_ROW = "Apr 08, 2024 | $694M | Series I | $65B | - | - | -"


def stripe_round(page_day):
    f = Fact("last_round_date", "2024-04", STRIPE_ROW, URL, "", page_day)
    return f, check_fact(f, Page(URL, "Stripe profile", STRIPE_ROW, page_day), "Stripe")


def test_a_round_on_a_page_with_stale_metadata_is_no_longer_rejected():
    assert stripe_round("2023-09-26")[1].unusable                      # what the old metadata date did (7c)
    f, v = stripe_round("2024-04-08")                                  # the page date from its own content (7d)
    fig = merge_field("last_round_date", [(f, v)])
    assert v.status == "verified" and fig.usable and fig.value == "2024-04"
    assert figure_is_stale(fig, TODAY)                                 # April 2024 is older than 18 months: marked, not hidden


# ------------------------------------------------------------------ the round date inferred from the article
RAMP_Q = "Ramp said on Thursday that it had raised $1.85 billion in a new funding round that values it at $60 billion."


def amount_fig(quote, as_of, status="verified"):
    return Figure("last_round_amount", 1_850_000_000, status, True, "https://techcrunch.com/2026/10/15/ramp", quote,
                  "techcrunch.com", ["techcrunch.com"], "", None, as_of)


def test_the_round_date_is_inferred_from_the_article_date():
    fig = infer_round_date({"last_round_amount": amount_fig(RAMP_Q, "2026-10-15")})
    assert (fig.field, fig.value, fig.status, fig.usable, fig.as_of) == ("last_round_date", "2026-10", "not_confirmed", True, "2026-10-15")
    assert "inferred" in fig.reason and fig.evidence_quote == RAMP_Q


@pytest.mark.parametrize("figures", [
    {"last_round_amount": amount_fig("Ramp raised $555 million in 2021 in a Series A round.", "2026-10-15")},   # another year
    {"last_round_amount": amount_fig("The $1.85 billion round values Ramp at $60 billion.", "2026-10-15")},     # no funding verb
    {"last_round_amount": amount_fig(RAMP_Q, "")},                                                              # page without a date
    {"last_round_amount": amount_fig(RAMP_Q, "2026")},                                                          # no month
    {"last_round_amount": amount_fig(RAMP_Q, "2026-10-15", status="not_confirmed")},                            # not verified
    {},
])
def test_no_inference_without_good_evidence(figures):
    assert infer_round_date(figures) is None


def test_a_date_the_model_gave_is_kept_and_an_unusable_one_is_replaced():
    good = Figure("last_round_date", "2026-06", "verified", True, "u", "q", "a.com", ["a.com"], "", None, "2026-06-20")
    assert infer_round_date({"last_round_amount": amount_fig(RAMP_Q, "2026-10-15"), "last_round_date": good}) is None
    bad = Figure("last_round_date", "2026-06", "not_confirmed", False, "u", "q", "a.com", [], "", None, "2021-01-01")
    assert infer_round_date({"last_round_amount": amount_fig(RAMP_Q, "2026-10-15"), "last_round_date": bad}).value == "2026-10"


def test_the_inferred_date_reaches_the_score_and_the_table():
    page = Page("https://techcrunch.com/2026/10/15/ramp", "Ramp raises $1.85B", RAMP_Q, "2026-10-15")
    f = Fact("last_round_amount", "$1.85 billion", RAMP_Q, page.url, page.title, page.date)
    figures = merge_all([(f, check_fact(f, page, "Ramp"))])
    assert figures["last_round_amount"].status == "verified" and figures["last_round_date"].value == "2026-10"
    rec = CompanyRecord("Ramp", figures, [], False)
    assert compute_signals([rec], TODAY)["Ramp"]["recent"] is not None
    assert cell(rec, "last_round_date", month, TODAY) == "2026-10?"
    assert compute_signals([rec], TODAY, use_unconfirmed=False)["Ramp"]["recent"] is None      # --strict leaves it out


# ------------------------------------------------------------------ traction words
@pytest.mark.parametrize("q,level", [
    ("The San Francisco-based company offers bank accounts integrated with various financial tools businesses need.", 0.0),
    ("MoonPay, which is profitable, says it helps businesses transact digital assets quickly and smoothly.", 0.5),
    ("It serves more than 300,000 accounts worldwide.", 1.0),
    ("Mercury counts more than 300,000 companies as customers.", 1.0),
    ("Ramp has 85,000 customers, up from 70,000 in June.", 1.0)])
def test_traction_needs_customers_or_revenue_not_product_words(q, level):
    assert traction_level(q) == level


def test_a_product_description_is_not_traction_evidence():
    q = "The San Francisco-based company offers bank accounts integrated with various financial tools businesses need."
    f = Fact("traction_evidence", "bank accounts", q, "https://a.com/x")
    fig = merge_field("traction_evidence", [(f, check_fact(f, Page("https://a.com/x", "Mercury profile", q), "Mercury"))])
    assert fig.usable is False
    assert compute_signals([CompanyRecord("Mercury", {"traction_evidence": fig})], TODAY)["Mercury"]["traction"] is None


# ------------------------------------------------------------------ the model reads the newest pages
def setup_four_pages(rig, monkeypatch):  # noqa: F811
    # one "domain" per URL, so pick_pages can choose all four (the fake server has only two real host names)
    monkeypatch.setattr("research.extract.registrable_domain", lambda u: u)
    keys = {"old": "2018/09/26/acme-old", "plain": "acme-plain", "mid": "2025/03/01/acme-mid", "new": "2026/05/20/acme-new"}
    for k in keys.values():
        monkeypatch.setitem(SITE, k, make_html(f"Acme Pay page {k}", "Acme Pay was founded in 2019 by two bankers."))
    urls = {name: f"{rig.a}/site/{key}" for name, key in keys.items()}
    base = Site.respond

    def respond(m, p, b):
        if p.startswith("/tavily/search") and b.get("topic") != "news":
            q = b["query"].lower()
            if "acme pay" in q and "top fintech" not in q:
                order = ["old", "plain", "mid", "new"]                      # the oldest page comes first in the search results
                return 200, {"results": [{"title": f"Acme Pay {n}", "url": urls[n], "content": "Acme Pay funding"} for n in order]}, "application/json"
        return base(m, p, b)
    Site.respond = respond
    return urls


def reading(rig):
    return [e["url"] for e in rig.events("research_company") if e["phase"] == "reading"]


def test_the_model_reads_the_two_newest_of_four_downloaded_pages(rig, monkeypatch):  # noqa: F811
    urls = setup_four_pages(rig, monkeypatch)
    research("FinTech", rig.web, rig.llm, rig.settings, rig.bus, max_candidates=1, with_news=False)
    fetching = [e["url"] for e in rig.events("research_company") if e["phase"] == "fetching" and "acme" in e["url"]]
    assert sorted(fetching) == sorted(urls.values())                       # all four were downloaded ...
    assert reading(rig) == [urls["new"], urls["mid"]]                      # ... but only the newest two went to the model
    assert [c for c in SmartLlama.calls if c[0] == "facts"] == [("facts", "Acme Pay")] * 2
    assert {e["page_date"] for e in rig.events("research_company") if e["phase"] == "reading"} == {"2026-05-20", "2025-03-01"}


def test_extra_pages_zero_reads_the_first_search_results_like_before(rig, monkeypatch):  # noqa: F811
    urls = setup_four_pages(rig, monkeypatch)
    research("FinTech", rig.web, rig.llm, rig.settings, rig.bus, max_candidates=1, with_news=False, extra_pages=0)
    # the old page is demoted, so the first two results are "plain" and "mid"; the dated one is read first
    assert reading(rig) == [urls["mid"], urls["plain"]]
    assert urls["new"] not in reading(rig) and urls["old"] not in reading(rig)


# ------------------------------------------------------------------ GDELT switched off
def test_gdelt_switched_off_sends_no_request_and_logs_nothing(rig):  # noqa: F811
    rig.web.disabled.add("gdelt")
    arts, ok, sources = news_mod.find_news_detail(rig.web, "Acme Pay", "FinTech")
    assert ok and sources == ["web"] and arts and {a["via"] for a in arts} == {"tavily"}
    assert not any(h[1].startswith("/gdelt/") for h in Site.hits)
    assert not [e for e in rig.events("fallback") if e.get("provider") == "gdelt"]


def test_the_command_line_has_the_extra_pages_switch():
    assert build_parser().parse_args(["FinTech", "--extra-pages", "4"]).extra_pages == 4
    d = build_parser().parse_args(["FinTech"])
    assert d.extra_pages == 3 and d.pages == 2