"""Tests for Step 7c: round dates that contradict their page, 'total' wording, connect time-out labels,
stale figures in the confidence, the CLI timing and table. (The IPv4 experiment of 7c was removed in 7d.)"""
from datetime import date

import pytest
import requests

from research import extract as ex
from research.errors import request_problem
from research.scoring import CompanyRecord, compute_signals, confidence, rank, rank_signals
from research.verify import Fact, Figure, Page, check_fact, date_contradiction, merge_field
from scripts.run_research import TABLE_HEAD, cell, event_lines, money, month, table_row

TODAY = date(2026, 10, 24)
URL = "https://tracxn.com/d/companies/moonpay"


# ------------------------------------------------------------------ error labels
def test_connect_and_read_timeouts_have_their_own_labels():
    assert request_problem(requests.ConnectTimeout()).reason == "timeout: could not connect in time"
    assert request_problem(requests.ReadTimeout()).reason == "timeout"
    assert request_problem(requests.Timeout()).reason == "timeout"
    assert request_problem(requests.ConnectTimeout()).retryable


# ------------------------------------------------------------------ a round date must fit its own page
def round_check(value, quote, page_date):
    f = Fact("last_round_date", value, quote, URL, "", page_date)
    return check_fact(f, Page(URL, "MoonPay profile", quote, page_date), "MoonPay")


SERIES_B = "MoonPay's latest funding round was a Series B round on Jun 12, 2026 for $100M."


def test_a_round_date_later_than_its_own_page_is_not_scored():
    v = round_check("2026-06-12", SERIES_B, "2021-10-14")                  # a 2021 page cannot know about June 2026
    assert v.status == "not_confirmed" and not v.fatal and v.unusable and "page cannot know" in " ".join(v.reasons)


def test_the_same_date_on_a_current_page_is_fine():
    assert round_check("2026-06-12", SERIES_B, "2026-06-20").status == "verified"
    assert round_check("2026-06-12", SERIES_B, "").status == "verified"    # no page date: nothing to compare with


def test_a_round_date_in_the_future_is_fatal():
    v = round_check("2099-01", "MoonPay raised money in January 2099 in a Series B round.", "")
    assert v.fatal and "future" in " ".join(v.reasons)


def test_date_contradiction_edges():
    assert date_contradiction(2026, 3, 15, "2026-03", TODAY) == ""            # same month
    assert date_contradiction(2026, 5, None, "2026-03", TODAY) == ""          # inside the 31 days of tolerance
    assert "cannot know" in date_contradiction(2026, 6, None, "2026-03", TODAY)
    assert date_contradiction(2026, 11, None, "", TODAY) == "future"
    assert date_contradiction(2026, 10, None, "", TODAY) == ""                # October has started
    assert date_contradiction(2025, 2, 30, "2025-02", TODAY) == ""            # impossible day: not our business


def test_the_moonpay_case_no_longer_drives_the_ranking():
    f = Fact("last_round_date", "2026-06-12", SERIES_B, URL, "", "2021-10-14")
    fig = merge_field("last_round_date", [(f, round_check("2026-06-12", SERIES_B, "2021-10-14"))])
    rec = CompanyRecord("MoonPay", {"last_round_date": fig}, [], False)
    assert fig.usable is False and compute_signals([rec], TODAY)["MoonPay"]["recent"] is None
    assert cell(rec, "last_round_date", month, TODAY) == "rejected"


# ------------------------------------------------------------------ more ways to say "total"
@pytest.mark.parametrize("text,amount,company", [
    ("The payments company now has $685 million in known capital raised.", "$685 million", "Stripe"),
    ("Founded in 2015, Alloy has now raised just over $200 million.", "$200 million", "Alloy")])
def test_known_capital_and_has_now_raised_count_as_a_total(text, amount, company):
    v = check_fact(Fact("total_raised_usd", amount, text, "https://a.com/x"), Page("https://a.com/x", f"{company} profile", text), company)
    assert v.status == "verified", v.reasons


def test_the_prompt_asks_for_one_entry_per_field_and_the_cache_key_changed():
    assert "One sentence can support several fields" in ex.FACTS_SYSTEM and ex.PROMPT_VERSION == "v3"


# ------------------------------------------------------------------ stale figures in the confidence
def fig(field, value, as_of):
    return Figure(field, value, "verified", True, "u", "q", "a.com", ["a.com"], "", None, as_of)


FRESH = CompanyRecord("F", {"total_raised_usd": fig("total_raised_usd", 1e8, "2026-05-20"),
                            "valuation_usd": fig("valuation_usd", 1e9, "2026-05-20")}, [], False)
MIXED = CompanyRecord("M", {"total_raised_usd": fig("total_raised_usd", 1e8, "2026-05-20"),
                            "valuation_usd": fig("valuation_usd", 1e9, "2018-09-27"),
                            "founded_year": fig("founded_year", 2010, "2018-09-27")}, [], False)


def test_a_stale_figure_still_scores_but_does_not_count_as_verified_in_the_confidence():
    sig = compute_signals([FRESH, MIXED], TODAY)
    assert sig["M"]["valuation"] is not None
    only = CompanyRecord("M", {k: v for k, v in MIXED.figures.items() if k != "founded_year"}, [], False)
    s2 = compute_signals([FRESH, only], TODAY)
    diff = confidence(FRESH, s2["F"], "Established", today=TODAY) - confidence(only, s2["M"], "Established", today=TODAY)
    assert diff == pytest.approx(0.3 * 0.5)                                   # verified share 2/2 against 1/2


def test_drop_stale_leaves_old_figures_out_but_a_founding_year_stays():
    sig = compute_signals([FRESH, MIXED], TODAY, use_stale=False)
    assert sig["M"]["valuation"] is None and sig["M"]["funding"] is not None and sig["M"]["maturity"] is not None
    rows = {r.name: r for r in rank_signals([FRESH, MIXED], sig, "Established", True, TODAY, False)}
    assert rows["F"].coverage == pytest.approx(0.30 + 0.30) and rows["M"].coverage == pytest.approx(0.30 + 0.15)
    kept = {r.name: r.coverage for r in rank([FRESH, MIXED], "Established", TODAY, True, False)}
    assert kept == {"F": pytest.approx(0.60), "M": pytest.approx(0.45)}


# ------------------------------------------------------------------ the CLI
def test_the_time_in_front_of_a_line_counts_from_the_last_printed_line():
    events = [{"kind": "research_company", "t": 100.0, "name": "Acme", "phase": "reading", "url": "https://a.com/x"},
              {"kind": "llm_call", "t": 160.0},                                       # not printed
              {"kind": "extracted", "t": 161.0, "url": "https://a.com/x", "page_date": "2026-05-20", "facts": 3, "verified": 2}]
    lines, last = event_lines(events, None)
    assert len(lines) == 2 and lines[0].startswith("       ") and lines[1].startswith("( 61s)") and last == 161.0
    more, _ = event_lines([{"kind": "verified", "t": 161.4, "field": "total_raised_usd", "value": 1, "status": "verified"}], last)
    assert more[0].startswith("       ")                                             # less than 1 s: no number


def test_the_table_fits_a_120_column_console_and_the_columns_line_up():
    def f(field, value, status="high", as_of="2026-05-20", level=None):
        return Figure(field, value, status, True, "u", "q", "a.com", ["a.com", "b.org"], "", level, as_of)
    rec = CompanyRecord("Plaid Technologies Incorporated", {
        "total_raised_usd": f("total_raised_usd", 7e8), "valuation_usd": f("valuation_usd", 5.2e9, as_of="2021-01-01"),
        "last_round_date": f("last_round_date", "2025-09-22", as_of="2021-01-01"), "founded_year": f("founded_year", 2017),
        "traction_evidence": f("traction_evidence", "x", level=1.0)}, [], True)
    rows = rank([rec, CompanyRecord("B", {}, [], False)], "Established", TODAY)
    head = TABLE_HEAD
    for r in rows:
        line = table_row(r, TODAY)
        assert len(line) <= 118 and line.index("|") == head.index("|")
    assert len(head) <= 118
    assert "2025-09++!" in table_row(rows[0], TODAY) and "$5.2B++!" in table_row(rows[0], TODAY)
    assert month("2026-05-20") == "2026-05" and money(5.2e9) == "$5.2B"