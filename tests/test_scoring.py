"""Tests for research/scoring.py. Fake companies only: the maths is proved without any API or model."""
import math
from datetime import date, timedelta

import pytest

from research.scoring import (PRESETS, CompanyRecord, compute_signals, confidence, confidence_label, decay, minmax,
                              news_raw, rank, rank_signals, round_date, score)
from research.verify import Figure

TODAY = date(2026, 10, 23)


def fig(field, value, status="verified", level=None, domains=("a.com",), usable=True):
    return Figure(field, value, status, usable, f"https://{domains[0]}/x", "a quote", domains[0], list(domains), "", level)


def arts(days_ago_by_domain):
    return [{"date": (TODAY - timedelta(days=d)).isoformat(), "title": "t", "url": f"https://{dom}/a", "domain": dom,
             "via": "gdelt"} for dom, d in days_ago_by_domain]


def company(name, raised=None, val=None, rdate=None, founded=None, traction=None, news=None, news_ok=True, status="verified"):
    f = {}
    if raised is not None:
        f["total_raised_usd"] = fig("total_raised_usd", raised, status)
    if val is not None:
        f["valuation_usd"] = fig("valuation_usd", val, status)
    if rdate is not None:
        f["last_round_date"] = fig("last_round_date", rdate, status)
    if founded is not None:
        f["founded_year"] = fig("founded_year", founded, status)
    if traction is not None:
        f["traction_evidence"] = fig("traction_evidence", "customers", status, level=traction)
    return CompanyRecord(name, f, news or [], news_ok)


GIANT = company("Giant", 5e9, 5e10, "2023-06", 2010, 1.0, arts([("a.com", 80)]))
ROCKET = company("Rocket", 2e8, 1e9, "2026-09", 2021, 1.0, arts([(f"d{i}.com", i) for i in range(10)]))
BABY = company("Baby", 3e6, None, "2026-10", 2025, 0.5, arts([("x.com", 3), ("y.com", 5)]))
POOL = [GIANT, ROCKET, BABY]


# ------------------------------------------------------------------ the building blocks
def test_decay_halves_at_the_half_life():
    assert decay(0, 30) == 1 and decay(30, 30) == 0.5 and decay(60, 30) == 0.25


def test_minmax():
    assert minmax([1, 2, 3]) == [0, 0.5, 1]
    assert minmax([7, 7]) == [0.5, 0.5]


def test_score_hand_calculated():
    s = {"funding": 1, "valuation": 1, "recent": 0, "news": 0, "traction": 0.5, "maturity": 1}
    v, cov = score(s, "Established")
    assert v == pytest.approx(82.5) and cov == pytest.approx(1.0)
    v, _ = score(s, "Early-stage")                          # maturity is inverted: 1 - 1 = 0
    assert v == pytest.approx(100 * (0.10 * 1 + 0.05 * 1 + 0.15 * 0.5))


def test_missing_signals_are_renormalised_not_punished():
    v, cov = score({"funding": 1, "traction": 0.5, "valuation": None, "recent": None, "news": None, "maturity": None},
                   "Established")
    assert v == pytest.approx(100 * (0.30 + 0.15 * 0.5) / 0.45)
    assert cov == pytest.approx(0.45)
    assert score({k: None for k in PRESETS["Established"]}, "Established") == (0.0, 0.0)


def test_every_preset_sums_to_one():
    for w in PRESETS.values():
        assert sum(w.values()) == pytest.approx(1.0)


# ------------------------------------------------------------------ signals
def test_pool_scaling_of_funding_valuation_news():
    sig = compute_signals(POOL, TODAY)
    assert sig["Giant"]["funding"] == pytest.approx(1.0) and sig["Baby"]["funding"] == pytest.approx(0.0)
    expected = (math.log10(2e8) - math.log10(3e6)) / (math.log10(5e9) - math.log10(3e6))
    assert sig["Rocket"]["funding"] == pytest.approx(expected)
    assert sig["Giant"]["valuation"] == 1.0 and sig["Rocket"]["valuation"] == 0.0 and sig["Baby"]["valuation"] is None
    assert sig["Rocket"]["news"] == 1.0 and sig["Giant"]["news"] == 0.0


def test_recent_funding_uses_a_12_month_half_life():
    c = company("C", rdate="2025-10-23")                       # exactly 12 months before TODAY (365 days)
    sig = compute_signals([c], TODAY)["C"]["recent"]
    assert sig == pytest.approx(2 ** (-(365 / 30.4375) / 12), rel=1e-6) and 0.49 < sig < 0.51
    assert compute_signals([company("D", rdate="2026-10-23")], TODAY)["D"]["recent"] == pytest.approx(1.0)


def test_future_round_date_counts_as_today_and_year_only_means_july():
    assert compute_signals([company("D", rdate="2027-01")], TODAY)["D"]["recent"] == pytest.approx(1.0)
    assert round_date("2026") == date(2026, 7, 1) and round_date("2026-03") == date(2026, 3, 1)
    assert round_date("2026-03-15") == date(2026, 3, 15) and round_date("junk") is None


def test_maturity_is_age_capped_at_15_years_over_15():
    sig = compute_signals([company("Old", founded=1990), company("Mid", founded=2021), company("New", founded=2026)], TODAY)
    assert sig["Old"]["maturity"] == 1.0
    assert sig["Mid"]["maturity"] == pytest.approx(5 / 15)
    assert sig["New"]["maturity"] == 0.0


def test_news_counts_one_article_per_domain_in_90_days_with_a_30_day_half_life():
    items = arts([("a.com", 30), ("a.com", 10), ("b.com", 91), ("c.com", 0)])
    items.append({"date": "", "title": "no date", "url": "https://d.com/x", "domain": "d.com"})
    items.append({"date": (TODAY + timedelta(days=5)).isoformat(), "title": "future", "url": "https://e.com/x", "domain": "e.com"})
    # a.com: newest is 10 days; b.com: older than 90 days (ignored); c.com: today; d.com: no date (ignored); e.com: future = age 0
    assert news_raw(items, TODAY) == pytest.approx(decay(10, 30) + 1.0 + 1.0)


def test_failed_news_search_is_no_data_but_zero_articles_is_zero():
    sig = compute_signals([company("A", news=[], news_ok=False), company("B", news=[], news_ok=True),
                           company("C", news=arts([("x.com", 1)]))], TODAY)
    assert sig["A"]["news"] is None and sig["B"]["news"] == 0.0 and sig["C"]["news"] == 1.0


def test_traction_comes_from_the_rule_level():
    sig = compute_signals([company("A", traction=1.0), company("B", traction=0.5), company("C", traction=0.0)], TODAY)
    assert [sig[n]["traction"] for n in "ABC"] == [1.0, 0.5, 0.0]


# ------------------------------------------------------------------ presets reorder the table
def test_ranking_changes_between_presets():
    assert [r.name for r in rank(POOL, "Established", TODAY)][0] == "Giant"
    assert [r.name for r in rank(POOL, "Fastest rising", TODAY)][0] == "Rocket"
    early = [r.name for r in rank(POOL, "Early-stage", TODAY)]
    assert early[0] == "Rocket" and early.index("Baby") < early.index("Giant")


def test_switching_preset_on_stored_signals_equals_a_full_recompute():
    sig = compute_signals(POOL, TODAY)
    for preset in PRESETS:
        a = rank_signals(POOL, sig, preset)
        b = rank(POOL, preset, TODAY)
        assert [(r.name, round(r.score, 6)) for r in a] == [(r.name, round(r.score, 6)) for r in b]


def test_ranks_are_numbered_and_ties_sorted_by_confidence_then_name():
    twins = [company("Zed", raised=1e6, founded=2020), company("Amy", raised=1e6, founded=2020)]
    rows = rank(twins, "Established", TODAY)
    assert [r.rank for r in rows] == [1, 2] and [r.name for r in rows] == ["Amy", "Zed"]


# ------------------------------------------------------------------ confidence
def test_full_coverage_all_verified_three_domains_is_confidence_one():
    f = {n: fig(n, v, domains=("a.com", "b.com", "c.com"), level=1.0 if n == "traction_evidence" else None)
         for n, v in (("total_raised_usd", 1e8), ("valuation_usd", 1e9), ("last_round_date", "2026-05"),
                      ("founded_year", 2018), ("traction_evidence", "users"))}
    c = CompanyRecord("Full", f, arts([("x.com", 4)]), True)
    sig = compute_signals([c], TODAY)["Full"]
    assert confidence(c, sig, "Established") == pytest.approx(1.0)


def test_missing_data_and_a_not_confirmed_figure_lower_the_confidence():
    c = company("Part", raised=1e8, rdate="2026-05", founded=2018, news=[], news_ok=True)
    c.figures["founded_year"] = fig("founded_year", 2018, "not_confirmed")           # used, because nothing better exists
    sig = compute_signals([c], TODAY)["Part"]
    assert {k for k, v in sig.items() if v is not None} == {"funding", "recent", "news", "maturity"}
    coverage = 0.30 + 0.05 + 0.05 + 0.15
    expected = 0.5 * coverage + 0.3 * (2 / 3) + 0.2 * (1 / 3)
    assert confidence(c, sig, "Established") == pytest.approx(expected)
    assert confidence_label(expected) == "Medium"


def test_strict_mode_drops_not_confirmed_figures_from_the_score():
    c = company("Part", raised=1e8, founded=2018, status="not_confirmed")
    assert compute_signals([c], TODAY, use_unconfirmed=True)["Part"]["maturity"] is not None
    strict = compute_signals([c], TODAY, use_unconfirmed=False)["Part"]
    assert strict["maturity"] is None and strict["funding"] is None


def test_a_fatal_figure_is_never_used():
    c = CompanyRecord("Fake", {"total_raised_usd": fig("total_raised_usd", 9e12, "not_confirmed", usable=False)}, [], False)
    assert compute_signals([c], TODAY)["Fake"]["funding"] is None
    row = rank([c], "Established", TODAY)[0]
    assert row.score == 0.0 and row.coverage == 0.0 and row.label == "Low"


def test_confidence_labels():
    assert [confidence_label(x) for x in (0.9, 0.75, 0.6, 0.5, 0.49, 0.0)] == ["High", "High", "Medium", "Medium", "Low", "Low"]


def test_record_survives_save_and_load():
    again = CompanyRecord.from_dict(ROCKET.to_dict())
    assert again.name == "Rocket" and again.figures["total_raised_usd"] == ROCKET.figures["total_raised_usd"]
    assert again.news == ROCKET.news and again.news_ok is True