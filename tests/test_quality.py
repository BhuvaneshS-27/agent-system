"""Tests for Step 7b: the fixes after the first real FinTech run (old figures, round vs total, GDELT time-outs,
Tavily ignoring days=90, unreliable round dates, traction and founded-year rules, Coverage column).
Same style as test_research.py: fake servers, no internet, no real model."""
from datetime import date, timedelta

import pytest
import requests

from research import extract as ex
from research import news as news_mod
from research.errors import ProviderError
from research.pagedate import best_date, url_date
from research.pipeline import research
from research.scoring import (CompanyRecord, compute_signals, figure_is_stale, is_stale, news_in_window,
                              rank_signals)
from research.verify import Fact, Figure, Page, check_fact, merge_field, snap_quote, traction_level
from research.web import WebTools
from core.bus import EventBus
from core.cache import normalise_url
from test_research import FACTS, SITE, Site, iso, rig  # noqa: F401  (rig is a fixture)
from test_research import html as make_html

TODAY = date(2026, 10, 24)
URL = "https://news.example.com/a"


def fact(field, value, quote, url=URL, page_date=""):
    return Fact(field, value, quote, url, "", page_date)


# ------------------------------------------------------------------ page dates
def test_dates_in_urls():
    assert url_date("https://www.cnbc.com/2026/05/20/fintech-mercury.html", TODAY) == "2026-05-20"
    assert url_date("https://x.com/news/2021-11-22-some-story", TODAY) == "2021-11-22"
    assert url_date("https://x.com/2021/11/some-story", TODAY) == "2021-11"
    assert url_date("https://x.com/p/2021/13/01/story", TODAY) == ""            # month 13
    assert url_date("https://x.com/2099/01/01/story", TODAY) == ""              # in the future
    assert url_date("https://x.com/a/b", TODAY) == ""


def test_page_metadata_date_wins_over_the_url_and_rubbish_falls_back():
    assert best_date("2021-11-22T10:00:00+00:00", "https://x.com/2020/01/01/a", TODAY) == "2021-11-22"
    assert best_date("not a date", "https://x.com/2020/01/01/a", TODAY) == "2020-01-01"
    assert best_date("2099-01-01", "https://x.com/a", TODAY) == ""
    assert best_date("", "https://x.com/a", TODAY) == ""


def test_old_url_dates_are_read_last():
    res = [{"title": "Stripe valued at $20B", "url": "https://a.com/2018/09/26/stripe", "snippet": "Stripe"},
           {"title": "Stripe news", "url": "https://b.com/2026/05/20/stripe", "snippet": "Stripe"},
           {"title": "Stripe site", "url": "https://c.com/stripe", "snippet": "Stripe"}]
    assert [r["url"] for r in ex.pick_pages(res, "Stripe", 2, today=TODAY)] == ["https://b.com/2026/05/20/stripe", "https://c.com/stripe"]


def test_a_page_date_is_never_guessed_from_the_body_text(rig, monkeypatch):  # noqa: F811
    # trafilatura's default turned "founded in 2019" into the date 2019-01-01 of the page
    monkeypatch.setitem(SITE, "plain", make_html("Plain page", "Acme Pay was founded in 2019 by two bankers."))
    out = rig.web.fetch(f"{rig.a}/site/plain")
    assert out.ok and out.date == ""


def test_page_date_comes_from_real_metadata_or_from_the_url(rig, monkeypatch):  # noqa: F811
    body = make_html("Dated page", "Acme Pay was founded in 2019 by two bankers.")
    meta = body.replace("<head>", '<head><meta property="article:published_time" content="2026-05-20T10:00:00Z">')
    monkeypatch.setitem(SITE, "with-meta", meta)
    monkeypatch.setitem(SITE, "2021/11/22/in-url", body)
    assert rig.web.fetch(f"{rig.a}/site/with-meta").date == "2026-05-20"
    assert rig.web.fetch(f"{rig.a}/site/2021/11/22/in-url").date == "2021-11-22"
    assert rig.web.fetch(f"{rig.a}/site/2021/11/22/in-url").cached                 # the date is stored with the cached page


def test_pages_cached_before_step_7b_get_their_date_from_the_url(rig):  # noqa: F811
    url = "https://www.cnbc.com/2026/05/20/fintech-mercury-valuation.html"
    rig.web.cache.put("fetch_page", normalise_url(url), {"url": url, "title": "T", "text": "x" * 200}, "trafilatura")
    out = rig.web.fetch(url)
    assert out.ok and out.cached and out.date == "2026-05-20"
    undated = "https://example.com/profile"
    rig.web.cache.put("fetch_page", normalise_url(undated), {"url": undated, "title": "T", "text": "x" * 200}, "trafilatura")
    assert rig.web.fetch(undated).date == ""


# ------------------------------------------------------------------ copy slips in quotes
SENT = "Acme Pay raised $120 million in a Series C round led by Big Fund in March 2026."
PAGE = Page(URL, "Acme Pay raises", "Intro text about the company and its products.\n" + SENT +
            "\nMore text follows about other things entirely, such as offices.")


def test_a_copy_slip_is_corrected_to_the_real_sentence():
    quote = "Acme Pay raised $120 million in a Series C round led by Big Fund in March, 2026"       # comma, no full stop
    v = check_fact(fact("last_round_amount", "$120 million", quote), PAGE, "Acme Pay")
    assert v.status == "verified" and v.quote == SENT and "corrected" in v.note
    fig = merge_field("last_round_amount", [(fact("last_round_amount", "$120 million", quote), v)])
    assert fig.evidence_quote == SENT and "corrected" in fig.reason


def test_correcting_a_quote_cannot_rescue_a_changed_number():
    quote = SENT.replace("$120 million", "$1,200 million")
    assert snap_quote(quote, PAGE) == SENT                                                          # it IS close ...
    v = check_fact(fact("last_round_amount", "$1.2 billion", quote), PAGE, "Acme Pay")
    assert v.status == "not_confirmed" and v.fatal                                                  # ... but the number is not there


def test_an_unrelated_quote_is_not_corrected():
    q = "The company has offices in Berlin and Paris and a large team."
    assert snap_quote(q, PAGE) is None
    assert check_fact(fact("founded_year", "2019", q), PAGE, "Acme Pay").fatal


# ------------------------------------------------------------------ one round is not the total
def test_a_round_amount_is_not_accepted_as_the_total():
    t = "Acme Pay raised $245 million in a Series B round led by Big Fund."
    p = Page(URL, "Acme Pay", t)
    v = check_fact(fact("total_raised_usd", "$245 million", t), p, "Acme Pay")
    assert v.status == "not_confirmed" and not v.fatal and v.unusable and "TOTAL" in " ".join(v.reasons)
    assert check_fact(fact("last_round_amount", "$245 million", t), p, "Acme Pay").status == "verified"
    assert merge_field("total_raised_usd", [(fact("total_raised_usd", "$245 million", t), v)]).usable is False


@pytest.mark.parametrize("t", ["Acme Pay has raised $685 million in total funding to date.",
                               "So far, Acme Pay has raised $685 million.",
                               "Acme Pay's cumulative funding stands at $685 million.",
                               "Acme Pay has raised $685 million since its founding."])
def test_total_words_make_a_total_verified(t):
    v = check_fact(fact("total_raised_usd", "$685 million", t), Page(URL, "Acme Pay", t), "Acme Pay")
    assert v.status == "verified", v.reasons


# ------------------------------------------------------------------ round dates
ROUND = "Acme Pay raised $60 million in a Series B round to expand across Europe."


def page_on(day, text=ROUND):
    return Page(URL, "Acme Pay raises $60 million", text, day)


def test_round_date_may_come_from_the_page_date_when_the_quote_has_none():
    v = check_fact(fact("last_round_date", "2021-11", ROUND), page_on("2021-11-22"), "Acme Pay")
    assert v.status == "verified" and v.parsed == "2021-11" and "publication date" in v.note
    assert check_fact(fact("last_round_date", "2021", ROUND), page_on("2021-11-22"), "Acme Pay").status == "verified"


@pytest.mark.parametrize("value,day,text", [
    ("2021-11", "2018-09-26", ROUND),                                    # the page is from another year
    ("2021-12", "2021-11-22", ROUND),                                    # the page is from another month
    ("2021-11", "", ROUND),                                              # the page has no date
    ("2021-11", "2021-11-22", "Acme Pay opened a new office in Madrid for its growing team."),   # nothing was raised
])
def test_round_date_fallback_refuses_when_the_page_does_not_support_it(value, day, text):
    v = check_fact(fact("last_round_date", value, text), page_on(day, text), "Acme Pay")
    assert v.status == "not_confirmed" and v.fatal


# ------------------------------------------------------------------ newest page wins
def checked(value, quote, url, page_date, field="total_raised_usd"):
    f = fact(field, value, quote, url, page_date)
    return f, check_fact(f, Page(url, "Acme Pay", quote, page_date), "Acme Pay")


def test_the_newest_verified_total_wins_and_its_age_is_recorded():
    old = checked("$685 million", "Acme Pay has raised $685 million in total funding to date.", "https://a.com/2018/x", "2018-09-27")
    new = checked("$2 billion", "Acme Pay has raised $2 billion in total funding to date.", "https://b.org/x", "2026-05-20")
    for order in ([old, new], [new, old]):
        fig = merge_field("total_raised_usd", order)
        assert fig.value == 2_000_000_000 and fig.as_of == "2026-05-20" and fig.status == "verified"
        assert "newest" in fig.reason and fig.domains == ["b.org"]


def test_a_page_with_a_date_beats_a_page_without_one():
    undated = checked("$100 million", "Acme Pay has raised $100 million in total.", "https://a.com/x", "")
    dated = checked("$300 million", "Acme Pay has raised $300 million in total.", "https://b.org/x", "2025-02-01")
    assert merge_field("total_raised_usd", [undated, dated]).value == 300_000_000


def test_newest_wins_but_agreement_still_makes_it_high():
    old = checked("$200 million", "Acme Pay has raised $200 million in total.", "https://a.com/x", "2024-01-01")
    new = checked("$210 million", "Acme Pay has raised $210 million in total.", "https://b.org/x", "2026-01-01")
    fig = merge_field("total_raised_usd", [old, new])
    assert fig.value == 210_000_000 and fig.status == "high" and fig.as_of == "2026-01-01"


def test_only_figures_that_can_go_out_of_date_are_marked_stale():
    def f(name):
        return Figure(name, 1, "verified", True, "u", "q", "d.com", ["d.com"], "", None, "2018-09-27")
    assert figure_is_stale(f("total_raised_usd"), TODAY) and figure_is_stale(f("valuation_usd"), TODAY)
    assert figure_is_stale(f("traction_evidence"), TODAY) and figure_is_stale(f("last_round_date"), TODAY)
    assert not figure_is_stale(f("founded_year"), TODAY) and not figure_is_stale(None, TODAY)


def test_stale_means_older_than_18_months():
    assert is_stale("2018-09-27", TODAY) and is_stale("2025-03", TODAY)
    assert not is_stale("2025-06", TODAY) and not is_stale("2026-05-20", TODAY)
    assert not is_stale("", TODAY) and not is_stale("junk", TODAY)


# ------------------------------------------------------------------ rules that keep a wrong value out of the score
def fig_of(field, value, quote, company="Alloy"):
    f = fact(field, value, quote)
    return merge_field(field, [(f, check_fact(f, Page(URL, company, quote), company))])


def test_a_founding_year_without_a_founded_word_is_kept_for_display_only():
    fig = fig_of("founded_year", "2020", "Alloy offers identity tools to banks in 2020.")
    assert fig.status == "not_confirmed" and fig.usable is False and fig.value == 2020
    sig = compute_signals([CompanyRecord("Alloy", {"founded_year": fig})], TODAY)
    assert sig["Alloy"]["maturity"] is None
    ok = fig_of("founded_year", "2015", "Alloy was founded in 2015 in New York.")
    assert ok.status == "verified" and ok.usable


def test_traction_with_no_customers_or_revenue_is_missing_not_zero():
    fig = fig_of("traction_evidence", "25 countries", "Alloy operates in 25 countries across the world.")
    assert fig.level == 0.0 and fig.usable is False
    assert compute_signals([CompanyRecord("Alloy", {"traction_evidence": fig})], TODAY)["Alloy"]["traction"] is None


@pytest.mark.parametrize("q,level", [("Tide has 1.6 million members worldwide.", 1.0),
                                     ("The bank now counts 300,000 cardholders.", 1.0),
                                     ("Mercury serves more than 300,000 businesses.", 1.0),
                                     ("It signed several new partners.", 0.5),
                                     ("It operates in 25 countries.", 0.0)])
def test_traction_words(q, level):
    assert traction_level(q) == level


# ------------------------------------------------------------------ GDELT: short wait, circuit breaker
def test_gdelt_waits_at_most_12_seconds_per_attempt(tmp_path, monkeypatch):
    seen = []

    def fake_get(url, **kw):
        seen.append(kw["timeout"])
        raise requests.Timeout()
    monkeypatch.setattr(news_mod.requests, "get", fake_get)
    out = WebTools(None, tmp_path, env={}, sleep=lambda s: None).search("q", providers=news_mod.GDELT_CHAIN, topic="news")
    assert not out.ok and seen == [(5, 12)] * 3 and news_mod.gdelt_news.breaker == 2


def test_circuit_breaker_skips_a_provider_that_keeps_failing_until_reset(tmp_path):
    calls = []

    def flaky(w, q):
        calls.append(q)
        raise ProviderError("timeout", True)
    flaky.breaker = 2

    def good(w, q):
        return [{"title": "t", "url": "https://x.com/a", "snippet": "s"}]
    bus = EventBus(tmp_path / "e.jsonl")
    web = WebTools(bus, tmp_path, env={}, sleep=lambda s: None)
    chain = [("flaky", flaky), ("good", good)]
    for q in ("a", "b", "c"):
        assert web.search(q, providers=chain, topic="news").ok
    assert calls == ["a"] * 3 + ["b"] * 3                                    # 3 tries each; the third search never tried it
    falls = [e for e in bus.read_since(0)[0] if e["kind"] == "fallback" and e["provider"] == "flaky"]
    assert falls[-1]["skipped"] is True and "circuit open" in falls[-1]["reason"] and falls[-1]["next"] == "good"
    web.reset_breakers()
    web.search("d", providers=chain, topic="news")
    assert calls[-3:] == ["d"] * 3


def test_a_provider_without_a_breaker_is_never_skipped(tmp_path):
    calls = []

    def flaky(w, q):
        calls.append(q)
        raise ProviderError("timeout", True)
    web = WebTools(None, tmp_path, env={}, sleep=lambda s: None)
    for q in ("a", "b", "c"):
        web.search(q, providers=[("flaky", flaky)], topic="news")
    assert len(calls) == 9


def test_every_run_starts_with_the_breaker_closed(rig):  # noqa: F811
    rig.web._fails["gdelt"] = 9
    research("FinTech", rig.web, rig.llm, rig.settings, rig.bus, max_candidates=1)
    assert any(h[1].startswith("/gdelt/") for h in Site.hits)


# ------------------------------------------------------------------ Tavily news window
def test_tavily_start_date_is_a_hint_that_is_dropped_once_if_tavily_rejects_it(rig):  # noqa: F811
    ok_route = Site.respond
    Site.respond = lambda m, p, b: ((400, "bad request", "text/plain")
                                    if p.startswith("/tavily/search") and b.get("topic") == "news" and "start_date" in b
                                    else ok_route(m, p, b))
    out = rig.web.search("Acme Pay FinTech startup news", topic="news", providers=news_mod.TAVILY_CHAIN)
    posts = [b for m, p, b in Site.hits if p.startswith("/tavily/search")]
    assert out.ok and len(posts) == 2 and "start_date" in posts[0] and "start_date" not in posts[1]
    assert posts[0]["start_date"] == (date.today() - timedelta(days=90)).isoformat() and posts[1]["days"] == 90
    assert rig.web.usage.month("tavily")["credits"] == 1                      # only the answered request counted


def test_articles_older_than_90_days_or_without_a_date_are_dropped_by_us(rig):  # noqa: F811
    ok_route = Site.respond

    def respond(m, p, b):
        if p.startswith("/tavily/search") and b.get("topic") == "news":
            return 200, {"results": [
                {"title": "Acme Pay: old story", "url": "https://old.example.org/1", "content": "x", "published_date": iso(200)},
                {"title": "Acme Pay: fresh story", "url": "https://new.example.org/1", "content": "x", "published_date": iso(10)},
                {"title": "Acme Pay: undated story", "url": "https://nodate.example.org/1", "content": "x"}]}, "application/json"
        return ok_route(m, p, b)
    Site.respond = respond
    arts, ok, sources = news_mod.find_news_detail(rig.web, "Acme Pay", "FinTech")
    urls = {a["url"] for a in arts}
    assert ok and sources == ["gdelt", "web"] and len(arts) == 4              # 3 from GDELT + 1 fresh Tavily story
    assert "https://new.example.org/1" in urls and not {"https://old.example.org/1", "https://nodate.example.org/1"} & urls


def test_find_news_still_returns_the_old_pair(rig):  # noqa: F811
    arts, ok = news_mod.find_news(rig.web, "Acme Pay", "FinTech")
    assert ok and len(arts) == 5


# ------------------------------------------------------------------ compare companies on the same news sources
def art(domain, days_ago, via):
    return {"date": (TODAY - timedelta(days=days_ago)).isoformat(), "title": "t", "url": f"https://{domain}/a",
            "domain": domain, "via": via}


def test_gdelt_articles_are_ignored_for_everybody_unless_every_company_has_gdelt():
    a = CompanyRecord("A", {}, [art(f"g{i}.com", 3, "gdelt") for i in range(6)] + [art("w.com", 3, "tavily")], True, [], [],
                      ["gdelt", "web"])
    b = CompanyRecord("B", {}, [art("w2.com", 3, "tavily")], True, [], [], ["web"])
    sig = compute_signals([a, b], TODAY)
    assert sig["A"]["news"] == sig["B"]["news"]                                # same single web article each
    both = CompanyRecord("B", {}, [art("w2.com", 3, "tavily"), art("g9.com", 3, "gdelt")], True, [], [], ["gdelt", "web"])
    sig = compute_signals([a, both], TODAY)
    assert sig["A"]["news"] > sig["B"]["news"]                                 # now GDELT counts for both


def test_records_from_older_runs_still_load():
    old = {"name": "X", "figures": {"founded_year": {"field": "founded_year", "value": 2019, "status": "verified",
                                                     "usable": True, "source_url": "u", "evidence_quote": "q",
                                                     "domain": "d.com", "domains": ["d.com"]}},
           "news": [], "news_ok": True, "urls": [], "notes": []}
    rec = CompanyRecord.from_dict(old)
    assert rec.news_sources == [] and rec.figures["founded_year"].as_of == ""
    assert CompanyRecord.from_dict(rec.to_dict()).news_sources == []


def test_news_in_window_counts_only_the_last_90_days():
    news = [art("a.com", 10, "x"), art("b.com", 91, "x"), art("c.com", 90, "x"), {"date": "", "url": "u", "domain": "d"}]
    assert news_in_window(news, TODAY) == 2


# ------------------------------------------------------------------ the whole pipeline keeps the age of each figure
def test_pipeline_prefers_the_newest_page_and_shows_stale_figures(rig, monkeypatch, capsys):  # noqa: F811
    old_q = "Acme Pay has raised $100 million in total funding to date."
    new_q = "Acme Pay has raised $300 million in total funding to date."
    founded = "Acme Pay was founded in 2019 by two bankers."
    monkeypatch.setitem(SITE, "2021/11/22/acme-old", make_html("Acme Pay profile", old_q, founded))
    monkeypatch.setitem(SITE, "2026/05/20/acme-new", make_html("Acme Pay at a glance", new_q))
    monkeypatch.setitem(FACTS, ("Acme Pay", "acme-old"), [("total_raised_usd", "$100 million", old_q), ("founded_year", "2019", founded)])
    monkeypatch.setitem(FACTS, ("Acme Pay", "acme-new"), [("total_raised_usd", "$300 million", new_q)])
    base = Site.respond

    def respond(m, p, b):
        if p.startswith("/tavily/search") and b.get("topic") != "news":
            q = b["query"].lower()
            if "acme pay" in q and "top fintech" not in q:
                return 200, {"results": [
                    {"title": "Acme Pay raises money", "url": f"{rig.a}/site/2021/11/22/acme-old", "content": "Acme Pay funding"},
                    {"title": "Acme Pay valuation", "url": f"{rig.b}/site/2026/05/20/acme-new", "content": "Acme Pay valuation"}]}, "application/json"
        return base(m, p, b)
    Site.respond = respond
    run = research("FinTech", rig.web, rig.llm, rig.settings, rig.bus, max_candidates=1, with_news=False)
    rec = run.records[0]
    total, year = rec.figures["total_raised_usd"], rec.figures["founded_year"]
    assert total.value == 300_000_000 and total.as_of == "2026-05-20" and "newest" in total.reason
    assert year.value == 2019 and year.as_of == "2021-11-22"
    assert any(e["as_of"] == "2026-05-20" for e in rig.events("verified") if e["field"] == "total_raised_usd")
    assert {e["page_date"] for e in rig.events("extracted")} == {"2021-11-22", "2026-05-20"}
    assert is_stale(year.as_of, TODAY) and not is_stale(total.as_of, TODAY)           # 2021 page, 2026 page
    assert not figure_is_stale(year, TODAY)                                             # but a founding year never goes stale

    from scripts.run_research import cell, money, print_detail, print_table
    assert cell(rec, "founded_year", str, TODAY) == "2019" and cell(rec, "total_raised_usd", money, TODAY) == "$300M"
    rec.figures["total_raised_usd"].as_of = "2018-09-27"                                # now pretend the total is from 2018
    assert cell(rec, "total_raised_usd", money, TODAY) == "$300M!"
    rows = rank_signals([rec], compute_signals([rec], TODAY), "Established")
    print_table(rows, "Established", 5, TODAY)
    print_detail(rows, 5, TODAY)
    out = capsys.readouterr().out
    assert " Cov " in out and "$300M!" in out and "STALE" in out and "source 2018-09-27" in out and new_q[:40] in out


# ------------------------------------------------------------------ extraction keeps the page date
def test_facts_carry_the_date_of_their_page(rig, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ex, "ask_json", lambda *a, **k: {"facts": [
        {"field": "founded_year", "value": "2019", "evidence_quote": "Acme Pay was founded in 2019."}]})
    page = Page("http://x.test/a", "Acme", "text " * 10, "2026-05-20")
    facts = ex.extract_facts(rig.llm, rig.settings, rig.web, rig.bus, "Acme Pay", page, fresh=True)
    assert facts[0].page_date == "2026-05-20"
    assert "TOTAL" not in ex.FACTS_SYSTEM and "total_raised_usd is only for" in ex.FACTS_SYSTEM
    assert "NOT countries" in ex.FACTS_SYSTEM


def test_figure_with_an_age_survives_save_and_load():
    f = Figure("founded_year", 2019, "verified", True, "u", "q", "d.com", ["d.com"], "", None, "2021-11-22")
    assert Figure.from_dict(f.to_dict()) == f