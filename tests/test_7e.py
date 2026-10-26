"""Tests for Step 7e: page dates from table rows only (not from sidebars), a date with no funding word is not
scored, a 'total' that is really one round becomes a round, two searches feed the page pool, traction must be
about customers, Jina's 403, GDELT opt-in. Fake servers only."""
from datetime import date
from types import SimpleNamespace

import pytest

from research import fetch as fetch_mod
from research.errors import ProviderError
from research.pagedate import page_date, table_row_date
from research.pipeline import research
from research.scoring import CompanyRecord, compute_signals
from research.verify import Fact, Page, check_fact, merge_all, merge_field, reclassify_rounds
from scripts.run_research import build_parser
from test_research import SITE, Site, rig  # noqa: F401  (rig is a fixture)
from test_research import html as make_html

TODAY = date(2026, 10, 24)
URL = "https://tracxn.com/d/companies/stripe"
ROW = "Apr 08, 2024 | $694M | Series I | $65B | - | - | -"

# What fintechfutures.com showed around a June 2026 article: lists of "latest news" with today's date,
# headlines with dollar amounts, and a webinar line that also uses '|' separators.
CHROME = ("Recent in FinTech\nLSEG to acquire Octus in $3.5bn all-cash deal\nOct 23, 2026\n"
          "Webinar | 18 October 2026 | Quantum computing in financial services\n"
          "Prism Data lands $20m Series A\nOct 21, 2026\nWebinar | 24 November 2026 | AI ROI in insurance")


# ------------------------------------------------------------------ dates in the page text
def test_a_table_row_date_can_lift_the_metadata_date_by_up_to_12_months():
    assert page_date("2023-09-26T10:00:00Z", URL, ROW, TODAY) == "2024-04-08"              # Tracxn: stale metadata
    assert page_date("2023-09-26", URL, "Round | Oct 17, 2023 | $40M | Series B", TODAY) == "2023-10-17"
    assert page_date("2021-10-14", URL, "Round | Oct 17, 2026 | $40M | Series B", TODAY) == "2021-10-14"   # 5 years: not believed


def test_news_site_sidebars_never_set_the_date():
    assert page_date("", "https://www.fintechfutures.com/venture-capital-funding/ramp-raises-750m", CHROME, TODAY) == ""
    assert page_date("2026-06-05", "https://f.com/x", CHROME, TODAY) == "2026-06-05"       # a real date stays
    assert table_row_date(CHROME, TODAY) == ""                                              # no amount in a '|' line, no date


def test_only_table_rows_with_an_amount_count_and_the_future_is_ignored():
    assert table_row_date("Round | Oct 17, 2026 | $40M | Series B", TODAY) == "2026-10-17"
    assert table_row_date("Round | Dec 17, 2026 | $40M | Series B", TODAY) == ""          # not yet
    assert table_row_date("Apr 08, 2024 and later Oct 2025 in a sentence about $5M raised.", TODAY) == ""
    assert table_row_date("Jan 5, 2021 | $1M | seed\nMar 9, 2022 | $9M | A", TODAY) == "2022-03-09"


def test_a_fetched_page_with_stale_metadata_gets_the_date_of_its_table_row(rig, monkeypatch):  # noqa: F811
    body = make_html("Stripe profile", ROW)
    meta = body.replace("<head>", '<head><meta property="article:published_time" content="2023-09-26T10:00:00Z">')
    monkeypatch.setitem(SITE, "tracxn-like", meta)
    assert rig.web.fetch(f"{rig.a}/site/tracxn-like").date == "2024-04-08"


def test_a_fetched_news_page_with_a_sidebar_stays_undated_or_keeps_its_url_date(rig, monkeypatch):  # noqa: F811
    page = make_html("Ramp's valuation jumps to $44bn", "Ramp raised a $750 million round.", "June 5, 2026",
                     "LSEG to acquire Octus in $3.5bn all-cash deal", "Oct 23, 2026")
    monkeypatch.setitem(SITE, "venture/ramp-raises-750m", page)
    monkeypatch.setitem(SITE, "2026/06/05/ramp-raises-750m", page)
    assert rig.web.fetch(f"{rig.a}/site/venture/ramp-raises-750m").date == ""              # nothing is guessed
    assert rig.web.fetch(f"{rig.a}/site/2026/06/05/ramp-raises-750m").date == "2026-06-05"


# ------------------------------------------------------------------ a date with no funding word is not scored
def test_a_date_from_an_acquisition_sentence_is_not_used_as_a_funding_date():
    q = "In March 2025, MoonPay expanded its enterprise footprint with two transactions: it acquired a crypto checkout startup."
    page = Page("https://en.wikipedia.org/wiki/MoonPay", "MoonPay", q, "2026-03-27")
    f = Fact("last_round_date", "2025-03", q, page.url, page.title, page.date)
    v = check_fact(f, page, "MoonPay")
    assert v.status == "not_confirmed" and not v.fatal and v.unusable and "no funding word" in " ".join(v.reasons)
    rec = CompanyRecord("MoonPay", {"last_round_date": merge_field("last_round_date", [(f, v)])}, [], False)
    assert compute_signals([rec], TODAY)["MoonPay"]["recent"] is None


# ------------------------------------------------------------------ one round filed as the total
STRIPE_Q = "In March 2023, Stripe completed its Series I fundraise of more than $6.5 billion at a $50 billion valuation."
STRIPE_PAGE = Page("https://en.wikipedia.org/wiki/Stripe,_Inc.", "Stripe, Inc. - Wikipedia", STRIPE_Q, "2026-10-19")


def total_fact(quote=STRIPE_Q, value="$6.5 billion"):
    return Fact("total_raised_usd", value, quote, STRIPE_PAGE.url, STRIPE_PAGE.title, STRIPE_PAGE.date)


def test_a_round_filed_as_the_total_becomes_the_last_round():
    f = total_fact()
    v = check_fact(f, STRIPE_PAGE, "Stripe")
    assert v.unusable and any("TOTAL" in r for r in v.reasons)                              # the total check refuses it ...
    figures = merge_all([(f, v)] + reclassify_rounds([(f, v)], STRIPE_PAGE, "Stripe"))
    assert figures["total_raised_usd"].usable is False                                       # ... it is still not a total
    assert figures["last_round_amount"].status == "verified" and figures["last_round_amount"].value == 6_500_000_000
    assert "describes one round" in figures["last_round_amount"].reason
    assert "last_round_date" not in figures                                                  # the quote names 2023, the page is 2026


def test_no_reclassification_when_the_model_already_gave_the_round_or_it_is_a_valuation():
    f = total_fact()
    v = check_fact(f, STRIPE_PAGE, "Stripe")
    own = Fact("last_round_amount", "$6.5 billion", STRIPE_Q, STRIPE_PAGE.url)
    assert reclassify_rounds([(f, v), (own, check_fact(own, STRIPE_PAGE, "Stripe"))], STRIPE_PAGE, "Stripe") == []
    val_q = "The company is valued at $159 billion according to people familiar with the matter."
    page = Page(STRIPE_PAGE.url, "Stripe", val_q, "2026-10-19")
    fv = Fact("total_raised_usd", "$159 billion", val_q, page.url)
    assert reclassify_rounds([(fv, check_fact(fv, page, "Stripe"))], page, "Stripe") == []


def test_fundraise_counts_as_a_funding_word():
    q = "Acme Pay completed a $90 million fundraise led by Big Fund in March 2026."
    f = Fact("last_round_amount", "$90 million", q, "https://a.com/x")
    assert check_fact(f, Page("https://a.com/x", "Acme Pay news", q), "Acme Pay").status == "verified"


# ------------------------------------------------------------------ traction must be about customers
def test_a_list_of_countries_is_not_traction_even_if_the_quote_mentions_members():
    q = "Tide currently maintains operations in the UK, India, Germany and France, where the company serves 1.6 million members."
    page = Page("https://a.com/x", "Tide profile", q)
    geo = Fact("traction_evidence", "operations in the UK, India, Germany, and France", q, page.url)
    v = check_fact(geo, page, "Tide")
    assert v.status == "not_confirmed" and v.unusable and "not about customers" in " ".join(v.reasons)
    good = Fact("traction_evidence", "1.6 million members", q, page.url)
    assert check_fact(good, page, "Tide").status == "verified"


# ------------------------------------------------------------------ two searches, a bigger pool
def two_search_setup(rig, monkeypatch, second_fails=False):  # noqa: F811
    monkeypatch.setattr("research.extract.registrable_domain", lambda u: u)             # one "domain" per URL
    keys = {"old": "2018/09/26/acme-old", "plain": "acme-plain", "mid": "2025/03/01/acme-mid", "new": "2026/10/15/acme-new"}
    for k in keys.values():
        monkeypatch.setitem(SITE, k, make_html(f"Acme Pay page {k}", "Acme Pay was founded in 2019 by two bankers."))
    urls = {n: f"{rig.a}/site/{k}" for n, k in keys.items()}
    base = Site.respond

    def respond(m, p, b):
        if p.startswith("/tavily/search") and b.get("topic") != "news":
            q = b["query"].lower()
            if "acme pay" in q and "top fintech" not in q:
                if "latest funding round" in q:
                    if second_fails:
                        return 500, "boom", "text/plain"
                    return 200, {"results": [{"title": "Acme Pay raises a new round", "url": urls["new"], "content": "Acme Pay"}]}, "application/json"
                return 200, {"results": [{"title": f"Acme Pay {n}", "url": urls[n], "content": "Acme Pay funding"}
                                         for n in ("old", "plain", "mid")]}, "application/json"
        return base(m, p, b)
    Site.respond = respond
    return urls


def acme_queries():
    return [b["query"] for m, p, b in Site.hits if p.startswith("/tavily/search") and b.get("topic") != "news"
            and "acme pay" in b["query"].lower() and "top fintech" not in b["query"].lower()]


def test_a_second_search_for_the_latest_round_feeds_the_page_pool(rig, monkeypatch):  # noqa: F811
    urls = two_search_setup(rig, monkeypatch)
    research("FinTech", rig.web, rig.llm, rig.settings, rig.bus, max_candidates=1, with_news=False)
    queries = acme_queries()
    assert len(queries) == 2 and f"latest funding round {date.today().strftime('%B %Y')}" in queries[1]
    reading = [e["url"] for e in rig.events("research_company") if e["phase"] == "reading"]
    assert reading == [urls["new"], urls["mid"]]                      # the page only the 2nd search found is the newest


def test_a_failing_second_search_does_not_stop_the_company(rig, monkeypatch):  # noqa: F811
    urls = two_search_setup(rig, monkeypatch, second_fails=True)
    run = research("FinTech", rig.web, rig.llm, rig.settings, rig.bus, max_candidates=1, with_news=False)
    reading = [e["url"] for e in rig.events("research_company") if e["phase"] == "reading"]
    # only three pages exist: both dated ones are read (newest first) before the undated one
    assert reading == [urls["mid"], urls["old"]] and run.records[0].name == "Acme Pay"


# ------------------------------------------------------------------ Jina and GDELT
def test_jina_403_is_reported_as_the_site_refusing_it_not_as_a_rejected_key(monkeypatch):
    monkeypatch.setattr(fetch_mod.requests, "get", lambda *a, **k: SimpleNamespace(status_code=403, text="", headers={}))
    w = SimpleNamespace(endpoint=lambda name: "https://r.jina.ai", count=lambda *a, **k: None)
    with pytest.raises(ProviderError) as e:
        fetch_mod.jina_fetch(w, "https://pitchbook.com/profiles/x")
    assert "refused by the site" in e.value.reason and "key" not in e.value.reason


def test_gdelt_is_opt_in_on_the_command_line():
    assert build_parser().parse_args(["FinTech"]).gdelt is False
    assert build_parser().parse_args(["FinTech", "--gdelt"]).gdelt is True