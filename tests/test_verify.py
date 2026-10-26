"""Tests for research/verify.py: the evidence checks that decide 'verified' or 'not confirmed'. No network, no LLM."""
from decimal import Decimal

import pytest

from research.verify import (Fact, Figure, Page, agree, check_fact, merge_all, merge_field, norm, parse_amounts,
                             parse_date_value, parse_value_amount, registrable_domain, traction_level)

PAGE_TEXT = (
    "Acme Pay is a payments startup based in Berlin.\n\n"
    "The company was founded in 2019 by two former bankers.\n\n"
    "In March 2026 Acme Pay raised $120 million in a Series C round led by Big Fund.\n\n"
    "The deal values the company at $1.5 billion, according to people familiar with the matter.\n\n"
    "Acme Pay says it now serves more than 12,000 customers in 30 countries.\n\n"
    "Total funding to date is $210 million.")


def page(text=PAGE_TEXT, url="https://news.example.com/acme", title="Acme Pay raises Series C"):
    return Page(url, title, text)


def fact(field, value, quote, url="https://news.example.com/acme"):
    return Fact(field, value, quote, url, "")


# ------------------------------------------------------------------ amounts
@pytest.mark.parametrize("text,expected", [
    ("raised $1.5 billion", [Decimal("1500000000")]),
    ("raised $250M", [Decimal("250000000")]),
    ("raised US$ 250 million", [Decimal("250000000")]),
    ("raised $2.5-billion", [Decimal("2500000000")]),
    ("$1,200,000 seed", [Decimal("1200000")]),
    ("raised 40 million dollars", [Decimal("40000000")]),
    ("USD 75 mn", [Decimal("75000000")]),
    ("$3.2bn valuation", [Decimal("3200000000")]),
])
def test_usd_amounts_are_scaled(text, expected):
    got = [a.value for a in parse_amounts(text) if a.usd]
    assert got == expected


@pytest.mark.parametrize("text", ["raised EUR 100 million", "raised \u20ac100 million", "raised \u00a350M",
                                  "raised CA$50 million", "founded in 2019", "12 members", "5 months ago"])
def test_other_currencies_and_plain_numbers_are_not_dollars(text):
    assert [a for a in parse_amounts(text) if a.usd] == []


def test_value_text_written_by_the_model():
    assert parse_value_amount("$1.5 billion") == Decimal("1500000000")
    assert parse_value_amount("1500000000") == Decimal("1500000000")
    assert parse_value_amount("about 40M") == Decimal("40000000")
    assert parse_value_amount("lots") is None


def test_norm_hides_only_formatting():
    assert norm("  **Acme**\u2019s   [raise](http://x.y)  \u2014 big ") == "acme's raise - big"


def test_dates():
    assert parse_date_value("2026-03") == (2026, 3, None)
    assert parse_date_value("2026-03-15") == (2026, 3, 15)
    assert parse_date_value("March 2026") == (2026, 3, None)
    assert parse_date_value("Sept 2025") == (2025, 9, None)
    assert parse_date_value("2024") == (2024, None, None)
    assert parse_date_value("last spring") is None


# ------------------------------------------------------------------ one fact
def test_a_true_quote_with_the_right_number_is_verified():
    v = check_fact(fact("total_raised_usd", "$210 million", "Total funding to date is $210 million."), page(), "Acme Pay")
    assert (v.status, v.fatal, v.parsed) == ("verified", False, 210_000_000)


def test_model_may_write_the_number_in_another_style():
    v = check_fact(fact("total_raised_usd", "210000000", "Total funding to date is $210 million."), page(), "Acme Pay")
    assert v.status == "verified" and v.parsed == 210_000_000


def test_invented_quote_is_fatal():
    v = check_fact(fact("total_raised_usd", "$900 million", "Acme Pay has raised a total of $900 million."), page(), "Acme Pay")
    assert v.status == "not_confirmed" and v.fatal and "not in the page text" in " ".join(v.reasons)


def test_real_quote_but_wrong_number_is_fatal():
    v = check_fact(fact("total_raised_usd", "$900 million", "Total funding to date is $210 million."), page(), "Acme Pay")
    assert v.status == "not_confirmed" and v.fatal and "not written in the quote" in " ".join(v.reasons)


def test_quote_matching_survives_case_spaces_and_curly_quotes():
    text = "Acme Pay\u2019s total funding to date is **$210  million**."
    v = check_fact(fact("total_raised_usd", "$210 million", "acme pay's total funding to date is $210 million."),
                   page(text), "Acme Pay")
    assert v.status == "verified"


def test_euro_amount_cannot_be_verified_as_dollars():
    p = page("Acme Pay raised \u20ac120 million in a Series C round.")
    v = check_fact(fact("last_round_amount", "$120 million", "Acme Pay raised \u20ac120 million in a Series C round."), p, "Acme Pay")
    assert v.fatal


def test_valuation_number_given_as_total_raised_is_not_confirmed():
    q = "The deal values the company at $1.5 billion, according to people familiar with the matter."
    v = check_fact(fact("total_raised_usd", "$1.5 billion", q), page(), "Acme Pay")
    assert v.status == "not_confirmed" and not v.fatal and "funding amount" in " ".join(v.reasons)
    ok = check_fact(fact("valuation_usd", "$1.5 billion", q), page(), "Acme Pay")
    assert ok.status == "verified"


def test_raised_at_a_valuation_sentence_keeps_the_two_numbers_apart():
    t = "Acme Pay raised $50 million at a $500 million valuation."
    p = page(t)
    assert check_fact(fact("last_round_amount", "$50 million", t), p, "Acme Pay").status == "verified"
    assert check_fact(fact("valuation_usd", "$500 million", t), p, "Acme Pay").status == "verified"
    assert check_fact(fact("last_round_amount", "$500 million", t), p, "Acme Pay").status == "not_confirmed"
    assert check_fact(fact("valuation_usd", "$50 million", t), p, "Acme Pay").status == "not_confirmed"


def test_quote_about_another_company_is_not_confirmed_but_not_fatal():
    t = "Rival Corp raised $90 million in a Series B round."
    v = check_fact(fact("last_round_amount", "$90 million", t), page(t, "https://list.example.com/top10", "Top 10 startups"), "Acme Pay")
    assert v.status == "not_confirmed" and not v.fatal and "names the company" in " ".join(v.reasons)


def test_company_named_only_in_page_title_or_url_is_accepted():
    t = "The company raised $90 million in a Series B round."
    assert check_fact(fact("last_round_amount", "$90 million", t), page(t, "https://x.com/a", "Acme Pay news"), "Acme Pay").status == "verified"
    assert check_fact(fact("last_round_amount", "$90 million", t), page(t, "https://acmepay.com/blog", "News"), "Acme Pay").status == "not_confirmed"
    assert check_fact(fact("last_round_amount", "$90 million", t), page(t, "https://x.com/acme-pay/news", "News"), "Acme Pay").status == "verified"


def test_founded_year():
    q = "The company was founded in 2019 by two former bankers."
    assert check_fact(fact("founded_year", "2019", q), page(), "Acme Pay").parsed == 2019
    assert check_fact(fact("founded_year", "2015", q), page(), "Acme Pay").fatal
    assert check_fact(fact("founded_year", "soon", q), page(), "Acme Pay").fatal


def test_last_round_date_needs_year_and_month_in_the_quote():
    q = "In March 2026 Acme Pay raised $120 million in a Series C round led by Big Fund."
    assert check_fact(fact("last_round_date", "2026-03", q), page(), "Acme Pay").parsed == "2026-03"
    assert check_fact(fact("last_round_date", "2026", q), page(), "Acme Pay").parsed == "2026"
    assert check_fact(fact("last_round_date", "2026-05", q), page(), "Acme Pay").fatal      # month not in quote
    assert check_fact(fact("last_round_date", "2025-03", q), page(), "Acme Pay").fatal      # year not in quote


def test_traction_levels():
    assert traction_level("Acme Pay says it now serves more than 12,000 customers in 30 countries.") == 1.0
    assert traction_level("It reached $40 million in annual recurring revenue.") == 1.0
    assert traction_level("The firm has signed several partners and customers.") == 0.5
    assert traction_level("The company has a nice office.") == 0.0


def test_traction_fact_is_verified_and_carries_its_level():
    q = "Acme Pay says it now serves more than 12,000 customers in 30 countries."
    v = check_fact(fact("traction_evidence", "12,000 customers", q), page(), "Acme Pay")
    assert v.status == "verified" and v.level == 1.0


def test_unknown_field_and_short_quote_are_fatal():
    assert check_fact(fact("ceo_name", "Jo", "Acme Pay is led by Jo Smith, the CEO."), page(), "Acme Pay").fatal
    assert check_fact(fact("founded_year", "2019", "2019"), page(), "Acme Pay").fatal


# ------------------------------------------------------------------ several sources
def test_registrable_domain():
    assert registrable_domain("https://www.bbc.co.uk/news/x") == "bbc.co.uk"
    assert registrable_domain("https://news.example.com/a") == "example.com"


def test_agreement_rules():
    assert agree("total_raised_usd", 100_000_000, 108_000_000)
    assert not agree("total_raised_usd", 100_000_000, 130_000_000)
    assert agree("founded_year", 2019, 2019) and not agree("founded_year", 2019, 2020)
    assert agree("last_round_date", "2026-03", "2026-03-15") and agree("last_round_date", "2026", "2026-03")
    assert not agree("last_round_date", "2026-03", "2026-04")


def _checked(field, value, quote, url, text, name="Acme Pay", title="Acme Pay"):
    f = fact(field, value, quote, url)
    return f, check_fact(f, Page(url, title, text), name)


def test_second_domain_with_the_same_number_upgrades_to_high():
    q1 = "Total funding to date is $210 million."
    q2 = "Acme Pay has raised $205 million in total funding."
    a = _checked("total_raised_usd", "$210 million", q1, "https://a.example.com/1", q1)
    b = _checked("total_raised_usd", "$205 million", q2, "https://b.other.org/2", q2)
    fig = merge_field("total_raised_usd", [a, b])
    assert fig.status == "high" and fig.value == 210_000_000 and fig.domains == ["example.com", "other.org"]


def test_same_domain_twice_is_not_high():
    q = "Total funding to date is $210 million."
    a = _checked("total_raised_usd", "$210 million", q, "https://a.example.com/1", q)
    b = _checked("total_raised_usd", "$210 million", q, "https://b.example.com/2", q)
    assert merge_field("total_raised_usd", [a, b]).status == "verified"


def test_disagreeing_sources_keep_the_value_with_most_support_and_say_so():
    q1, q2, q3 = ("Acme Pay raised $100 million in total.", "Acme Pay raised $104 million in total.",
                  "Acme Pay raised $400 million in total.")
    items = [_checked("total_raised_usd", v, q, f"https://{d}/x", q) for v, q, d in
             (("$400 million", q3, "c.com"), ("$100 million", q1, "a.com"), ("$104 million", q2, "b.com"))]
    fig = merge_field("total_raised_usd", items)
    assert fig.status == "high" and fig.value == 100_000_000 and "disagree" in fig.reason


def test_verified_beats_not_confirmed_and_fatal_is_kept_only_for_display():
    good = _checked("founded_year", "2019", "The company was founded in 2019 by two former bankers.", "https://a.com/1", PAGE_TEXT)
    bad = _checked("founded_year", "2012", "Acme Pay was founded in 2012 in Paris.", "https://b.com/1", "other text entirely")
    fig = merge_field("founded_year", [bad, good])
    assert fig.status == "verified" and fig.value == 2019 and fig.usable
    only_bad = merge_field("founded_year", [bad])
    assert only_bad.status == "not_confirmed" and not only_bad.usable and only_bad.value == 2012


def test_soft_failure_is_usable_as_a_fallback():
    t = "Rival Corp raised $90 million in a Series B round."
    soft = _checked("last_round_amount", "$90 million", t, "https://a.com/1", t, title="Top 10")
    fig = merge_field("last_round_amount", [soft])
    assert fig.status == "not_confirmed" and fig.usable and fig.value == 90_000_000


def test_merge_all_and_figure_roundtrip():
    items = [_checked("total_raised_usd", "$210 million", "Total funding to date is $210 million.", "https://a.com/1", PAGE_TEXT),
             _checked("founded_year", "2019", "The company was founded in 2019 by two former bankers.", "https://a.com/1", PAGE_TEXT)]
    figs = merge_all(items)
    assert set(figs) == {"total_raised_usd", "founded_year"}
    again = Figure.from_dict(figs["founded_year"].to_dict())
    assert again == figs["founded_year"]