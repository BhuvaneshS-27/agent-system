"""Tests for Step 7: news (GDELT + Tavily topic=news), pacing, extraction, the whole pipeline.

Nothing touches the real internet or a real model. One fake local server plays Tavily, GDELT and a small
website (two 'domains': 127.0.0.1 and localhost); a 'smart' fake llama-server answers by looking at
which company and page it is asked about, and one of its answers is a deliberate hallucination."""
import dataclasses
import json
import threading
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
from ddgs.exceptions import DDGSException

from agents.coding.llm import LLMClient, LLMUnavailable
from core.bus import EventBus
from core.settings import load_settings
from research import extract as ex
from research import news as news_mod
from research.errors import WebStopped
from research.extract import ExtractError, find_candidates, pick_pages, pick_windows
from research.pipeline import ResearchStopped, load_run, research, run_path
from research.scoring import rank
from research.verify import Page
from research.web import WebTools

TODAY = date.today()
FILLER = ["This paragraph is general background text about the financial technology industry and how the companies "
          "in it compete for customers, talent and regulatory approval in many different countries.",
          "Another paragraph of background explains that payments, lending and banking software are the three "
          "largest areas, and that many startups try to combine them into one product for small businesses."]


def html(title, *paras):
    body = "".join(f"<p>{p}</p>" for p in (FILLER[0], *paras, FILLER[1]))
    return f"<html><head><title>{title}</title></head><body><article><h1>{title}</h1>{body}</article></body></html>"


S = {  # sentences the fake model quotes (they are really on the fake pages)
    "acme_founded": "Acme Pay was founded in 2019 by two former bankers in Berlin.",
    "acme_round": "In March 2026 Acme Pay raised $120 million in a Series C round led by Big Fund.",
    "acme_total": "Acme Pay's total funding to date is $210 million.",
    "acme_val": "The deal values Acme Pay at $1.5 billion, according to people familiar with the matter.",
    "acme_trac": "Acme Pay says it now serves more than 12,000 customers in 30 countries.",
    "acme_b_total": "Acme Pay has raised $205 million in total funding.",
    "acme_b_founded": "Acme Pay was founded in 2019.",
    "bolt_round": "Bolt Bank raised $60 million in a Series B round in January 2025.",
    "bolt_founded": "Bolt Bank was founded in 2018 in London.",
}
SITE = {
    "list1": html("Top fintech startups", "Our list includes Acme Pay, a payments company, and Bolt Bank, a digital bank."),
    "list2": html("More fintech startups", "Also worth watching is Cobalt Cash, which builds tools for small shops."),
    "acme-a": html("Acme Pay profile", S["acme_founded"], S["acme_round"], S["acme_total"], S["acme_val"], S["acme_trac"]),
    "acme-b": html("Acme Pay at a glance", S["acme_b_total"], S["acme_b_founded"]),
    "bolt-a": html("Bolt Bank profile", S["bolt_round"], S["bolt_founded"]),
    "bolt-b": html("Bolt Bank opens new office", "Bolt Bank opened an office in Madrid last week for its growing team."),
    "cobalt-a": html("Cobalt Cash profile", "Cobalt Cash builds payment terminals for small shops and cafes in several cities."),
}
FACTS = {  # (company, page tag) -> [(field, value, quote)]
    ("Acme Pay", "acme-a"): [("total_raised_usd", "$210 million", S["acme_total"]), ("last_round_amount", "$120 million", S["acme_round"]),
                             ("last_round_date", "2026-03", S["acme_round"]), ("valuation_usd", "$1.5 billion", S["acme_val"]),
                             ("founded_year", "2019", S["acme_founded"]), ("traction_evidence", "12,000 customers", S["acme_trac"])],
    ("Acme Pay", "acme-b"): [("total_raised_usd", "$205 million", S["acme_b_total"]), ("founded_year", "2019", S["acme_b_founded"])],
    ("Bolt Bank", "bolt-a"): [("total_raised_usd", "$900 million", "Bolt Bank has raised a total of $900 million."),   # invented
                              ("last_round_amount", "$60 million", S["bolt_round"]), ("last_round_date", "2025-01", S["bolt_round"]),
                              ("founded_year", "2018", S["bolt_founded"])],
    ("Bolt Bank", "bolt-b"): [],
    ("Cobalt Cash", "cobalt-a"): "garbage",                                                                              # not JSON
}
CANDIDATES = ["Acme Pay", "Bolt Bank", "Cobalt Cash", "Ghost Fin", "acme pay"]     # Ghost Fin is not in any text; one duplicate


class SmartLlama(BaseHTTPRequestHandler):
    calls: list = []
    on_facts = None

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        if "You list company names" in system:
            SmartLlama.calls.append(("candidates", ""))
            out = json.dumps({"candidates": [{"name": n} for n in CANDIDATES]})
        else:
            company = user.split("COMPANY: ", 1)[1].split("\n", 1)[0]
            SmartLlama.calls.append(("facts", company))
            if SmartLlama.on_facts:
                SmartLlama.on_facts()
            facts = next((v for (c, tag), v in FACTS.items() if c == company and tag in user), [])
            out = facts if isinstance(facts, str) else json.dumps(
                {"facts": [{"field": f, "value": v, "evidence_quote": q} for f, v, q in facts]})
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for c in ({"choices": [{"delta": {"content": out}}]},
                      {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 50, "completion_tokens": 20}}):
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        data = json.dumps({"choices": [{"message": {"content": out}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 50, "completion_tokens": 20}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class Site(BaseHTTPRequestHandler):
    respond = None          # fn(method, path, json_body) -> (status, payload, content type)
    hits: list = []

    def log_message(self, *a):
        pass

    def _go(self, body=None):
        Site.hits.append((self.command, self.path, body))
        status, payload, ctype = Site.respond(self.command, self.path, body)
        data = payload if isinstance(payload, bytes) else (payload.encode() if isinstance(payload, str) else json.dumps(payload).encode())
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._go()

    def do_POST(self):
        self._go(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"))


def iso(days_ago):
    return (TODAY - timedelta(days=days_ago)).isoformat()


def gdelt_date(days_ago):
    return (TODAY - timedelta(days=days_ago)).strftime("%Y%m%dT120000Z")


class Rig:
    pass


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    class NoDDGS:
        def __init__(self, timeout=None):
            pass

        def text(self, *a, **k):
            raise DDGSException("No results found.")

        news = text
    monkeypatch.setattr("research.search.DDGS", NoDDGS)
    monkeypatch.setattr("research.news.DDGS", NoDDGS)
    Site.hits, SmartLlama.calls, SmartLlama.on_facts = [], [], None
    state = {"gdelt_bolt": 0}
    r = Rig()
    r.state = state
    r.web_srv = ThreadingHTTPServer(("127.0.0.1", 0), Site)
    r.llm_srv = ThreadingHTTPServer(("127.0.0.1", 0), SmartLlama)
    for s in (r.web_srv, r.llm_srv):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    port = r.web_srv.server_address[1]
    r.a, r.b = f"http://127.0.0.1:{port}", f"http://localhost:{port}"      # two different "domains"

    def respond(method, path, body):
        if path.startswith("/site/"):
            return (200, SITE[path[6:]], "text/html; charset=utf-8") if path[6:] in SITE else (404, "no", "text/plain")
        if path.startswith("/tavily/search"):
            q = body["query"].lower()
            if body.get("topic") == "news":
                if "acme pay" in q:
                    return 200, {"results": [
                        {"title": "Acme Pay expands to Spain", "url": "https://news.example.org/acme-spain", "content": "x", "published_date": iso(9)},
                        {"title": "Fintech trends this autumn", "url": "https://trends.example.net/t", "content": "nothing relevant", "published_date": iso(3)},
                        {"title": "Acme Pay names a new CFO", "url": "https://cfo.example.io/acme", "content": "x", "published_date": iso(2)}]}, "application/json"
                return 200, {"results": []}, "application/json"
            if "top fintech startups" in q:
                return 200, {"results": [{"title": "Top fintech startups", "url": f"{r.a}/site/list1", "content": "Acme Pay and Bolt Bank lead."},
                                         {"title": "More fintech startups", "url": f"{r.b}/site/list2", "content": "Cobalt Cash is rising."}]}, "application/json"
            for name, tag in (("acme pay", "acme"), ("bolt bank", "bolt"), ("cobalt cash", "cobalt")):
                if name in q:
                    res = [{"title": f"{name.title()} on video", "url": "https://www.youtube.com/watch?v=1", "content": name},
                           {"title": f"{name.title()} profile", "url": f"{r.a}/site/{tag}-a", "content": f"{name.title()} funding"}]
                    if tag != "cobalt":
                        res.append({"title": f"{name.title()} at a glance", "url": f"{r.b}/site/{tag}-b", "content": f"{name.title()} facts"})
                    return 200, {"results": res}, "application/json"
            return 200, {"results": []}, "application/json"
        if path.startswith("/gdelt/api/v2/doc/doc"):
            q = parse_qs(urlparse(path).query)["query"][0]
            if '"Acme Pay"' in q:
                arts = [{"url": f"https://{d}/acme", "title": f"Acme news {i}", "seendate": gdelt_date(n), "domain": d}
                        for i, (d, n) in enumerate((("news.example.org", 9), ("bank.example.com", 20), ("wire.example.co.uk", 40)))]
                return 200, {"articles": arts}, "application/json"
            if '"Bolt Bank"' in q:
                state["gdelt_bolt"] += 1
                if state["gdelt_bolt"] == 1:
                    return 429, "Please limit requests to one every 5 seconds", "text/plain"
                return 200, {"articles": [{"url": "https://b.example.org/bolt", "title": "Bolt", "seendate": gdelt_date(5), "domain": "b.example.org"}]}, "application/json"
            return 200, "", "text/plain"                                                       # nothing matched
        return 404, "no route", "text/plain"
    Site.respond = respond

    r.waits = []
    r.bus = EventBus(tmp_path / "events.jsonl")
    r.cache_dir = tmp_path / "cache"
    r.web = WebTools(r.bus, r.cache_dir, env={"TAVILY_API_KEY": "tvly-test"}, sleep=r.waits.append, allow_private=True, timeout=(2, 2))
    r.web.endpoints.update({"tavily": r.a + "/tavily", "gdelt": r.a + "/gdelt"})
    r.settings = dataclasses.replace(load_settings(), llm_url=f"http://127.0.0.1:{r.llm_srv.server_address[1]}",
                                     max_tokens=256, cache_dir=r.cache_dir)
    r.llm = LLMClient(r.settings, r.bus, retry_wait=0)
    r.events = lambda kind: [e for e in r.bus.read_since(0)[0] if e["kind"] == kind]
    r.run = lambda **kw: research("FinTech", r.web, r.llm, r.settings, r.bus, max_candidates=5, **kw)
    yield r
    r.web_srv.shutdown()
    r.llm_srv.shutdown()


# ------------------------------------------------------------------ extraction helpers
def test_short_page_is_sent_whole_and_long_page_keeps_the_useful_paragraphs():
    assert pick_windows("short text", "Acme", 100) == "short text"
    junk = "\n".join(f"Paragraph {i} about nothing in particular, just filler words to make it long." for i in range(200))
    text = junk + "\nAcme Pay raised $120 million in a Series C round.\n" + junk
    out = pick_windows(text, "Acme Pay", 1500)
    assert len(out) <= 1500 and "Acme Pay raised $120 million in a Series C round." in out
    assert out.split("\n")[0].startswith("Paragraph 0")                    # the first paragraph is always kept


def test_every_window_is_word_for_word_from_the_page():
    text = "\n".join(f"Line {i}: Acme Pay raised ${i} million in round {i}." for i in range(300))
    out = pick_windows(text, "Acme Pay", 1200)
    assert all(line in text for line in out.split("\n"))


def test_pick_pages_prefers_pages_about_the_company_one_per_domain_no_social():
    res = [{"title": "Video", "url": "https://www.youtube.com/watch?v=1", "snippet": "Acme"},
           {"title": "Other news", "url": "https://a.com/1", "snippet": "nothing"},
           {"title": "Acme Pay profile", "url": "https://b.com/1", "snippet": ""},
           {"title": "Acme Pay again", "url": "https://b.com/2", "snippet": ""},
           {"title": "Acme Pay wiki", "url": "https://c.org/1", "snippet": ""}]
    assert [r["url"] for r in pick_pages(res, "Acme Pay", 2)] == ["https://b.com/1", "https://c.org/1"]


def test_schemas_use_only_keywords_that_llama_cpp_supports():
    allowed = {"type", "properties", "required", "additionalProperties", "items", "enum", "minItems", "maxItems",
               "minLength", "maxLength"}

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k != "properties":
                    assert k in allowed, k
                walk(v) if k != "properties" else [walk(x) for x in v.values()]
    walk(ex.facts_schema())
    walk(ex.candidates_schema(15))
    assert ex.candidates_schema(7)["properties"]["candidates"]["maxItems"] == 7


# ------------------------------------------------------------------ news
def test_tavily_news_request_uses_topic_news_and_90_days(rig):
    out = rig.web.search("Acme Pay FinTech startup news", tags={}, topic="news", providers=news_mod.TAVILY_CHAIN)
    body = next(b for m, p, b in Site.hits if p.startswith("/tavily/search"))
    assert body["topic"] == "news" and body["days"] == 90 and out.ok and out.tool == "news_search"
    assert out.results[0]["date"] == iso(9) and out.results[0]["domain"] == "example.org"
    assert rig.web.usage.month("tavily")["credits"] == 1


def test_gdelt_request_and_empty_body_means_no_news_not_failure(rig):
    out = rig.web.search(news_mod.gdelt_query("Cobalt Cash", "FinTech"), topic="news", providers=news_mod.GDELT_CHAIN)
    path = next(p for m, p, b in Site.hits if p.startswith("/gdelt/"))
    q = parse_qs(urlparse(path).query)
    assert q["mode"] == ["artlist"] and q["format"] == ["json"] and q["timespan"] == ["90d"] and q["sort"] == ["datedesc"]
    assert q["query"] == ['"Cobalt Cash" ("FinTech" OR startup OR funding) sourcelang:english']
    assert out.ok and out.results == []


def test_gdelt_429_is_retried_and_requests_are_spaced_6_seconds(rig):
    rig.web.search(news_mod.gdelt_query("Bolt Bank", "FinTech"), topic="news", providers=news_mod.GDELT_CHAIN)
    retry = rig.events("retry")[0]
    assert retry["provider"] == "gdelt" and "429" in retry["reason"]
    assert rig.state["gdelt_bolt"] == 2
    assert any(5.0 < w <= 6.0 for w in rig.waits)                   # the pace wait before the second request
    assert 2.0 in rig.waits                                          # tenacity's own wait after the 429


def test_gdelt_text_answer_with_http_200_is_an_error_that_is_not_json(rig):
    Site.respond = lambda m, p, b: (200, "The specified phrase is too short.", "text/plain")
    out = rig.web.search('"X" sourcelang:english', topic="news", providers=news_mod.GDELT_CHAIN)
    assert not out.ok and "text, not JSON" in out.error and rig.events("retry") == []     # a bad query is not retried


def test_find_news_merges_dedupes_and_drops_unrelated_semantic_hits(rig):
    arts, ok = news_mod.find_news(rig.web, "Acme Pay", "FinTech")
    urls = [a["url"] for a in arts]
    assert ok and len(arts) == 5 and "https://trends.example.net/t" not in urls
    assert len(set(urls)) == len(urls) and arts == sorted(arts, key=lambda a: a["date"], reverse=True)
    assert {a["via"] for a in arts} == {"gdelt", "tavily"}


def test_find_news_survives_one_source_failing_and_reports_both_failing(rig):
    ok_route = Site.respond
    Site.respond = lambda m, p, b: (500, "boom", "text/plain") if p.startswith("/gdelt/") else ok_route(m, p, b)
    arts, ok = news_mod.find_news(rig.web, "Acme Pay", "FinTech")
    assert ok and len(arts) == 2                                       # only Tavily answered
    assert len(rig.events("retry")) >= 2                               # GDELT was retried before giving up
    Site.respond = lambda m, p, b: (500, "boom", "text/plain")
    arts, ok = news_mod.find_news(rig.web, "Bolt Bank", "FinTech")
    assert not ok and arts == []


def test_news_and_normal_search_do_not_share_cache_entries(rig):
    rig.web.search("Acme Pay", tags={})
    n = len([h for h in Site.hits if h[1].startswith("/tavily/search")])
    rig.web.search("Acme Pay", tags={}, topic="news", providers=news_mod.TAVILY_CHAIN)
    assert len([h for h in Site.hits if h[1].startswith("/tavily/search")]) == n + 1
    rig.web.search("Acme Pay", tags={}, topic="news", providers=news_mod.TAVILY_CHAIN)       # now cached
    assert len([h for h in Site.hits if h[1].startswith("/tavily/search")]) == n + 1


def test_pace_waits_between_calls_and_stop_ends_the_wait(tmp_path):
    waits = []
    web = WebTools(None, tmp_path, env={}, sleep=waits.append)
    web.pace("gdelt", 6.0)
    web.pace("gdelt", 6.0)
    web.pace("other", 6.0)
    assert len(waits) == 1 and 5.0 < waits[0] <= 6.0
    real = WebTools(None, tmp_path / "b", env={})
    ev = threading.Event()
    ev.set()
    real._local.stop = ev
    real.pace("gdelt", 6.0)                                            # the first call never waits
    with pytest.raises(WebStopped):
        real.pace("gdelt", 6.0)


# ------------------------------------------------------------------ candidates
def test_candidates_are_only_names_that_are_written_in_the_text(rig):
    names, sources = find_candidates(rig.web, rig.llm, rig.settings, rig.bus, "FinTech", 5)
    assert names == ["Acme Pay", "Bolt Bank", "Cobalt Cash"]            # Ghost Fin and the duplicate are gone
    assert len(sources) == 2


def test_candidate_search_failure_is_a_clear_error(rig):
    Site.respond = lambda m, p, b: (500, "boom", "text/plain")
    with pytest.raises(ExtractError, match="search for candidates failed"):
        find_candidates(rig.web, rig.llm, rig.settings, rig.bus, "FinTech", 5)


# ------------------------------------------------------------------ the whole pipeline
def test_full_run_verifies_figures_rejects_the_invented_one_and_isolates_the_broken_company(rig):
    run = rig.run()
    by = {r.name: r for r in run.records}
    assert list(by) == ["Acme Pay", "Bolt Bank", "Cobalt Cash"]

    acme = by["Acme Pay"].figures
    assert acme["total_raised_usd"].status == "high" and acme["total_raised_usd"].value == 210_000_000
    assert len(acme["total_raised_usd"].domains) == 2                  # 127.0.0.1 and localhost
    assert acme["founded_year"].status == "high" and acme["founded_year"].value == 2019
    assert acme["last_round_date"].status == "verified" and acme["last_round_date"].value == "2026-03"
    assert acme["valuation_usd"].value == 1_500_000_000 and acme["valuation_usd"].status == "verified"
    assert acme["traction_evidence"].level == 1.0
    assert all(f.source_url.startswith("http") and f.evidence_quote for f in acme.values())

    bolt = by["Bolt Bank"].figures
    assert bolt["total_raised_usd"].status == "not_confirmed" and not bolt["total_raised_usd"].usable   # invented quote
    assert bolt["last_round_amount"].status == "verified" and bolt["founded_year"].value == 2018

    cobalt = by["Cobalt Cash"]
    assert cobalt.figures == {} and any("no usable answer" in n for n in cobalt.notes)                  # garbage JSON
    assert run.stats["failed_companies"] == 0 and run.stats["pages"] == 5

    assert len(by["Acme Pay"].news) == 5 and by["Acme Pay"].news_ok
    assert len(by["Bolt Bank"].news) == 1 and by["Cobalt Cash"].news_ok and by["Cobalt Cash"].news == []
    assert any(e["provider"] == "gdelt" and "429" in e["reason"] for e in rig.events("retry") if "provider" in e)

    kinds = [e["kind"] for e in rig.bus.read_since(0)[0]]
    assert kinds[0] == "research_start" and kinds[-1] == "research_done" and kinds.index("candidates") < kinds.index("verified")
    assert {"extracted", "news_result", "verified", "web_start", "provider_try", "llm_call"} <= set(kinds)
    assert [e["name"] for e in rig.events("research_company") if e["phase"] == "start"] == ["Acme Pay", "Bolt Bank", "Cobalt Cash"]
    assert all(e.get("agent") == "research" for e in rig.events("verified"))
    assert any(e["status"] == "not_confirmed" and not e["usable"] for e in rig.events("verified"))
    assert "youtube" not in " ".join(h[1] for h in Site.hits)                                            # never read

    ranked = rank(run.records, "Established", TODAY)
    assert ranked[0].name == "Acme Pay" and ranked[0].label in ("High", "Medium")
    assert ranked[-1].score == 0.0 or ranked[-1].name != "Acme Pay"


def test_run_is_saved_and_reloaded_with_the_same_ranking(rig):
    run = rig.run()
    path = run_path(rig.settings, "FinTech")
    assert path.exists() and path.name == "fintech.json" and run.path == str(path)
    again = load_run(path)
    for preset in ("Established", "Fastest rising", "Early-stage"):
        assert [(r.name, round(r.score, 6)) for r in rank(again.records, preset, TODAY)] == \
               [(r.name, round(r.score, 6)) for r in rank(run.records, preset, TODAY)]
    assert again.category == "FinTech" and again.model == rig.settings.model_name


def test_second_run_uses_the_cache_for_pages_searches_news_and_model_answers(rig):
    rig.run()
    llm_calls, hits = len(SmartLlama.calls), len(Site.hits)
    run2 = rig.run()
    assert len(Site.hits) == hits                                               # no network call at all
    # the only new model calls are Cobalt's 3 tries: a FAILED answer is never cached, so it is asked again
    assert [c for c in SmartLlama.calls[llm_calls:]] == [("facts", "Cobalt Cash")] * 3
    assert len(rig.events("llm_cache_hit")) >= 5 and len(run2.records) == 3
    rig.run(fresh=True)                                                         # fresh=True asks everything again
    assert len(SmartLlama.calls) > llm_calls and len(Site.hits) > hits


def test_llm_answers_are_cached_per_model_name(rig):
    rig.run()
    first = len(SmartLlama.calls)
    rig.settings = dataclasses.replace(rig.settings, model_name="Another model")
    rig.run()
    assert len(SmartLlama.calls) > first                                        # a different model never reuses answers


def test_without_news_no_news_request_is_made(rig):
    run = rig.run(with_news=False)
    assert all(r.news == [] and not r.news_ok for r in run.records)
    assert not [h for h in Site.hits if h[1].startswith("/gdelt/")]


def test_stop_during_the_run_raises_and_saves_nothing(rig):
    stop = threading.Event()
    rig.llm = LLMClient(rig.settings, rig.bus, retry_wait=0, stop_event=stop)         # streaming client: Stop reaches the model call
    SmartLlama.on_facts = stop.set                                                    # Stop is pressed when the first page is read
    with pytest.raises(ResearchStopped):
        rig.run(stop_event=stop)
    assert rig.events("research_stopped") and not rig.events("research_done")
    assert not run_path(rig.settings, "FinTech").exists()


def test_stop_before_the_start_ends_at_once(rig):
    stop = threading.Event()
    stop.set()
    with pytest.raises(ResearchStopped):
        rig.run(stop_event=stop)
    assert SmartLlama.calls == []


def test_model_server_down_ends_the_run_with_a_clear_event(rig):
    rig.llm = LLMClient(dataclasses.replace(rig.settings, llm_url="http://127.0.0.1:9"), rig.bus, retry_wait=0)
    with pytest.raises(LLMUnavailable):
        rig.run()
    assert "cannot reach llama-server" in rig.events("research_failed")[0]["reason"]


def test_a_company_that_crashes_is_isolated(rig, monkeypatch):
    real = ex.extract_facts

    def flaky(llm, settings, web, bus, company, page, **kw):
        if company == "Bolt Bank":
            raise RuntimeError("bug in one company")
        return real(llm, settings, web, bus, company, page, **kw)
    monkeypatch.setattr("research.pipeline.extract_facts", flaky)
    run = rig.run()
    by = {r.name: r for r in run.records}
    assert by["Bolt Bank"].notes and "failed" in by["Bolt Bank"].notes[0] and by["Bolt Bank"].figures == {}
    assert by["Acme Pay"].figures["total_raised_usd"].status == "high"
    assert run.stats["failed_companies"] == 1 and rig.events("error")


def test_extract_facts_ignores_unknown_fields_and_duplicates(rig, monkeypatch):
    monkeypatch.setattr(ex, "ask_json", lambda *a, **k: {"facts": [
        {"field": "ceo_name", "value": "Jo", "evidence_quote": "The CEO is Jo Smith of Acme."},
        {"field": "founded_year", "value": "2019", "evidence_quote": S["acme_founded"]},
        {"field": "founded_year", "value": "2019", "evidence_quote": S["acme_founded"]},
        {"field": "founded_year", "value": "", "evidence_quote": "x" * 20}]})
    facts = ex.extract_facts(rig.llm, rig.settings, rig.web, rig.bus, "Acme Pay", Page("http://x.test/a", "Acme", "text " * 10), fresh=True)
    assert [(f.field, f.value, f.source_url) for f in facts] == [("founded_year", "2019", "http://x.test/a")]


def test_every_progress_line_formatter_of_the_cli_script_works_on_real_events(rig):
    from scripts.run_research import EVENT_LINES, cell, money
    run = rig.run()
    seen = set()
    for e in rig.bus.read_since(0)[0]:
        if e["kind"] in EVENT_LINES:
            assert isinstance(EVENT_LINES[e["kind"]](e), str)
            seen.add(e["kind"])
    assert {"research_start", "candidates", "research_company", "extracted", "verified", "news_result", "retry",
            "research_done"} <= seen
    bolt = next(r for r in run.records if r.name == "Bolt Bank")
    assert cell(bolt, "total_raised_usd", money) == "rejected" and cell(bolt, "founded_year") == "2018"
    assert [money(x) for x in (1.5e9, 2.1e8, 4e4, 12, None)] == ["$1.5B", "$210M", "$40K", "$12", "-"]