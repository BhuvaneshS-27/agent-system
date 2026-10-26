"""Startup Research tab: run the pipeline, watch progress, ranked table, formula, chart, news, preset switch."""
import time
from datetime import date
from pathlib import Path

from nicegui import ui

from research.pipeline import load_run
from research.runner import research_job
from research.scoring import (DISCLAIMER, FORMULA, NORMALISATION, PRESETS, SIGNAL_LABELS, SIGNALS, YOUNGER_IS_BETTER,
                              compute_signals, figure_is_stale, news_in_window, rank_signals)
from ui.common import format_event

EXAMPLES = ["FinTech", "HealthTech", "Climate tech", "AI agents", "Cybersecurity"]
FIG_COLS = {"raised": ("Raised", "total_raised_usd"), "valuation": ("Valuation", "valuation_usd"),
            "last_round": ("Last round", "last_round_date"), "founded": ("Founded", "founded_year"),
            "traction": ("Traction", "traction_evidence")}
ICONS = {"high": ("verified", "positive"), "verified": ("check_circle", "positive"),
         "not_confirmed": ("help_outline", "warning")}
TRACTION = {1: "strong", 0.5: "some", 0: "none", 1.0: "strong", 0.0: "none"}
CONF_COLOR = {"High": "positive", "Medium": "warning", "Low": "negative"}


def money(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if v >= size:
            return f"${v / size:.3g}{unit}"
    return f"${v:.0f}"


def fmt_value(fname, fig) -> str:
    if fname in ("total_raised_usd", "valuation_usd"):
        return money(fig.value)
    if fname == "last_round_date":
        return str(fig.value)[:7]
    if fname == "traction_evidence":
        return TRACTION.get(fig.level, "some")
    return str(fig.value)


def figure_cells(rec, key, fname, today) -> dict:
    fig = rec.figures.get(fname)
    if fig is None or fig.value in (None, ""):
        return {key: "-", key + "_url": "", key + "_icon": "", key + "_color": "", key + "_tip": "no data found"}
    if not fig.usable:
        icon, color, state = "block", "negative", "rejected (never used)"
    else:
        icon, color = ICONS.get(fig.status, ("help_outline", "warning"))
        state = {"high": "verified + second source", "verified": "verified", "not_confirmed": "not confirmed"}.get(fig.status, fig.status)
    stale = figure_is_stale(fig, today)
    tip = f'{state}: "{(fig.evidence_quote or "")[:140]}" ({fig.domain}, source date {fig.as_of or "unknown"})'
    if stale:
        tip += " - source older than 18 months"
    return {key: fmt_value(fname, fig) + (" !" if stale else ""), key + "_url": fig.source_url, key + "_icon": icon,
            key + "_color": color, key + "_tip": tip}


def build_research_tab(ctx, pump) -> None:
    orch = ctx.orchestrator
    st = {"run": None, "signals": None, "idx": 0, "total": 0, "name": ""}
    cache_folder = Path(ctx.settings.cache_dir) / "research"

    def saved_paths() -> dict:
        files = sorted(cache_folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if cache_folder.exists() else []
        return {p.stem: p for p in files}

    # ------------------------------------------------------------ run controls
    with ui.card().classes("w-full"):
        ui.label("Startup research").classes("text-lg font-bold")
        ui.label(DISCLAIMER).classes("text-xs opacity-70")
        with ui.row().classes("items-center w-full gap-2"):
            category = ui.input("Category", placeholder="e.g. FinTech").classes("w-64").on("keydown.enter", lambda: run_research())
            limit = ui.select({3: "3 companies", 5: "5 companies", 10: "10 companies", 15: "15 companies"}, value=5, label="How many").classes("w-40")
            fresh = ui.checkbox("Ignore cache")
            news_sw = ui.checkbox("News", value=True)
            gdelt = ui.checkbox("Also GDELT (slow)")
            run_btn = ui.button("Run research", icon="play_arrow", on_click=lambda: run_research())
            stop_btn = ui.button("Stop", icon="stop", color="negative", on_click=lambda: stop())
        with ui.row().classes("items-center gap-1"):
            ui.label("Examples:").classes("text-xs opacity-70")
            for c in EXAMPLES:
                ui.chip(c, on_click=lambda c=c: category.set_value(c)).props("dense clickable")
        progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
        status = ui.label("Idle. Run a category, or load a saved run below.").classes("text-sm")
        with ui.expansion("Research activity (every step as an event)", icon="list").classes("w-full"):
            log = ui.log(max_lines=500).classes("w-full h-64")

    # ------------------------------------------------------------ results
    with ui.card().classes("w-full"):
        with ui.row().classes("items-center gap-3"):
            saved = ui.select(list(saved_paths()), label="Saved runs").classes("w-56")
            ui.button("Load", icon="folder_open", on_click=lambda: load_saved())
            preset = ui.select(list(PRESETS), value="Established", label="Ranking preset", on_change=lambda e: show()).classes("w-48")
            strict = ui.switch("Strict (ignore not-confirmed figures)", on_change=lambda e: recompute())
            speed = ui.label("").classes("text-xs opacity-70")
        info = ui.label("No run loaded.").classes("text-sm opacity-80")
        columns = [{"name": "rank", "label": "#", "field": "rank", "align": "left"},
                   {"name": "name", "label": "Startup", "field": "name", "align": "left"},
                   {"name": "score", "label": "Score", "field": "score"},
                   {"name": "conf", "label": "Confidence", "field": "conf"}]
        columns += [{"name": k, "label": lbl, "field": k, "align": "left"} for k, (lbl, _) in FIG_COLS.items()]
        columns += [{"name": "news_n", "label": "News 90d", "field": "news_n"}]
        columns += [{"name": "s_" + s, "label": SIGNAL_LABELS[s] + " (0-100)", "field": "s_" + s} for s in SIGNALS]
        table = ui.table(columns=columns, rows=[], row_key="name", pagination={"rowsPerPage": 0}).classes("w-full").props("dense flat")
        table.add_slot("body-cell-conf", '<q-td :props="props"><q-badge :color="props.row.conf_color">{{ props.value }}</q-badge></q-td>')
        for key in FIG_COLS:
            table.add_slot(f"body-cell-{key}", f'''
              <q-td :props="props">
                <a v-if="props.row.{key}_url" :href="props.row.{key}_url" target="_blank" style="color:inherit">{{{{ props.value }}}}</a>
                <span v-else>{{{{ props.value }}}}</span>
                <q-icon v-if="props.row.{key}_icon" :name="props.row.{key}_icon" :color="props.row.{key}_color" size="xs" class="q-ml-xs">
                  <q-tooltip max-width="420px">{{{{ props.row.{key}_tip }}}}</q-tooltip></q-icon>
              </q-td>''')
        ui.label("Icons: double-check = verified by 2 domains, check = verified, ? = not confirmed, block = rejected. "
                 "Click a figure to open its source; hover the icon for the quote. ! = source older than 18 months. "
                 "'-' = no data (the score renormalises over the signals that exist).").classes("text-xs opacity-60")

    with ui.expansion("How the score is calculated", icon="functions").classes("w-full"):
        ui.label(FORMULA).classes("font-mono text-sm")
        ui.label("decay(t) = 2 ^ (-t / h)    (h = half-life)").classes("font-mono text-sm")
        wcols = [{"name": "sig", "label": "Signal", "field": "sig", "align": "left"}]
        wcols += [{"name": p, "label": p, "field": p} for p in PRESETS]
        wrows = []
        for s in SIGNALS:
            row = {"sig": SIGNAL_LABELS[s]}
            for p, w in PRESETS.items():
                row[p] = f"{w[s]:.2f}" + (" (younger is better)" if s == "maturity" and p in YOUNGER_IS_BETTER else "")
            wrows.append(row)
        ui.table(columns=wcols, rows=wrows, row_key="sig").props("dense flat")
        ui.label("Normalisation rules").classes("font-bold q-mt-sm")
        for line in NORMALISATION:
            ui.label("- " + line).classes("text-sm")
        ui.label("A figure is 'verified' only if its quote is in the downloaded page AND the number in the quote equals the "
                 "extracted number. A second domain within 10 percent makes it high trust. 'Not confirmed' figures are used "
                 "only when nothing better exists (switch Strict on to ignore them). Missing data is not punished in the "
                 "score; it lowers the confidence instead. " + DISCLAIMER).classes("text-sm")

    with ui.card().classes("w-full"):
        with ui.row().classes("items-center gap-3"):
            ui.label("Total raised and valuation").classes("font-bold")
            logaxis = ui.switch("Log axis", value=True, on_change=lambda e: show_chart())
            ui.label("Grey short bar = unknown").classes("text-xs opacity-60")
        chart = ui.echart({"xAxis": {"type": "category", "data": []}, "yAxis": {"type": "value"}, "series": []}).classes("w-full h-96")

    with ui.card().classes("w-full"):
        ui.label("News per startup").classes("font-bold")
        news_box = ui.column().classes("w-full")

    # ------------------------------------------------------------ actions
    def run_research() -> None:
        cat = " ".join((category.value or "").split())
        if not cat:
            ui.notify("Type a category first", type="warning")
            return
        try:
            orch.start("research", research_job, settings=ctx.settings, web=ctx.web, category=cat,
                       limit=int(limit.value), fresh=bool(fresh.value), with_news=bool(news_sw.value), gdelt=bool(gdelt.value))
        except RuntimeError as e:
            ui.notify(str(e), type="warning")

    def stop() -> None:
        if not orch.stop("research"):
            ui.notify("Nothing is running", type="info")

    def refresh_saved(select: str = "") -> None:
        names = list(saved_paths())
        saved.set_options(names, value=select or saved.value)

    def load_saved() -> None:
        path = saved_paths().get(saved.value or "")
        if path is None:
            ui.notify("Pick a saved run first", type="warning")
            return
        load_path(path)

    def load_path(path) -> None:
        try:
            run = load_run(path)
        except Exception as e:
            ui.notify(f"Could not load {Path(path).name}: {e}", type="negative")
            return
        st["run"] = run
        info.set_text(f"{run.category}: {len(run.records)} companies, researched {run.created} with {run.model} ({run.seconds} s)")
        recompute()

    def recompute() -> None:
        """Normalised signals are computed once per run (and when Strict changes); presets only re-weight them."""
        run = st["run"]
        if run is None:
            return
        st["signals"] = compute_signals(run.records, date.today(), not strict.value)
        show()

    def show() -> None:
        run = st["run"]
        if run is None or st["signals"] is None:
            return
        t0 = time.perf_counter()
        today = date.today()
        rows = rank_signals(run.records, st["signals"], preset.value, not strict.value, today)[:10]
        out = []
        for r in rows:
            d = {"rank": r.rank, "name": r.name, "score": f"{r.score:.1f}", "conf": f"{r.label} {r.confidence:.2f}",
                 "conf_color": CONF_COLOR.get(r.label, "grey"), "news_n": news_in_window(r.record.news, today)}
            for key, (_, fname) in FIG_COLS.items():
                d.update(figure_cells(r.record, key, fname, today))
            for s in SIGNALS:
                v = r.signals.get(s)
                d["s_" + s] = "-" if v is None else round(v * 100)
            out.append(d)
        table.rows = out
        table.update()
        speed.set_text(f"re-ranked in {(time.perf_counter() - t0) * 1000:.0f} ms (no network)")
        st["rows"] = rows
        show_chart()
        show_news()

    def show_chart() -> None:
        rows = st.get("rows") or []
        log_axis = bool(logaxis.value)
        known_vals = [f.value for r in rows for fn in ("total_raised_usd", "valuation_usd")
                      for f in [r.record.figures.get(fn)] if f and f.usable and isinstance(f.value, (int, float)) and f.value > 0]
        if log_axis:
            ymin = 10 ** (len(str(int(min(known_vals)))) - 2) if known_vals else 1e5
            ph = ymin * 3
        else:
            ymin = 0
            ph = (max(known_vals) * 0.03) if known_vals else 1

        def series(label, fname):
            data = []
            for r in rows:
                f = r.record.figures.get(fname)
                v = f.value if f and f.usable and isinstance(f.value, (int, float)) and f.value > 0 else None
                item = {"value": v if v is not None else ph, "known": v is not None, "text": money(v) if v else ""}
                if v is None:
                    item["itemStyle"] = {"color": "#888888", "opacity": 0.4}
                data.append(item)
            return {"name": label, "type": "bar", "data": data}

        opts = {"color": ["#5470c6", "#91cc75"], "textStyle": {"color": "#cccccc"},
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"},
                            ":formatter": "function(ps){return ps[0].name+'<br>'+ps.map(function(p){return p.seriesName+': '+(p.data.known?p.data.text:'unknown')}).join('<br>')}"},
                "legend": {"data": ["Total raised", "Valuation"], "textStyle": {"color": "#cccccc"}},
                "grid": {"left": 70, "right": 20, "bottom": 70, "top": 40},
                "xAxis": {"type": "category", "data": [r.name for r in rows], "axisLabel": {"rotate": 30}},
                "yAxis": {"type": "log" if log_axis else "value", "min": ymin,
                          "axisLabel": {":formatter": "function(v){if(v>=1e9)return '$'+(v/1e9)+'B';if(v>=1e6)return '$'+(v/1e6)+'M';if(v>=1e3)return '$'+(v/1e3)+'K';return '$'+v}"}},
                "series": [series("Total raised", "total_raised_usd"), series("Valuation", "valuation_usd")]}
        chart.options.clear()
        chart.options.update(opts)
        chart.update()

    def show_news() -> None:
        today = date.today()
        news_box.clear()
        with news_box:
            for r in st.get("rows") or []:
                rec = r.record
                items = sorted(rec.news, key=lambda a: a.get("date") or "", reverse=True)[:8]
                with ui.expansion(f"{r.rank}. {r.name}: {news_in_window(rec.news, today)} articles in the last 90 days",
                                  icon="newspaper").classes("w-full"):
                    if not items:
                        ui.label("News search failed." if not rec.news_ok else "No news found.").classes("text-sm opacity-70")
                    for a in items:
                        with ui.row().classes("items-center gap-2"):
                            ui.label(a.get("date") or "").classes("text-xs opacity-60")
                            ui.link(a.get("title") or a["url"], a["url"], new_tab=True).classes("text-sm")
                            ui.label(a.get("domain") or "").classes("text-xs opacity-60")
                    for note in rec.notes[:3]:
                        ui.label("note: " + note[:140]).classes("text-xs text-orange")

    # ------------------------------------------------------------ events
    def set_status(text: str) -> None:
        status.set_text(text)

    def on_event(ev: dict) -> None:
        if ev.get("agent") != "research":
            return
        log.push(format_event(ev))
        k = ev["kind"]
        if k == "research_start":
            st.update(idx=0, total=0)
            progress.set_value(0)
            set_status(f"Searching for {ev['category']} startups ...")
        elif k == "candidates":
            st["total"] = max(ev["count"], 1)
            set_status(f"{ev['count']} candidates: " + ", ".join(ev["names"])[:200])
        elif k == "research_company":
            if ev["phase"] == "start":
                st.update(idx=ev["index"], total=ev["total"], name=ev["name"])
                progress.set_value((ev["index"] - 1) / max(ev["total"], 1))
                set_status(f"[{ev['index']}/{ev['total']}] {ev['name']}")
            else:
                what = {"fetching": "fetching", "reading": "model reading", "searching": "searching", "news": "news"}.get(ev["phase"], ev["phase"])
                set_status(f"[{st['idx']}/{st['total']}] {ev['name']}: {what} " + (ev.get("url") or "")[:70])
        elif k == "verified":
            set_status(f"[{st['idx']}/{st['total']}] {ev['name']}: {ev['field']} = {ev['value']} -> {ev['status']}")
        elif k == "research_done":
            progress.set_value(1)
            set_status(f"Done: {ev['companies']} companies in {ev['seconds']} s")
            if ev.get("path"):
                refresh_saved(Path(ev["path"]).stem)
                load_path(ev["path"])
        elif k == "research_stopped":
            set_status("Stopped.")
        elif k == "research_failed":
            set_status("Failed: " + str(ev.get("reason", ""))[:200])
        elif k == "job_failed":
            ui.notify("Research failed: " + str(ev.get("error", ""))[:200], type="negative")
            set_status("Failed: " + str(ev.get("error", ""))[:200])

    pump.subscribe(on_event)

    def buttons() -> None:
        running = orch.is_running("research")
        run_btn.set_enabled(not running)
        stop_btn.set_enabled(running)

    ui.timer(1.0, buttons)
    names = list(saved_paths())
    if names:                                   # show the newest saved run straight away
        saved.set_value(names[0])
        load_path(saved_paths()[names[0]])