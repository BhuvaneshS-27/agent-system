"""Job body for the Startup Research tab: runs research() as an orchestrator job."""
from agents.coding.llm import LLMClient, LLMError
from core.jobs import JobFailed
from research.extract import ExtractError
from research.pipeline import ResearchStopped, research


def research_job(job, bus, *, settings, web, category, limit=5, fresh=False, with_news=True, gdelt=False):
    llm = LLMClient(settings, bus, stop_event=job.stop_event)
    state, detail = llm.health()
    if state != "online":
        raise JobFailed(f"the model server is {state} ({detail}). Start it: python scripts/start_llm.py")
    job.provider_used = settings.model_name
    job.attempts = 1
    was_off = "gdelt" in web.disabled
    if gdelt:
        web.disabled.discard("gdelt")
    else:
        web.disabled.add("gdelt")
    try:
        research(category, web, llm, settings, bus, max_candidates=limit, with_news=with_news,
                 fresh=fresh, stop_event=job.stop_event)
    except ResearchStopped:
        return                                   # stop_event is set, so the orchestrator marks the job "stopped"
    except (LLMError, ExtractError) as e:
        raise JobFailed(str(e)) from e
    finally:
        if was_off:
            web.disabled.add("gdelt")
        else:
            web.disabled.discard("gdelt")