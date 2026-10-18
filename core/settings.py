"""Loads .env (secrets) and config.yaml (profiles) into one frozen Settings object."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    profile: str
    llm_url: str
    model_name: str
    hf_repo: str
    quant: str
    model_path: str
    ctx: int
    threads: int
    max_steps: int
    max_tokens: int
    cache_ram_mib: Optional[int]
    extra_args: tuple
    host: str
    port: int
    open_browser: bool
    title: str
    workspace_dir: Path
    outputs_dir: Path
    runs_dir: Path
    cache_dir: Path

    @property
    def summary(self) -> str:
        return f"{self.model_name} {self.quant} | ctx {self.ctx} | max {self.max_steps} steps"


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    profile = os.getenv("PROFILE", "dev").strip().lower()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    profiles = cfg["profiles"]
    if profile not in profiles:
        raise ValueError(f"PROFILE={profile!r} is not defined in config.yaml. Use one of: {list(profiles)}")
    p = profiles[profile]
    paths = cfg["app"]["paths"]
    cache_ram = p.get("cache_ram_mib")

    settings = Settings(
        profile=profile,
        llm_url=os.getenv("LLM_URL", "http://127.0.0.1:8081").rstrip("/"),
        model_name=p["model_name"],
        hf_repo=p.get("hf_repo") or "",
        quant=p["quant"],
        model_path=p.get("model_path") or "",
        ctx=int(p["ctx"]),
        threads=int(p["threads"]),
        max_steps=int(p["max_steps"]),
        max_tokens=int(p.get("max_tokens", 1024)),
        cache_ram_mib=int(cache_ram) if cache_ram is not None else None,
        extra_args=tuple(str(a) for a in (p.get("extra_args") or [])),
        host=p["host"],
        port=int(cfg["app"]["port"]),
        open_browser=bool(p["open_browser"]),
        title=cfg["app"]["title"],
        workspace_dir=ROOT / paths["workspace"],
        outputs_dir=ROOT / paths["outputs"],
        runs_dir=ROOT / paths["runs"],
        cache_dir=ROOT / paths["cache"],
    )
    for d in (settings.workspace_dir, settings.outputs_dir, settings.runs_dir, settings.cache_dir):
        d.mkdir(parents=True, exist_ok=True)
    return settings