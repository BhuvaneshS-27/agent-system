"""Starts llama-server with the settings of the active PROFILE (config.yaml + .env).

    python scripts/start_llm.py             start the server (leave this window open)
    python scripts/start_llm.py --dry-run   only print the command
"""
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.settings import load_settings  # noqa: E402


def build_command(s) -> list:
    cmd = [os.getenv("LLAMA_SERVER") or "llama-server"]
    if s.model_path:
        cmd += ["-m", s.model_path]
    elif s.hf_repo:
        cmd += ["-hf", f"{s.hf_repo}:{s.quant}"]
    else:
        raise SystemExit("config.yaml: set hf_repo or model_path for this profile.")
    port = urlparse(s.llm_url).port or 8081
    cmd += ["-c", str(s.ctx), "-t", str(s.threads), "--parallel", "1", "--jinja",
            "--host", "127.0.0.1", "--port", str(port), "--no-mmproj"]
    if s.cache_ram_mib is not None:
        cmd += ["--cache-ram", str(s.cache_ram_mib)]
    cmd += list(s.extra_args)
    return cmd


def main() -> int:
    s = load_settings()
    cmd = build_command(s)
    print(f"Profile: {s.profile} | {s.summary}")
    print("Command:", subprocess.list2cmdline(cmd))
    if "--dry-run" in sys.argv:
        return 0
    print("Health check: http://127.0.0.1:%s/health  (answers when the model has loaded)" % (urlparse(s.llm_url).port or 8081))
    print("Stop the server with Ctrl+C.\n")
    try:
        return subprocess.run(cmd).returncode
    except FileNotFoundError:
        print("\nERROR: llama-server was not found.\n"
              "Put LLAMA_SERVER=C:\\llama.cpp\\b11718\\llama-server.exe in your .env file "
              "(the full path of llama-server.exe).")
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())


    