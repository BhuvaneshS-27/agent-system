"""Checks the .env file without printing any secret value:  python scripts/check_env.py
Shows which lines python-dotenv can read, which it rejects (and why that may be), which
variables are empty, and how the variable names differ from .env.example."""
import io
import sys
from pathlib import Path

from dotenv.parser import parse_stream

ROOT = Path(__file__).resolve().parent.parent
path = ROOT / ".env"
if not path.exists():
    sys.exit(f"No .env file at {path}. Create it with:  Copy-Item .env.example .env")

raw = path.read_bytes()
if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
    sys.exit("The file is saved as UTF-16. Re-save it as UTF-8 (VS Code: bottom right, 'Save with Encoding').")
text = raw.decode("utf-8-sig")
if raw.startswith(b"\xef\xbb\xbf"):
    print("note: the file starts with a UTF-8 BOM (harmless here, but UTF-8 without BOM is cleaner)\n")


def names(src: str) -> set:
    """Variable names on valid NAME=value lines."""
    return {b.key for b in parse_stream(io.StringIO(src))
            if b.key is not None and not b.error and "=" in b.original.string}


problems = 0
for b in parse_stream(io.StringIO(text)):
    line_no, line = b.original.line, b.original.string.rstrip("\r\n")
    if b.error or (b.key is not None and "=" not in line):
        problems += 1
        hints = []
        if "=" not in line:
            hints.append("no '=' sign")
        if ":" in line.split("=")[0]:
            hints.append("':' where '=' should be")
        if " " in line.split("=")[0].strip():
            hints.append("spaces inside the variable name")
        if line.lstrip().startswith(("-", "*", ">")):
            hints.append("starts with a list bullet")
        if line.count('"') % 2 or line.count("'") % 2:
            hints.append("unbalanced quote")
        print(f"line {line_no:>2}: REJECTED  ({len(line)} characters; {', '.join(hints) or 'unknown format'})")
    elif b.key is not None:
        print(f"line {line_no:>2}: {b.key:<22} {'set' if b.value else 'EMPTY'}")

print(f"\n{problems} rejected line(s)." + ("  Fix them so each one reads NAME=value" if problems else "  All lines are valid."))

example = ROOT / ".env.example"
if example.exists():
    want, have = names(example.read_text(encoding="utf-8-sig")), names(text)
    missing, extra = sorted(want - have), sorted(have - want)
    print("\nCompared with .env.example:")
    print("  missing in .env :", ", ".join(missing) or "none")
    print("  only in .env    :", ", ".join(extra) or "none",
          "  (remove them if they are obsolete)" if extra else "")