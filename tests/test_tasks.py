"""The presets must be valid: Python files compile, and the 'forces a fix' presets really fail first."""
import subprocess
import sys

import pytest

from agents.coding.tasks import TASKS


@pytest.mark.parametrize("name", list(TASKS))
def test_seeded_python_files_compile(name):
    for path, content in TASKS[name].files.items():
        if path.endswith(".py"):
            compile(content, path, "exec")
    for p in TASKS[name].protected:
        assert p in TASKS[name].files


def run(tmp_path, code):
    return subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=30)


def test_hidden_bug_preset_fails_on_the_required_tests(tmp_path):
    preset = TASKS["4b. Hidden bug in a helper (forces a fix)"]
    (tmp_path / "textutil.py").write_text(preset.files["textutil.py"])
    # the three cases the task demands: each one fails with the seeded helper
    for text, expected in (("", 0), ("  two   words ", 2), ("a\nb", 2)):
        r = run(tmp_path, f"from textutil import word_count; assert word_count({text!r}) == {expected}")
        assert r.returncode != 0 and "AssertionError" in r.stderr, text
    assert "'a\\\\nb'" in preset.task or "a\\nb" in preset.task        # the task names the newline case


def test_hidden_bug_is_fixable_with_split(tmp_path):
    (tmp_path / "textutil.py").write_text("def word_count(text):\n    return len(text.split())\n")
    r = run(tmp_path, "from textutil import word_count; assert word_count('') == 0 and word_count('  two   words ') == 2 "
                      "and word_count('a\\nb') == 2; print('ok')")
    assert r.returncode == 0 and "ok" in r.stdout


def test_shell_presets_mention_run_shell():
    assert "run_shell" in TASKS["5. Install a package (needs approval)"].task
    assert "rm -rf" in TASKS["5b. Dangerous command (safety list blocks it)"].task