"""All write paths must stay inside the project directory."""
from pathlib import Path

import pytest

from qwen_tool_sft import paths


def test_project_root_layout(project_root):
    # on the shared server the checkout directory name is fixed
    norm = str(paths.PROJECT_ROOT).replace("\\", "/")
    if "psf/job" in norm:
        assert paths.PROJECT_ROOT.name == "qwen-tool-calling-sft"
    # regardless of machine, the root must contain the project marker files
    for marker in ("AGENTS.md", "configs", "scripts", "src", "tests", "eval"):
        assert (project_root / marker).exists(), marker


def test_default_dirs_inside_root():
    for d in [paths.DATA_DIR, paths.EVAL_DIR, paths.RUNS_DIR,
              paths.MODELS_DIR, paths.CACHE_DIR]:
        assert paths.PROJECT_ROOT in d.parents or d == paths.PROJECT_ROOT


def test_relative_resolves_inside():
    p = paths.resolve_path("data/train.jsonl")
    assert p == (paths.PROJECT_ROOT / "data" / "train.jsonl").resolve()


@pytest.mark.parametrize("bad", [
    "/etc/passwd",
    "/home/wmy/secret",
    "/Users/icecee/qwen35-finetune/data",
    "/mnt/ssd2/psf/job/other_project/x",
    "../outside.txt",
])
def test_outside_paths_rejected(bad):
    with pytest.raises(ValueError):
        paths.resolve_path(bad)


def test_no_hardcoded_personal_paths(project_root):
    forbidden = ["/Users/", "icecee", "/Users/icecee"]
    scan_roots = [project_root / "scripts", project_root / "src"]
    for root in scan_roots:
        for f in root.rglob("*.py"):
            text = f.read_text(encoding="utf-8")
            for token in forbidden:
                assert token not in text, f"{f} contains forbidden token {token}"
