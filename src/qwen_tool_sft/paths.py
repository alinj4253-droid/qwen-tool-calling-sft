"""Project-local path resolution.

Every data / model / run artifact MUST live inside PROJECT_ROOT
(/mnt/ssd2/psf/job/qwen-tool-calling-sft). No personal absolute paths
(macOS-style home dirs, other users' home dirs, etc.) are ever used for writes.
"""
from __future__ import annotations

from pathlib import Path

# src/qwen_tool_sft/paths.py -> parents[2] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = PROJECT_ROOT / "eval"
RUNS_DIR = PROJECT_ROOT / "runs"
MODELS_DIR = PROJECT_ROOT / "models_local"
CONFIGS_DIR = PROJECT_ROOT / "configs"
CACHE_DIR = PROJECT_ROOT / ".cache"
HF_CACHE_DIR = CACHE_DIR / "hf"
MS_CACHE_DIR = CACHE_DIR / "modelscope"

REQUIRED_DIRS = [DATA_DIR, EVAL_DIR, RUNS_DIR, MODELS_DIR, CACHE_DIR]


def ensure_project_dirs() -> None:
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def resolve_path(p: str | Path, base: Path | None = None) -> Path:
    """Resolve a user supplied path.

    Relative paths are resolved against PROJECT_ROOT (or ``base``).
    The result MUST stay inside PROJECT_ROOT -- otherwise ValueError.
    """
    p = Path(p)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = ((base or PROJECT_ROOT) / p).resolve()
    root = PROJECT_ROOT.resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError(
            f"Refusing to use path outside project root: {resolved} "
            f"(root={root})"
        )
    return resolved


def model_path(name_or_path: str) -> Path:
    """Resolve a model name/path: existing local path wins, else MODELS_DIR/<name>."""
    p = Path(name_or_path)
    if p.exists():
        return p.resolve()
    return resolve_path(MODELS_DIR / name_or_path)
