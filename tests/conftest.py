import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def tokenizer():
    """Load the project-local base tokenizer once; skip when not downloaded."""
    model_dir = PROJECT_ROOT / "models_local" / "Qwen" / "Qwen3-4B-Base"
    if not (model_dir / "tokenizer_config.json").exists():
        pytest.skip("base model tokenizer not downloaded yet")
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - env without transformers
        pytest.skip(f"transformers unavailable: {exc}")
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    return tok
