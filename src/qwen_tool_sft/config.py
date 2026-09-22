"""YAML config loading with CLI overrides and validation."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from qwen_tool_sft import paths


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        # accept "configs/foo.yaml" from project root, bare "foo.yaml",
        # or an absolute path; never blindly prepend configs/ twice.
        for cand in (paths.PROJECT_ROOT / p, paths.CONFIGS_DIR / p,
                     paths.CONFIGS_DIR / p.name):
            if cand.exists():
                p = cand
                break
    if not p.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config {p} must contain a mapping")
    return cfg


def deep_update(base: dict, overrides: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def parse_kv(items: list[str]) -> dict[str, Any]:
    """Parse key=value CLI overrides; values are coerced as YAML scalars."""
    result: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got: {item}")
        k, v = item.split("=", 1)
        result[k.strip()] = yaml.safe_load(v)
    return result


def require_keys(cfg: dict, keys: list[str], name: str = "config") -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise KeyError(f"{name} missing required keys: {missing}")
