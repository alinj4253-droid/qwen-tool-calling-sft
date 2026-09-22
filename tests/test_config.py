"""Config files must be valid and reference project-local paths."""
import yaml

from qwen_tool_sft import paths

TRAIN_REQUIRED = [
    "run_name", "model_path", "data_dir", "lora_r", "lora_alpha",
    "learning_rate", "max_seq_length", "per_device_train_batch_size",
    "gradient_accumulation_steps",
]
EVAL_REQUIRED = ["run_name", "model_path", "eval_set"]


def _load(p):
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def test_train_configs(project_root):
    for name in ("train_smoke.yaml", "train_sft_10k.yaml", "train_sft_30k.yaml"):
        cfg = _load(project_root / "configs" / name)
        for k in TRAIN_REQUIRED:
            assert k in cfg, f"{name} missing {k}"
        # data dir must resolve inside project
        paths.resolve_path(cfg["data_dir"])
        assert cfg["lora_alpha"] >= cfg["lora_r"]
        assert cfg["max_seq_length"] <= 4096


def test_eval_configs(project_root):
    for name in ("eval_base.yaml", "eval_sft.yaml"):
        cfg = _load(project_root / "configs" / name)
        for k in EVAL_REQUIRED:
            assert k in cfg, f"{name} missing {k}"
        paths.resolve_path(cfg["eval_set"])
        if cfg.get("adapter_path"):
            paths.resolve_path(cfg["adapter_path"])


def test_data_configs(project_root):
    for name in ("data_smoke.yaml", "data_full.yaml"):
        cfg = _load(project_root / "configs" / name)
        p = paths.resolve_path(cfg["output_dir"])
        assert paths.DATA_DIR == p or paths.DATA_DIR in p.parents


def test_accelerate_config(project_root):
    cfg = _load(project_root / "configs" / "accelerate_dual4090.yaml")
    assert cfg["distributed_type"] == "MULTI_GPU"
    assert cfg["num_processes"] == 2
    assert cfg["mixed_precision"] == "bf16"
    assert cfg["use_cpu"] is False


def test_load_config_path_forms(project_root, monkeypatch):
    from qwen_tool_sft.config import load_config
    monkeypatch.chdir(project_root)
    by_rel = load_config("configs/data_smoke.yaml")
    by_name = load_config("data_smoke.yaml")
    by_abs = load_config(str(project_root / "configs" / "data_smoke.yaml"))
    assert by_rel == by_name == by_abs
    assert by_rel["stream"] is True
