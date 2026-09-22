"""Validate the handcrafted eval set structure and coverage."""
from qwen_tool_sft.dataio import iter_jsonl, validate_tool_def

REQUIRED_CATEGORIES = {
    "single_tool", "multi_tool_choice", "argument_filling", "no_tool",
    "wrong_tool_trap", "multi_argument", "parallel_calls",
}


def _load(project_root):
    return list(iter_jsonl(project_root / "eval" / "tool_calling_eval.jsonl"))


def test_eval_overall_shape(project_root):
    cases = _load(project_root)
    assert len(cases) >= 30
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate ids"
    cats = {c["category"] for c in cases}
    assert REQUIRED_CATEGORIES <= cats


def test_eval_case_fields(project_root):
    for c in _load(project_root):
        assert c.get("id") and c.get("category")
        msgs = c["messages"]
        assert msgs and msgs[0]["role"] == "user" and isinstance(msgs[0]["content"], str)
        tools = c.get("tools")
        assert tools is None or (isinstance(tools, list) and all(validate_tool_def(t) for t in tools))
        exp = c["expect"]
        assert exp["mode"] in ("tool", "no_tool")
        assert isinstance(exp.get("calls"), list)
        if exp["mode"] == "tool":
            assert tools, f"{c['id']}: tool case needs tools"
            assert exp["calls"]
            names = {t["function"]["name"] for t in tools}
            for call in exp["calls"]:
                assert call["name"] in names, f"{c['id']}: {call['name']} not in tools"
                assert isinstance(call["arguments"], dict)
        if c["category"] == "no_tool":
            assert exp["mode"] == "no_tool" and tools is None
        if c["category"] == "wrong_tool_trap":
            assert exp["mode"] == "no_tool" and tools


def test_eval_balance(project_root):
    cases = _load(project_root)
    tool = [c for c in cases if c["expect"]["mode"] == "tool"]
    no_tool = [c for c in cases if c["expect"]["mode"] == "no_tool"]
    assert len(tool) >= 20
    assert len(no_tool) >= 8
    traps = [c for c in cases if c["category"] == "wrong_tool_trap"]
    assert len(traps) >= 3
    parallel = [c for c in cases if c["category"] == "parallel_calls"]
    assert all(len(c["expect"]["calls"]) >= 2 for c in parallel)
