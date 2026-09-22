"""Train/benchmark contamination checks (P0-3)."""
from __future__ import annotations

from qwen_tool_sft.contamination import (
    NearDuplicateIndex, char_shingles, cosine, run_contamination,
    user_query_text,
)


def _bench(qid, query, tools=None, category="single_tool"):
    return {"id": qid, "category": category,
            "messages": [{"role": "user", "content": query}], "tools": tools or []}


def _train(query, tools=None):
    return {"messages": [{"role": "user", "content": query}], "tools": tools or []}


def test_exact_query_match_detected():
    train = [_train("帮我查询北京今天的天气怎么样")]
    bench = [_bench("b1", "帮我查询北京今天的天气怎么样")]
    rep = run_contamination(train, bench, threshold=0.9)
    assert rep["summary"]["n_exact_query_match"] == 1
    assert rep["summary"]["exact_train_duplicate_required_zero"] is False


def test_same_query_different_tools_is_query_dup_but_not_query_tools_dup():
    t_weather = [{"type": "function", "function": {"name": "get_weather"}}]
    t_email = [{"type": "function", "function": {"name": "send_email"}}]
    train = [_train("帮我发一封邮件", t_weather)]
    bench = [_bench("b1", "帮我发一封邮件", t_email)]
    rep = run_contamination(train, bench, threshold=0.9)
    assert rep["summary"]["n_exact_query_match"] == 1
    assert rep["summary"]["n_exact_query_and_tools_match"] == 0


def test_clean_benchmark_passes():
    train = [_train("帮我查询北京今天的天气怎么样"),
             _train("把这段文字翻译成英文")]
    bench = [_bench("b1", "推荐一部适合周末看的科幻电影"),
             _bench("b2", "帮我订一张明天去上海的高铁票")]
    rep = run_contamination(train, bench, threshold=0.8)
    assert rep["summary"]["n_exact_query_match"] == 0
    assert rep["summary"]["n_near_duplicate_cases"] == 0
    assert rep["summary"]["exact_train_duplicate_required_zero"] is True


def test_near_duplicate_paraphrase_flagged():
    train = [_train("请帮我查询一下北京今天的天气情况如何，谢谢")]
    # same sentence with minor edits -> high char 4-gram overlap
    bench_near = _bench("b1", "请帮我查询一下北京今天的天气情况如何谢谢啊")
    bench_far = _bench("b2", "给我讲一个关于太空旅行的科幻故事吧")
    rep = run_contamination(train, [bench_near, bench_far], threshold=0.8)
    assert rep["summary"]["n_near_duplicate_cases"] == 1
    near_ids = {r["bench_id"] for r in rep["near_duplicates"]}
    assert "b1" in near_ids and "b2" not in near_ids


def test_english_punctuation_difference_is_near_duplicate():
    train = [_train("what is the weather like in Beijing today")]
    bench = [_bench("b1", "what is the weather like in Beijing today?")]
    rep = run_contamination(train, bench, threshold=0.8)
    # trailing punctuation: not a canonical exact match, but high-similarity
    assert rep["summary"]["n_near_duplicate_cases"] == 1
    assert rep["near_duplicates"][0]["similarity"] > 0.9


def test_tool_responses_excluded_from_query_text():
    s = {"messages": [
        {"role": "user", "content": "USER_QUERY_X"},
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "tool", "name": "f", "content": "TOOL_ANSWER_Y"}]}
    text = user_query_text(s)
    assert "USER_QUERY_X" in text
    assert "TOOL_ANSWER_Y" not in text


def test_cosine_basic():
    a = char_shingles("帮我查询北京今天的天气")
    b = char_shingles("帮我查询北京今天的天气情况")
    c = char_shingles("完全无关的另一句话关于股票市场")
    assert cosine(a, b) > 0.7
    assert cosine(a, c) < 0.3


def test_index_returns_top_matches():
    train = [_train("查询北京天气"), _train("翻译这段文字"),
             _train("查询北京天气预报十五天")]
    idx = NearDuplicateIndex(train)
    tops = idx.top_matches(_bench("b", "查询北京天气"), top_k=2)
    # exact copy ranks first; the longer same-prefix query is second
    assert len(tops) == 2
    assert tops[0][0] == 0 and abs(tops[0][1] - 1.0) < 1e-6
    assert tops[0][1] >= tops[1][1]
