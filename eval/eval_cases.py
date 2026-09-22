#!/usr/bin/env python3
"""Handcrafted tool-calling evaluation set (36 bilingual cases).

Categories:
  single_tool        单工具调用
  multi_tool_choice  多工具选择（含干扰工具）
  argument_filling   参数填充
  no_tool            无需工具（纯对话/知识题，无工具）
  wrong_tool_trap    错误工具诱导（提供工具但不应调用）
  multi_argument     多参数工具
  parallel_calls     单轮多工具调用

Ground-truth arguments are deliberately objective (numbers, dates, codes,
enums, booleans, quoted strings) so scoring is deterministic.

Regenerate:
  python eval/eval_cases.py --write eval/tool_calling_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fn(name: str, description: str, properties: dict, required: list) -> dict:
    return {"type": "function",
            "function": {"name": name, "description": description,
                         "parameters": {"type": "object", "properties": properties,
                                        "required": required}}}


WEATHER = fn("get_weather", "查询指定城市的天气", {
    "city": {"type": "string", "description": "城市名称"},
    "date": {"type": "string", "description": "日期，格式 YYYY-MM-DD"},
    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "温度单位"}},
    ["city"])

CALCULATOR = fn("calculator", "执行数学计算，输入一个数学表达式", {
    "expression": {"type": "string", "description": "数学表达式，例如 1+2*3"}}, ["expression"])

SEARCH_WEB = fn("search_web", "在网上搜索信息", {
    "query": {"type": "string", "description": "搜索关键词"},
    "max_results": {"type": "integer", "description": "返回结果数量"}}, ["query"])

TRANSLATE = fn("translate", "把文本翻译成目标语言", {
    "text": {"type": "string", "description": "待翻译文本"},
    "target_language": {"type": "string", "enum": ["zh", "en", "ja", "fr"],
                        "description": "目标语言 ISO 代码"}},
    ["text", "target_language"])

CALENDAR = fn("create_calendar_event", "创建日历事件", {
    "title": {"type": "string", "description": "事件标题"},
    "date": {"type": "string", "description": "日期 YYYY-MM-DD"},
    "start_time": {"type": "string", "description": "开始时间 HH:MM"},
    "end_time": {"type": "string", "description": "结束时间 HH:MM"},
    "location": {"type": "string", "description": "地点"}},
    ["title", "date", "start_time", "end_time"])

REMINDER = fn("set_reminder", "创建一个提醒", {
    "text": {"type": "string", "description": "提醒内容"},
    "time": {"type": "string", "description": "提醒时间"}}, ["text", "time"])

EMAIL = fn("send_email", "发送电子邮件", {
    "to": {"type": "string", "description": "收件人邮箱"},
    "subject": {"type": "string", "description": "邮件主题"},
    "body": {"type": "string", "description": "邮件正文"}}, ["to", "subject", "body"])

CURRENCY = fn("convert_currency", "按汇率换算货币金额", {
    "amount": {"type": "number", "description": "金额"},
    "from_currency": {"type": "string", "description": "源货币 ISO 代码，如 USD"},
    "to_currency": {"type": "string", "description": "目标货币 ISO 代码，如 CNY"}},
    ["amount", "from_currency", "to_currency"])

STOCK = fn("get_stock_price", "查询股票当前价格", {
    "symbol": {"type": "string", "description": "股票代码，如 AAPL"}}, ["symbol"])

SEARCH_FILES = fn("search_files", "在目录中搜索匹配文件名模式的文件", {
    "directory": {"type": "string", "description": "搜索目录"},
    "pattern": {"type": "string", "description": "文件名 glob 模式"},
    "recursive": {"type": "boolean", "description": "是否递归搜索子目录"}},
    ["directory", "pattern", "recursive"])

NEWS = fn("get_news", "获取新闻头条", {
    "category": {"type": "string", "enum": ["tech", "sports", "finance", "entertainment"]},
    "count": {"type": "integer", "description": "新闻条数"}}, ["category", "count"])

FLIGHT = fn("book_flight", "预订机票", {
    "origin": {"type": "string", "description": "出发城市或机场代码"},
    "destination": {"type": "string", "description": "到达城市或机场代码"},
    "date": {"type": "string", "description": "出发日期 YYYY-MM-DD"},
    "passengers": {"type": "integer", "description": "乘客人数"},
    "class": {"type": "string", "enum": ["economy", "business", "first"], "description": "舱位"}},
    ["origin", "destination", "date", "passengers", "class"])


def c(cid, category, content, tools, expect):
    return {"id": cid, "category": category,
            "messages": [{"role": "user", "content": content}],
            "tools": tools, "expect": expect}


def call(name, **args):
    return {"name": name, "arguments": args}


CASES = [
    # ---------------- single tool (6) ----------------
    c("st-01", "single_tool", "请帮我查一下北京 2026-10-01 的天气。", [WEATHER],
      {"mode": "tool", "calls": [call("get_weather", city="北京", date="2026-10-01")]}),
    c("st-02", "single_tool", "What's the weather like in Tokyo?", [WEATHER],
      {"mode": "tool", "calls": [call("get_weather", city="Tokyo")]}),
    c("st-03", "single_tool", "请用计算器计算表达式 123*456 的结果。", [CALCULATOR],
      {"mode": "tool", "calls": [call("calculator", expression="123*456")]}),
    c("st-04", "single_tool", "What is the current stock price of Apple? The ticker symbol is AAPL.",
      [STOCK], {"mode": "tool", "calls": [call("get_stock_price", symbol="AAPL")]}),
    c("st-05", "single_tool", "请把“你好世界”翻译成英语，目标语言代码填 en。", [TRANSLATE],
      {"mode": "tool", "calls": [call("translate", text="你好世界", target_language="en")]}),
    c("st-06", "single_tool",
      "Search for files matching report.pdf in the /home/docs directory without recursing into subdirectories.",
      [SEARCH_FILES],
      {"mode": "tool", "calls": [call("search_files", directory="/home/docs",
                                      pattern="report.pdf", recursive=False)]}),

    # ---------------- multi tool choice (6) ----------------
    c("mt-01", "multi_tool_choice", "请用计算器算一下表达式 88/4 等于多少。",
      [WEATHER, CALCULATOR, SEARCH_WEB],
      {"mode": "tool", "calls": [call("calculator", expression="88/4")]}),
    c("mt-02", "multi_tool_choice",
      "Translate the text 'good morning' to Japanese with target language code ja.",
      [WEATHER, TRANSLATE, STOCK],
      {"mode": "tool", "calls": [call("translate", text="good morning", target_language="ja")]}),
    c("mt-03", "multi_tool_choice",
      "请创建一个提醒：提醒内容填“给妈妈打电话”，提醒时间填“明天下午三点”。",
      [EMAIL, REMINDER, CALENDAR],
      {"mode": "tool", "calls": [call("set_reminder", text="给妈妈打电话", time="明天下午三点")]}),
    c("mt-04", "multi_tool_choice", "Show me the latest 5 technology news articles.",
      [NEWS, STOCK, WEATHER],
      {"mode": "tool", "calls": [call("get_news", category="tech", count=5)]}),
    c("mt-05", "multi_tool_choice",
      "请用货币换算工具把 100 美元换成人民币，源货币代码 USD，目标货币代码 CNY。",
      [CURRENCY, CALCULATOR, STOCK],
      {"mode": "tool", "calls": [call("convert_currency", amount=100,
                                      from_currency="USD", to_currency="CNY")]}),
    c("mt-06", "multi_tool_choice",
      "Search the web for the keyword 'RTX 4090 specs' and return 3 results.",
      [SEARCH_WEB, WEATHER, TRANSLATE],
      {"mode": "tool", "calls": [call("search_web", query="RTX 4090 specs", max_results=3)]}),

    # ---------------- argument filling (8) ----------------
    c("af-01", "argument_filling",
      "查一下上海 2026-09-24 的天气，温度单位用华氏度，unit 参数填 fahrenheit。", [WEATHER],
      {"mode": "tool", "calls": [call("get_weather", city="上海", date="2026-09-24",
                                      unit="fahrenheit")]}),
    c("af-02", "argument_filling", "Convert 250 USD to EUR using the currency conversion tool.",
      [CURRENCY], {"mode": "tool",
                   "calls": [call("convert_currency", amount=250, from_currency="USD",
                                  to_currency="EUR")]}),
    c("af-03", "argument_filling",
      "请给张三发一封邮件：收件人填 zhangsan@example.com，主题填“会议通知”，正文填“请参加周五的会议”。",
      [EMAIL], {"mode": "tool", "calls": [call("send_email", to="zhangsan@example.com",
                                               subject="会议通知", body="请参加周五的会议")]}),
    c("af-04", "argument_filling",
      "Schedule a meeting titled 'Dentist appointment' on 2026-10-05 from 09:00 to 10:00.",
      [CALENDAR], {"mode": "tool", "calls": [call("create_calendar_event",
                                                  title="Dentist appointment", date="2026-10-05",
                                                  start_time="09:00", end_time="10:00")]}),
    c("af-05", "argument_filling",
      "帮我订一张 2026-10-20 从北京到上海的机票，1 位乘客，经济舱（class 填 economy）。", [FLIGHT],
      {"mode": "tool", "calls": [{"name": "book_flight", "arguments": {
          "origin": "北京", "destination": "上海", "date": "2026-10-20",
          "passengers": 1, "class": "economy"}}]}),
    c("af-06", "argument_filling",
      "Set a reminder: text should be 'buy milk' and time should be '2026-09-23 08:00'.",
      [REMINDER], {"mode": "tool", "calls": [call("set_reminder", text="buy milk",
                                                  time="2026-09-23 08:00")]}),
    c("af-07", "argument_filling", "给我看 3 条财经新闻，category 参数填 finance。", [NEWS],
      {"mode": "tool", "calls": [call("get_news", category="finance", count=3)]}),
    c("af-08", "argument_filling",
      "Find all .py files under /work/project, including subdirectories (recursive=true).",
      [SEARCH_FILES], {"mode": "tool", "calls": [call("search_files", directory="/work/project",
                                                      pattern="*.py", recursive=True)]}),

    # ---------------- no tool (6, tools=null) ----------------
    c("nt-01", "no_tool", "你好，请用一句话介绍一下你自己。", None, {"mode": "no_tool", "calls": []}),
    c("nt-02", "no_tool", "What is the capital city of France?", None,
      {"mode": "no_tool", "calls": []}),
    c("nt-03", "no_tool", "请简单解释一下什么是光合作用。", None, {"mode": "no_tool", "calls": []}),
    c("nt-04", "no_tool", "Write a short haiku about autumn.", None,
      {"mode": "no_tool", "calls": []}),
    c("nt-05", "no_tool", "1 加 1 等于几？", None, {"mode": "no_tool", "calls": []}),
    c("nt-06", "no_tool", "Give me one practical tip for better sleep.", None,
      {"mode": "no_tool", "calls": []}),

    # ---------------- wrong tool trap (4, tools present but must NOT call) --
    c("wt-01", "wrong_tool_trap", "我最近有点焦虑，你能安慰我一下吗？",
      [WEATHER, CALCULATOR], {"mode": "no_tool", "calls": []}),
    c("wt-02", "wrong_tool_trap", "Tell me a short joke about programmers.",
      [STOCK, CURRENCY], {"mode": "no_tool", "calls": []}),
    c("wt-03", "wrong_tool_trap", "请写一首关于春天的五言绝句。",
      [SEARCH_FILES, FLIGHT], {"mode": "no_tool", "calls": []}),
    c("wt-04", "wrong_tool_trap", "What do you think is the meaning of life? Answer in one sentence.",
      [NEWS, WEATHER], {"mode": "no_tool", "calls": []}),

    # ---------------- multi argument (4) ----------------
    c("ma-01", "multi_argument",
      "Book a flight from NYC to LAX on 2026-11-11 for 2 passengers in business class (class=business).",
      [FLIGHT], {"mode": "tool", "calls": [{"name": "book_flight", "arguments": {
          "origin": "NYC", "destination": "LAX", "date": "2026-11-11",
          "passengers": 2, "class": "business"}}]}),
    c("ma-02", "multi_argument",
      "请在日历上创建事件：标题填“课题组组会”，日期填 2026-09-28，开始时间填 19:00，结束时间填 21:00，地点填“口腔医学院305”。",
      [CALENDAR], {"mode": "tool", "calls": [call(
        "create_calendar_event", title="课题组组会", date="2026-09-28",
        start_time="19:00", end_time="21:00", location="口腔医学院305")]}),
    c("ma-03", "multi_argument",
      "Send an email to boss@company.com with subject 'Q3 report' and body 'The Q3 report is attached.'",
      [EMAIL], {"mode": "tool", "calls": [call("send_email", to="boss@company.com",
                                               subject="Q3 report",
                                               body="The Q3 report is attached.")]}),
    c("ma-04", "multi_argument",
      "请用搜索工具搜索，query 参数填“4090 LoRA 训练”，max_results 填 5。", [SEARCH_WEB],
      {"mode": "tool", "calls": [call("search_web", query="4090 LoRA 训练", max_results=5)]}),

    # ---------------- parallel calls (2) ----------------
    c("pc-01", "parallel_calls", "请同时查一下北京和上海在 2026-10-01 的天气。", [WEATHER],
      {"mode": "tool", "calls": [call("get_weather", city="北京", date="2026-10-01"),
                                  call("get_weather", city="上海", date="2026-10-01")]}),
    c("pc-02", "parallel_calls",
      "Compare the current stock prices of Apple (AAPL) and Microsoft (MSFT); call the stock tool once for each.",
      [STOCK], {"mode": "tool", "calls": [call("get_stock_price", symbol="AAPL"),
                                          call("get_stock_price", symbol="MSFT")]}),
]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default="")
    args = ap.parse_args()
    text = "\n".join(json.dumps(c, ensure_ascii=False) for c in CASES) + "\n"
    if args.write:
        p = Path(args.write)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        print(f"wrote {len(CASES)} cases -> {p}")
    else:
        print(text)


if __name__ == "__main__":
    main()
