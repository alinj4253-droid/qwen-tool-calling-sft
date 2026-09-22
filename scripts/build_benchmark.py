#!/usr/bin/env python3
"""Build an INDEPENDENT held-out tool-calling benchmark (task section 9).

The benchmark is generated deterministically from seeded pools and templates
- it is NOT derived from any training source - and covers:

  single_tool            one tool call, 0-3 distractor schemas in context
  multiple_tools         two DIFFERENT tools in one turn
  parallel_same_tool     >=2 calls to the SAME tool with different args
  parallel_diff_tool     >=3 different tools requested simultaneously
  no_tool                no tools exposed; ordinary questions/chitchat
  wrong_tool_trap        tools exposed but the right action is to abstain
  multi_argument         one call with 4-6 arguments
  distractor             8-12 schemas in context, one correct call

Every expected argument value appears verbatim in the user utterance, so
scoring needs no semantic matching beyond metrics.normalize_value.

  python scripts/build_benchmark.py --out eval/benchmark_extended.jsonl --seed 20260922

After building, run scripts/check_contamination.py against the training data.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qwen_tool_sft import paths  # noqa: E402


def tool(name, desc, props, req=None):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props,
                       "required": req or list(props.keys())}}}


T = {
    "weather": tool("get_weather", "Query weather for a city.",
                    {"city": {"type": "string"},
                     "date": {"type": "string", "description": "YYYY-MM-DD"}},
                    req=["city"]),
    "stock": tool("get_stock_price", "Get the latest stock price by ticker symbol.",
                  {"symbol": {"type": "string"}}),
    "email": tool("send_email", "Send an email.",
                  {"to": {"type": "string"}, "subject": {"type": "string"},
                   "body": {"type": "string"}, "cc": {"type": "string"}},
                  req=["to", "subject"]),
    "calendar": tool("create_calendar_event", "Create a calendar event.",
                     {"title": {"type": "string"}, "date": {"type": "string"},
                      "time": {"type": "string"},
                      "duration_minutes": {"type": "integer"},
                      "location": {"type": "string"},
                      "attendees": {"type": "string"}},
                     req=["title", "date", "time", "duration_minutes", "location"]),
    "reminder": tool("set_reminder", "Set a reminder.",
                     {"text": {"type": "string"}, "time": {"type": "string"}}),
    "search": tool("search_web", "Search the web.", {"query": {"type": "string"}}),
    "translate": tool("translate_text", "Translate text.",
                      {"text": {"type": "string"}, "source_lang": {"type": "string"},
                       "target_lang": {"type": "string"}},
                     req=["text", "target_lang"]),
    "currency": tool("convert_currency", "Convert money between currencies.",
                     {"amount": {"type": "number"}, "from_currency": {"type": "string"},
                      "to_currency": {"type": "string"}}),
    "flight": tool("book_flight", "Book a flight.",
                   {"origin": {"type": "string"}, "destination": {"type": "string"},
                    "date": {"type": "string"}, "passengers": {"type": "integer"},
                    "travel_class": {"type": "string", "enum": ["economy", "business"]}}),
    "order": tool("query_order", "Query an e-commerce order by id.",
                  {"order_id": {"type": "string"}}),
    "news": tool("get_news", "Get news articles about a topic.",
                 {"topic": {"type": "string"}, "count": {"type": "integer"}},
                 req=["topic"]),
    "music": tool("play_music", "Play a song.",
                  {"song": {"type": "string"}, "artist": {"type": "string"}}),
    "alarm": tool("set_alarm", "Set an alarm.", {"time": {"type": "string"}}),
    "time": tool("get_world_time", "Get current time for a city/timezone.",
                 {"city": {"type": "string"}}),
    "calc": tool("calculate", "Evaluate a math expression.",
                 {"expression": {"type": "string"}}),
    "todo": tool("add_todo", "Add a todo item.",
                 {"content": {"type": "string"}, "priority": {"type": "string",
                 "enum": ["low", "medium", "high"]}, "due_date": {"type": "string"}},
                 req=["content"]),
    "thermostat": tool("set_temperature", "Set room temperature.",
                       {"room": {"type": "string"}, "temperature": {"type": "integer"}}),
    "food": tool("order_food", "Order food delivery.",
                 {"dish": {"type": "string"}, "restaurant": {"type": "string"},
                  "quantity": {"type": "integer"}, "address": {"type": "string"}}),
    "restaurant": tool("search_restaurant", "Search restaurants.",
                       {"cuisine": {"type": "string"}, "location": {"type": "string"}}),
    "message": tool("send_message", "Send an instant message to a contact.",
                    {"contact": {"type": "string"}, "message": {"type": "string"}}),
    "traffic": tool("get_traffic", "Get traffic conditions between two places.",
                    {"origin": {"type": "string"},
                     "destination": {"type": "string"}}),
    "file": tool("create_file", "Create a text file.",
                 {"filename": {"type": "string"}, "content": {"type": "string"}}),
    "hotel": tool("book_hotel", "Book a hotel.",
                  {"city": {"type": "string"}, "checkin": {"type": "string"},
                   "checkout": {"type": "string"}, "guests": {"type": "integer"},
                   "room_type": {"type": "string", "enum": ["single", "double", "suite"]}}),
    "rate": tool("get_exchange_rate", "Get an exchange rate.",
                 {"from_currency": {"type": "string"},
                  "to_currency": {"type": "string"}}),
}
TOOL_LIST = list(T.values())
TOOLS = {t["function"]["name"]: t for t in TOOL_LIST}

CITIES = [("北京", "Beijing"), ("上海", "Shanghai"), ("广州", "Guangzhou"),
          ("深圳", "Shenzhen"), ("成都", "Chengdu"), ("杭州", "Hangzhou"),
          ("东京", "Tokyo"), ("纽约", "New York"), ("伦敦", "London"),
          ("巴黎", "Paris"), ("悉尼", "Sydney"), ("新加坡", "Singapore"),
          ("柏林", "Berlin"), ("多伦多", "Toronto"), ("首尔", "Seoul")]
SYMBOLS = ["AAPL", "MSFT", "GOOG", "TSLA", "NVDA", "AMZN", "META", "BABA",
           "PDD", "JD", "NFLX", "AMD", "INTC", "CSCO", "ORCL"]
CURRENCIES = ["USD", "CNY", "EUR", "JPY", "GBP", "AUD", "CAD", "SGD", "KRW"]
LANGS = [("中文", "Chinese"), ("英文", "English"), ("日文", "Japanese"),
         ("法文", "French"), ("德语", "German")]
ROOMS = [("客厅", "living room"), ("卧室", "bedroom"), ("书房", "study"),
         ("会议室", "meeting room")]
CUISINES = [("川菜", "Sichuan"), ("粤菜", "Cantonese"), ("日料", "Japanese"),
            ("意大利菜", "Italian"), ("火锅", "hotpot")]
EMAILS = ["alice@example.com", "bob@example.com", "team@example.org",
          "wang@example.cn", "li@example.cn"]
CONTACTS = [("妈妈", "mom"), ("张老师", "Professor Zhang"),
            ("同事小李", "my colleague Li"), ("客服", "customer support")]
DISHES = [("宫保鸡丁", "Kung Pao chicken"), ("牛肉面", "beef noodles"),
          ("披萨", "pizza"), ("寿司", "sushi"), ("沙拉", "salad")]
SONGS = [("晴天", "Sunny Day"), ("Hey Jude", "Hey Jude"),
         ("稻香", "Rice Field"), ("Yesterday", "Yesterday")]
ARTISTS = [("周杰伦", "Jay Chou"), ("Beatles", "the Beatles"),
           ("林俊杰", "JJ Lin"), ("Adele", "Adele")]
TOPICS = ["人工智能", "climate change", "新能源汽车", "space exploration",
          "奥运会", "quantum computing", "世界杯", "renewable energy"]
QUERIES = ["厦门旅游攻略", "Python list sort example", "空气炸锅食谱",
           "how to reset a router", "nearby gyms", "DSLR vs mirrorless",
           "高铁退票规则", "best mechanical keyboards 2026"]
ORDERS = ["A-1029384", "B-5832017", "C-8890245", "D-3307192", "E-6612048"]
DATES = ["2026-05-12", "2026-06-03", "2026-07-18", "2026-08-25",
         "2026-09-09", "2026-10-21", "2026-11-02", "2026-12-15"]
TIMES = ["07:15", "08:30", "09:00", "10:45", "12:20", "14:00",
         "16:30", "18:10", "19:45", "21:00"]
TITLES = [("团队周会", "team weekly meeting"), ("牙医预约", "dentist appointment"),
          ("项目评审", "project review"), ("面试", "job interview")]
TEXTS = ["明天带伞", "给妈妈回电话", "提交报销单", "续费域名",
         "取快递", "打印会议材料"]
EXPRESSIONS = ["(123 + 456) * 2", "980 / 35", "17 * 23 - 59", "2 ** 10",
               "sqrt(144) + 38", "(88 - 19) / 3"]
NO_TOOL_ZH = ["你觉得人活着的意义是什么？", "写一首关于秋天的短诗", "讲一个冷笑话",
              "为什么天空是蓝色的？", "帮我想三个有创意的生日礼物点子",
              "怎么缓解焦虑？", "用一句话解释相对论", "给我讲个睡前故事",
              "面试的时候怎么自我介绍比较好？", "如何坚持每天读书？",
              '“醍醐灌顶”是什么意思？', "帮我润色一句祝福语：生日快乐",
              "人生第一份工作应该怎么选？", "拖延症有什么改善方法？",
              "推荐几个培养专注力的小习惯", "怎么安慰心情不好的朋友？",
              "议论文开头一般怎么写？", "解释一下破窗效应",
              "给我的博客起个名字，主题是摄影", "如何看待人工智能的发展？"]
NO_TOOL_EN = ["What do you think is the meaning of life?",
              "Write a short poem about autumn.", "Tell me a bad joke.",
              "Why is the sky blue?",
              "Give me three creative birthday gift ideas.",
              "How can I deal with anxiety before exams?",
              "Explain relativity in one sentence.", "Tell me a bedtime story.",
              "How should I introduce myself in a job interview?",
              "How do I build a daily reading habit?",
              "What does the phrase 'break the ice' mean?",
              "Polish this greeting: happy birthday!",
              "How should I choose my first job?",
              "What are practical ways to beat procrastination?",
              "Suggest habits that improve focus.",
              "How can I comfort an upset friend?",
              "How do I start an argumentative essay?",
              "Explain the broken windows theory.",
              "Suggest a name for my photography blog.",
              "What is your opinion on the development of AI?"]


def pick(rng, pool):
    return pool[rng.randrange(len(pool))]


def case(cid, category, lang, content, tools, expect_calls):
    return {
        "id": cid, "category": category, "language": lang,
        "messages": [{"role": "user", "content": content}],
        "tools": tools,
        "expect": {"mode": "tool" if expect_calls else "no_tool",
                   "calls": [{"name": n, "arguments": a} for n, a in expect_calls]},
    }


# ---------- intent generators: return (utterance, tools, calls) ----------

def gen_single(rng, lang):
    zh = lang == "zh"
    choice = rng.randrange(12)
    if choice == 0:
        c = pick(rng, CITIES)[0 if zh else 1]
        u = f"帮我查一下{c}的天气" if zh else f"What's the weather like in {c}?"
        return u, [TOOLS["get_weather"]], [("get_weather", {"city": c})]
    if choice == 1:
        s = pick(rng, SYMBOLS)
        u = f"查一下股票{s}现在的价格" if zh else f"Check the current price of {s} stock."
        return u, [TOOLS["get_stock_price"]], [("get_stock_price", {"symbol": s})]
    if choice == 2:
        q = pick(rng, QUERIES)
        u = f"帮我搜索：{q}" if zh else f"Search the web for: {q}"
        return u, [TOOLS["search_web"]], [("search_web", {"query": q})]
    if choice == 3:
        c = pick(rng, CITIES)[0 if zh else 1]
        u = f"{c}现在几点了？" if zh else f"What time is it in {c} right now?"
        return u, [TOOLS["get_world_time"]], [("get_world_time", {"city": c})]
    if choice == 4:
        o = pick(rng, ORDERS)
        u = f"帮我查一下订单{o}的物流状态" if zh else f"Track the status of order {o}."
        return u, [TOOLS["query_order"]], [("query_order", {"order_id": o})]
    if choice == 5:
        e = pick(rng, EXPRESSIONS)
        u = f"帮我计算{e}" if zh else f"Calculate {e} for me."
        return u, [TOOLS["calculate"]], [("calculate", {"expression": e})]
    if choice == 6:
        tm = pick(rng, TIMES)
        u = f"设一个{tm}的闹钟" if zh else f"Set an alarm for {tm}."
        return u, [TOOLS["set_alarm"]], [("set_alarm", {"time": tm})]
    if choice == 7:
        topic = pick(rng, TOPICS)
        u = f"给我看看{topic}相关的新闻" if zh else f"Show me news about {topic}."
        return u, [TOOLS["get_news"]], [("get_news", {"topic": topic})]
    if choice == 8:
        s, a = pick(rng, list(zip(SONGS, ARTISTS)))
        song, artist = s[0 if zh else 1], a[0 if zh else 1]
        u = f"播放{artist}的{song}" if zh else f"Play {song} by {artist}."
        return u, [TOOLS["play_music"]], [("play_music", {"song": song, "artist": artist})]
    if choice == 9:
        cuisine = pick(rng, CUISINES)[0 if zh else 1]
        loc = pick(rng, CITIES)[0 if zh else 1]
        u = f"在{loc}附近找一家{cuisine}餐厅" if zh else f"Find a {cuisine} restaurant near {loc}."
        return u, [TOOLS["search_restaurant"]], [
            ("search_restaurant", {"cuisine": cuisine, "location": loc})]
    if choice == 10:
        target = pick(rng, LANGS)[0 if zh else 1]
        phrase = "会议改到下午三点" if zh else "the meeting is moved to 3 PM"
        u = f"把“{phrase}”翻译成{target}" if zh else f'Translate "{phrase}" into {target}.'
        src = "中文" if zh else "English"
        return u, [TOOLS["translate_text"]], [
            ("translate_text", {"text": phrase, "source_lang": src, "target_lang": target})]
    room = pick(rng, ROOMS)[0 if zh else 1]
    temp = rng.randint(18, 28)
    u = f"把{room}的空调调到{temp}度" if zh else f"Set the {room} thermostat to {temp} degrees."
    return u, [TOOLS["set_temperature"]], [
        ("set_temperature", {"room": room, "temperature": temp})]


def gen_parallel_same(rng, lang):
    zh = lang == "zh"
    kind = rng.randrange(6)
    if kind == 0:
        cs = rng.sample(CITIES, 3)
        vals = [c[0 if zh else 1] for c in cs]
        u = (f"同时帮我查一下{vals[0]}、{vals[1]}和{vals[2]}的天气"
             if zh else f"Give me the weather for {vals[0]}, {vals[1]} and {vals[2]} at once.")
        return u, [TOOLS["get_weather"]], [("get_weather", {"city": v}) for v in vals]
    if kind == 1:
        ss = rng.sample(SYMBOLS, 3)
        u = (f"一次性查一下{ss[0]}、{ss[1]}、{ss[2]}这三只股票的价格"
             if zh else f"Look up the prices of {ss[0]}, {ss[1]} and {ss[2]} in one go.")
        return u, [TOOLS["get_stock_price"]], [("get_stock_price", {"symbol": s}) for s in ss]
    if kind == 2:
        cs = rng.sample(CITIES, 2)
        vals = [c[0 if zh else 1] for c in cs]
        u = (f"分别告诉我{vals[0]}和{vals[1]}现在的时间"
             if zh else f"Tell me the current time in {vals[0]} and {vals[1]} separately.")
        return u, [TOOLS["get_world_time"]], [("get_world_time", {"city": v}) for v in vals]
    if kind == 3:
        ts = rng.sample(TEXTS, 2)
        tms = rng.sample(TIMES, 2)
        u = (f"帮我设两个提醒：{tms[0]}提醒我{ts[0]}，{tms[1]}提醒我{ts[1]}"
             if zh else f"Set two reminders: {tms[0]} for '{ts[0]}' and {tms[1]} for '{ts[1]}'.")
        return u, [TOOLS["set_reminder"]], [
            ("set_reminder", {"text": ts[0], "time": tms[0]}),
            ("set_reminder", {"text": ts[1], "time": tms[1]})]
    if kind == 4:
        qs = rng.sample(QUERIES, 2)
        u = (f"同时搜索这两个问题：{qs[0]} 和 {qs[1]}"
             if zh else f"Run two searches at the same time: {qs[0]} and {qs[1]}.")
        return u, [TOOLS["search_web"]], [("search_web", {"query": q}) for q in qs]
    topics = rng.sample(TOPICS, 2)
    u = (f"各给我一条关于{topics[0]}和{topics[1]}的新闻"
         if zh else f"Get me one news article each on {topics[0]} and {topics[1]}.")
    return u, [TOOLS["get_news"]], [
        ("get_news", {"topic": t}) for t in topics]


def gen_multiple(rng, lang):
    zh = lang == "zh"
    kind = rng.randrange(5)
    if kind == 0:
        c = pick(rng, CITIES)[0 if zh else 1]
        u = (f"查一下{c}现在的时间，再顺便看看天气"
             if zh else f"Check the current time in {c} and also its weather.")
        return u, [TOOLS["get_world_time"], TOOLS["get_weather"]], [
            ("get_world_time", {"city": c}), ("get_weather", {"city": c})]
    if kind == 1:
        q = pick(rng, QUERIES)
        topic = pick(rng, TOPICS)
        u = (f"先搜一下{q}，再找几条{topic}的新闻"
             if zh else f"First search for {q}, then find some news about {topic}.")
        return u, [TOOLS["search_web"], TOOLS["get_news"]], [
            ("search_web", {"query": q}), ("get_news", {"topic": topic})]
    if kind == 2:
        amt = rng.choice([100, 250, 1000, 3500])
        f, t = rng.sample(CURRENCIES, 2)
        q = pick(rng, QUERIES)
        u = (f"帮我把{amt}{f}换成{t}，再搜一下{q}"
             if zh else f"Convert {amt} {f} to {t}, and also search for {q}.")
        return u, [TOOLS["convert_currency"], TOOLS["search_web"]], [
            ("convert_currency", {"amount": amt, "from_currency": f, "to_currency": t}),
            ("search_web", {"query": q})]
    if kind == 3:
        s, a = pick(rng, list(zip(SONGS, ARTISTS)))
        song, artist = s[0 if zh else 1], a[0 if zh else 1]
        tm = pick(rng, TIMES)
        u = (f"放一首{artist}的{song}，然后设一个{tm}的闹钟"
             if zh else f"Play {song} by {artist}, then set an alarm for {tm}.")
        return u, [TOOLS["play_music"], TOOLS["set_alarm"]], [
            ("play_music", {"song": song, "artist": artist}),
            ("set_alarm", {"time": tm})]
    contact = pick(rng, CONTACTS)[0 if zh else 1]
    msg = "我会晚到十分钟" if zh else "I will be ten minutes late"
    content = "整理会议纪要" if zh else "organize meeting notes"
    day = pick(rng, DATES)
    u = (f"给{contact}发消息说{msg}，顺便加个{day}前{content}的待办"
         if zh else f"Message {contact} saying '{msg}', and add a todo to {content} before {day}.")
    return u, [TOOLS["send_message"], TOOLS["add_todo"]], [
        ("send_message", {"contact": contact, "message": msg}),
        ("add_todo", {"content": content, "due_date": day})]


def gen_parallel_diff(rng, lang):
    zh = lang == "zh"
    kind = rng.randrange(3)
    if kind == 0:
        c = pick(rng, CITIES)[0 if zh else 1]
        c2 = pick(rng, CITIES)[0 if zh else 1]
        u = (f"一次性帮我做三件事：查{c}的天气、查{c}的时间、再看下从{c2}到{c}的路况"
             if zh else f"Do three things at once: weather in {c}, time in {c}, "
                       f"and traffic from {c2} to {c}.")
        return u, [TOOLS["get_weather"], TOOLS["get_world_time"], TOOLS["get_traffic"]], [
            ("get_weather", {"city": c}), ("get_world_time", {"city": c}),
            ("get_traffic", {"origin": c2, "destination": c})]
    if kind == 1:
        email = pick(rng, EMAILS)
        content = "准备季度汇报" if zh else "prepare the quarterly report"
        day, tm = pick(rng, DATES), pick(rng, TIMES)
        u = (f"同时：给{email}发一封主题为周报的邮件，加一个{day}前{content}的待办，"
             f"再设个{tm}的提醒提醒我写周报"
             if zh else f"At the same time: email {email} with subject 'weekly report', "
                       f"add a todo to {content} by {day}, and set a reminder at {tm} "
                       f"to write it.")
        return u, [TOOLS["send_email"], TOOLS["add_todo"], TOOLS["set_reminder"]], [
            ("send_email", {"to": email, "subject": "周报" if zh else "weekly report"}),
            ("add_todo", {"content": content, "due_date": day}),
            ("set_reminder", {"text": content, "time": tm})]
    topic = pick(rng, TOPICS)
    q = pick(rng, QUERIES)
    phrase = "欢迎加入新团队" if zh else "welcome to the new team"
    target = pick(rng, LANGS)[0 if zh else 1]
    u = (f"一起处理：搜{topic}的新闻、在网上查{q}、把“{phrase}”翻译成{target}"
         if zh else f"Handle these together: news on {topic}, web search for {q}, "
                   f"and translate '{phrase}' into {target}.")
    return u, [TOOLS["get_news"], TOOLS["search_web"], TOOLS["translate_text"]], [
        ("get_news", {"topic": topic}),
        ("search_web", {"query": q}),
        ("translate_text", {"text": phrase,
                            "source_lang": "中文" if zh else "English",
                            "target_lang": target})]


def gen_multi_argument(rng, lang):
    zh = lang == "zh"
    kind = rng.randrange(4)
    if kind == 0:
        o = pick(rng, CITIES)[0 if zh else 1]
        d = pick(rng, CITIES)[0 if zh else 1]
        date = pick(rng, DATES)
        pax = rng.randint(1, 4)
        cls = rng.choice(["economy", "business"])
        u = (f"帮我订一张{date}从{o}到{d}的机票，{pax}个人，舱位{cls}"
             if zh else f"Book a {cls} flight from {o} to {d} on {date} for {pax} passenger(s).")
        return u, [TOOLS["book_flight"]], [("book_flight", {
            "origin": o, "destination": d, "date": date,
            "passengers": pax, "travel_class": cls})]
    if kind == 1:
        import datetime
        c = pick(rng, CITIES)[0 if zh else 1]
        d1 = pick(rng, DATES)
        d2 = (datetime.date.fromisoformat(d1)
              + datetime.timedelta(days=rng.randint(2, 5))).isoformat()
        guests = rng.randint(1, 4)
        rt = rng.choice(["single", "double", "suite"])
        u = (f"在{c}订酒店，{d1}入住{d2}退房，{guests}位客人，房型{rt}"
             if zh else f"Book a {rt} hotel room in {c}, check-in {d1}, "
                       f"check-out {d2}, for {guests} guest(s).")
        return u, [TOOLS["book_hotel"]], [("book_hotel", {
            "city": c, "checkin": d1, "checkout": d2, "guests": guests,
            "room_type": rt})]
    if kind == 2:
        title = pick(rng, TITLES)[0 if zh else 1]
        date, tm = pick(rng, DATES), pick(rng, TIMES)
        dur = rng.choice([30, 45, 60, 90, 120])
        loc = pick(rng, ROOMS)[0 if zh else 1]
        att = pick(rng, EMAILS)
        u = (f"创建日程：{title}，日期{date}，时间{tm}，时长{dur}分钟，"
             f"地点{loc}，参会人{att}"
             if zh else f"Create an event '{title}' on {date} at {tm}, {dur} minutes, "
                       f"location {loc}, attendee {att}.")
        return u, [TOOLS["create_calendar_event"]], [("create_calendar_event", {
            "title": title, "date": date, "time": tm,
            "duration_minutes": dur, "location": loc, "attendees": att})]
    dish = pick(rng, DISHES)[0 if zh else 1]
    rest = ("川菜馆" if zh else "the downtown bistro")
    qty = rng.randint(1, 5)
    addr = ("中山路12号" if zh else "12 Main Street")
    u = (f"在{rest}点{qty}份{dish}，送到{addr}"
         if zh else f"Order {qty} portion(s) of {dish} from {rest}, delivered to {addr}.")
    return u, [TOOLS["order_food"]], [("order_food", {
        "dish": dish, "restaurant": rest, "quantity": qty, "address": addr})]


def gen_no_tool(rng, lang, trap):
    zh = lang == "zh"
    pool = NO_TOOL_ZH if zh else NO_TOOL_EN
    content = pick(rng, pool)
    if trap:
        # tools ARE available but none is appropriate
        names = rng.sample(list(TOOLS), rng.randint(3, 6))
        return content, [TOOLS[n] for n in names], []
    return content, [], []


def with_distractors(rng, tools_target, n_total):
    others = [t for t in TOOL_LIST if t not in tools_target]
    rng.shuffle(others)
    pool = tools_target + others[: n_total - len(tools_target)]
    rng.shuffle(pool)
    return pool


# ---------- assembly ----------

PLAN = {
    "single_tool": 60,
    "multiple_tools": 30,
    "parallel_same_tool": 50,
    "parallel_diff_tool": 20,
    "multi_argument": 30,
    "no_tool": 34,
    "wrong_tool_trap": 26,
    "distractor": 30,
}


def build(seed: int):
    rng = random.Random(seed)
    cases = []
    counters = {k: 0 for k in PLAN}

    def add(category, lang, content, tools, calls):
        counters[category] += 1
        cid = f"ext-{category}-{lang}-{counters[category]:03d}"
        cases.append(case(cid, category, lang, content, tools, calls))

    for category, n in PLAN.items():
        for i in range(n):
            lang = "zh" if i % 2 == 0 else "en"
            if category == "single_tool":
                u, tl, cl = gen_single(rng, lang)
                extra = rng.sample([t for t in TOOL_LIST if t not in tl], rng.randint(0, 3))
                add(category, lang, u, tl + extra, cl)
            elif category == "multiple_tools":
                u, tl, cl = gen_multiple(rng, lang)
                add(category, lang, u, tl, cl)
            elif category == "parallel_same_tool":
                u, tl, cl = gen_parallel_same(rng, lang)
                add(category, lang, u, tl, cl)
            elif category == "parallel_diff_tool":
                u, tl, cl = gen_parallel_diff(rng, lang)
                add(category, lang, u, tl, cl)
            elif category == "multi_argument":
                u, tl, cl = gen_multi_argument(rng, lang)
                add(category, lang, u, tl, cl)
            elif category == "no_tool":
                u, tl, cl = gen_no_tool(rng, lang, trap=False)
                add(category, lang, u, tl, cl)
            elif category == "wrong_tool_trap":
                u, tl, cl = gen_no_tool(rng, lang, trap=True)
                add(category, lang, u, tl, cl)
            elif category == "distractor":
                u, tl, cl = gen_single(rng, lang)
                tl = with_distractors(rng, tl, rng.randint(8, 12))
                add(category, lang, u, tl, cl)

    rng.shuffle(cases)
    # re-id after shuffle for stable ids
    for i, c in enumerate(cases):
        c["id"] = f"ext-{c['category']}-{c['language']}-{i:04d}"
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/benchmark_extended.jsonl")
    ap.add_argument("--seed", type=int, default=20260922)
    args = ap.parse_args()
    cases = build(args.seed)
    out = paths.resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases),
                   encoding="utf-8")
    by_cat = {}
    for c in cases:
        by_cat[c["category"]] = by_cat.get(c["category"], 0) + 1
    print(f"wrote {len(cases)} cases -> {out}")
    print(json.dumps(by_cat, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
