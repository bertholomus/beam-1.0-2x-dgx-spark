#!/usr/bin/env python3
"""probe5.py: the five checked prompts for the Beam lane (2 prose, 2 code, 1 tool call), temperature 0.
Each reply is saved whole (content, reasoning, tool calls, usage, wall time) to OUT/probe-<n>.json and checked:
  prose: the final answer holds the expected fact; code: the returned function passes tests (run in a throwaway container with
  no network); tool: one get_weather call with city Paris and unit celsius (beam tool parser). Every reply must carry reasoning.
Env: PORT (8000), MODEL (beam-1.0), API_KEY or API_KEY_FILE, EFFORT (reasoning_effort, default low), OUT (dir), IMAGE (code tests).
Exit 0 only when all five pass."""
import json, os, re, subprocess, sys, time, urllib.request

PORT = os.environ.get("PORT", "8000"); MODEL = os.environ.get("MODEL", "beam-1.0")
KEY = os.environ.get("API_KEY") or open(os.environ.get("API_KEY_FILE", os.path.expanduser("~/beam-serve/api_key"))).read().strip()
EFFORT = os.environ.get("EFFORT", "low"); OUT = os.environ.get("OUT", os.path.expanduser("~/beam-serve/probe"))
IMAGE = os.environ.get("IMAGE", "beam-exl3-gb10"); MAXTOK = int(os.environ.get("MAXTOK", "6000"))
os.makedirs(OUT, exist_ok=True)

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city", "unit"]}}}

PROBES = [
    ("prose-1", "In three sentences, explain why the daytime sky looks blue.", None,
     lambda c: bool(re.search(r"scatter", c, re.I)) and bool(re.search(r"wavelength|blue light|shorter", c, re.I)), "mentions scattering of shorter wavelengths"),
    ("prose-2", "A train leaves at 3:15 PM and the trip takes 2 hours and 50 minutes. At what time does it arrive? "
                "End your answer with the time in the form H:MM PM.", None,
     lambda c: "6:05" in c, "arrives 6:05 PM"),
    ("code-1", "Write a Python function is_prime(n) that returns True if the integer n is prime and False otherwise "
               "(n may be negative, 0 or 1). Reply with one Python code block and nothing else.", None,
     "is_prime", "passes is_prime tests"),
    ("code-2", "Write a Python function fib(n) that returns the n-th Fibonacci number iteratively, with fib(0) = 0 and fib(1) = 1. "
               "Reply with one Python code block and nothing else.", None,
     "fib", "passes fib tests"),
    ("tool-1", "What is the weather in Paris right now? Use celsius.", [WEATHER], None, "calls get_weather(Paris, celsius)"),
]
TESTS = {
    "is_prime": "assert [n for n in range(-5, 40) if is_prime(n)] == [2,3,5,7,11,13,17,19,23,29,31,37]\nassert is_prime(7919) and not is_prime(7917)\nprint('PASS')",
    "fib": "assert [fib(i) for i in range(11)] == [0,1,1,2,3,5,8,13,21,34,55]\nassert fib(50) == 12586269025\nprint('PASS')",
}


def chat(msgs, tools):
    body = {"model": MODEL, "messages": msgs, "temperature": 0, "max_tokens": MAXTOK, "reasoning_effort": EFFORT}
    if tools:
        body["tools"] = tools; body["tool_choice"] = "auto"
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t = time.time(); d = json.load(urllib.request.urlopen(req, timeout=3600)); return d, time.time() - t


def run_code(src, test):
    prog = src + "\n\n" + test + "\n"
    p = subprocess.run(["docker", "run", "--rm", "-i", "--network", "none", "--entrypoint", "python3", IMAGE, "-"],
                       input=prog.encode(), capture_output=True, timeout=120)
    return p.returncode == 0 and b"PASS" in p.stdout, (p.stdout + p.stderr).decode()[-400:]


ok_all = True
for i, (name, prompt, tools, check, want) in enumerate(PROBES, 1):
    d, wall = chat([{"role": "user", "content": prompt}], tools)
    m = d["choices"][0]["message"]; content = m.get("content") or ""; reasoning = m.get("reasoning") or m.get("reasoning_content") or ""
    calls = m.get("tool_calls") or []
    if name.startswith("tool"):
        args = {}
        try:
            args = json.loads(calls[0]["function"]["arguments"]) if calls else {}
        except Exception:
            pass
        passed = len(calls) == 1 and calls[0]["function"]["name"] == "get_weather" and \
            str(args.get("city", "")).lower().startswith("paris") and args.get("unit") == "celsius"
        note = json.dumps([c["function"] for c in calls])[:300]
    elif isinstance(check, str):
        blocks = re.findall(r"```(?:python)?\n(.*?)```", content, re.S)
        src = blocks[0] if blocks else content
        passed, note = run_code(src, TESTS[check])
    else:
        passed, note = check(content), content[-200:]
    passed = passed and len(reasoning.strip()) > 0
    ok_all &= passed
    u = d.get("usage", {})
    rec = {"probe": name, "prompt": prompt, "want": want, "pass": passed, "wall_s": round(wall, 2), "usage": u,
           "finish_reason": d["choices"][0].get("finish_reason"), "reasoning_chars": len(reasoning), "content": content,
           "reasoning": reasoning, "tool_calls": calls, "check_note": note}
    json.dump(rec, open(f"{OUT}/probe-{i}-{name}.json", "w"), indent=1)
    print(f"{name}: {'PASS' if passed else 'FAIL'} ({want}) wall {wall:.1f}s completion {u.get('completion_tokens')} tok, "
          f"reasoning {len(reasoning)} chars, finish {rec['finish_reason']}", flush=True)
print("PROBES_PASS 5/5" if ok_all else "PROBES_FAIL", flush=True)
sys.exit(0 if ok_all else 1)
