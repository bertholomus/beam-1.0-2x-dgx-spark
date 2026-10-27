#!/usr/bin/env python3
"""maxctx.py: one cold request whose prompt fills the context to just under the served max_model_len, then 8 new
tokens. The prompt is real code from bench/corpus.txt (repeated, rotated, when the context is longer than the corpus) behind a
unique first line, so nothing comes from the prefix cache; it is sized with the server's own tokenizer (/tokenize, chat form):
chars per token from a 300 KB sample, then a binary search between 97 % and 102 % of that estimate, so no /tokenize call is much
larger than the request itself (the API server's memory grows with the text it tokenizes, on the head node).
Streams the reply: prefill time = time to the first streamed token. Prints one JSON line: prompt tokens, prefill s and tok/s,
total s, max_model_len (from /v1/models), finish reason.
A lane with other users: GATE=1 waits until /metrics shows no running or waiting request for QUIET_S (20) s before it starts, and
YIELD_S > 0 aborts the request (prints "yielded") once another request has waited that many seconds behind it.
Env PORT, MODEL, API_KEY|API_KEY_FILE, CORPUS ($RECIPE/bench/corpus.txt), MARGIN (tokens left free, 72), GATE, QUIET_S, GATE_MAX_S (1800), YIELD_S (0 = never)."""
import http.client, json, os, socket, sys, threading, time, urllib.request
PORT = os.environ.get("PORT", "8000"); MODEL = os.environ.get("MODEL", "beam-1.0"); MARGIN = int(os.environ.get("MARGIN", "72"))
GATE = os.environ.get("GATE", "0") == "1"; QUIET_S = float(os.environ.get("QUIET_S", "20"))
GATE_MAX_S = float(os.environ.get("GATE_MAX_S", "1800")); YIELD_S = float(os.environ.get("YIELD_S", "0"))
KEY = os.environ.get("API_KEY") or open(os.environ.get("API_KEY_FILE", os.path.expanduser("~/beam-serve/api_key"))).read().strip()
T = open(os.environ.get("CORPUS", os.path.join(os.environ.get("RECIPE", os.path.expanduser("~/beam-serve/recipe")), "bench/corpus.txt"))).read(); H = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
def call(path, body=None):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode() if body is not None else None, H)
    return json.load(urllib.request.urlopen(r, timeout=7200))
def gauges():                     # requests the server holds: running, waiting
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/metrics", headers={"Authorization": f"Bearer {KEY}"}); g = [0.0, 0.0]
    for line in urllib.request.urlopen(r, timeout=10).read().decode().splitlines():
        base = line.split("{", 1)[0].split(" ", 1)[0]
        if base in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
            g[base.endswith("waiting")] += float(line.rsplit(" ", 1)[1])
    return g
def say(d):
    print(json.dumps(d), flush=True)
if GATE:
    t0 = time.time(); since = None
    while True:
        since = (since or time.time()) if sum(gauges()) == 0 else None
        if since and time.time() - since >= QUIET_S:
            break
        if time.time() - t0 > GATE_MAX_S:
            say({"skipped": f"server never {QUIET_S:.0f} s without requests within {GATE_MAX_S:.0f} s"}); sys.exit(2)
        time.sleep(1)
MAXLEN = next(m["max_model_len"] for m in call("/v1/models")["data"] if m["id"] == MODEL)
target = MAXLEN - 8 - MARGIN      # room for the 8 new tokens
src = f"Session {time.time_ns()}: read the following source files.\n\n"
k = 0
while len(src) < 5.5 * target:     # code runs ~4.5 chars/token; repeat the corpus rotated by a different offset each time
    rot = (k * 7919 * 104729) % len(T); src += T[rot:] + "\n\n" + T[:rot] + "\n\n"; k += 1
tail = "\n\nExplain what this code does, step by step."
count = lambda c: call("/tokenize", {"model": MODEL, "messages": [{"role": "user", "content": src[:c] + tail}]})["count"]
cpt = 300000 / count(300000)       # chars per token of this text
lo, hi = int(0.97 * target * cpt), min(len(src), int(1.02 * target * cpt))
if count(hi) <= target:
    lo, hi = hi, len(src)
while hi - lo > 400:               # binary search on characters
    mid = (lo + hi) // 2
    lo, hi = (mid, hi) if count(mid) <= target else (lo, mid)
while count(lo) > target:          # lo was above the target from the start (estimate off): step down 1 %
    lo = int(lo * 0.99)
body = {"model": MODEL, "messages": [{"role": "user", "content": src[:lo] + tail}], "max_tokens": 8, "temperature": 0, "ignore_eos": True,
        "reasoning_effort": "low", "stream": True, "stream_options": {"include_usage": True}}
conn = http.client.HTTPConnection("127.0.0.1", int(PORT), timeout=7200); yielded = []; done = threading.Event()
def watch():                       # YIELD_S: give way to another user's request waiting behind this prefill
    behind = None
    while not done.wait(2):
        try:
            run, wait = gauges()
        except Exception:
            continue
        behind = (behind or time.time()) if (wait > 0 or run > 1) else None
        if behind and time.time() - behind >= YIELD_S:
            yielded.append(round(time.perf_counter() - t)); conn.sock.shutdown(socket.SHUT_RDWR); return
first = None; usage = {}; finish = None
t = time.perf_counter(); conn.request("POST", "/v1/chat/completions", json.dumps(body).encode(), H)
if YIELD_S > 0:
    threading.Thread(target=watch, daemon=True).start()
try:
    for line in conn.getresponse():
        line = line.strip()
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        d = json.loads(line[6:])
        usage = d.get("usage") or usage
        for ch in d.get("choices", []):
            dl = ch.get("delta", {}); finish = ch.get("finish_reason") or finish
            if first is None and (dl.get("content") or dl.get("reasoning") or dl.get("reasoning_content")):
                first = time.perf_counter() - t
except Exception:
    if not yielded:
        raise
done.set(); w = time.perf_counter() - t; conn.close()
if yielded:
    say({"yielded": f"another request waited {YIELD_S:.0f} s behind this prefill; aborted after {yielded[0]} s", "chars": lo}); sys.exit(3)
pt = usage.get("prompt_tokens")
say({"prompt_tokens": pt, "completion_tokens": usage.get("completion_tokens"), "prefill_s": round(first, 1),
     "prefill_tps": round(pt / first), "total_s": round(w, 1), "max_model_len": MAXLEN, "fill_pct": round(100 * (pt + 8) / MAXLEN, 2),
     "finish": finish})
