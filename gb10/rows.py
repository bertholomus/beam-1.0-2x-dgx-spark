#!/usr/bin/env python3
"""rows.py: the measured rows for the Beam lane, with the recipe's own methods (bench/bench-fixed.py,
bench/bench-sbs.py, bench/prefill-sweep.py).
  decode, 1 stream:  warm the prompt's prefix cache, then pairs of 16 and 400 new tokens (ignore_eos, temperature 0,
                     reasoning_effort low); decode tok/s = 384 / (t400 - t16); REPS pairs, median. Prose = word-soup document +
                     "write a detailed technical explanation of speculative decoding" (bench-sbs.py); code = real code from
                     bench/corpus.txt + "explain what this code does" (bench-fixed.py).
  decode, 4 streams: 4 different prompts, warmed, sent together; 4 x 384 / (t400 - t16), REPS pairs, median (aggregate).
  cold prefill:      fresh prompts (unique seeds, no prefix-cache hit), max_tokens 1, non-streaming: prompt_tokens / wall.
  TTFT:              streaming, time to the first delta (reasoning or content); short chat prompt, and behind a cold prefill.
  max context:       one request with a prompt just under MAX_MODEL_LEN, 8 new tokens.
Env PORT, MODEL, API_KEY|API_KEY_FILE, CORPUS, ROWS (comma list of row names), REPS (3), OUT (json path), SEED (fixed prompts)."""
import json, os, random, statistics as st, sys, threading, time, urllib.request

PORT = os.environ.get("PORT", "8000"); MODEL = os.environ.get("MODEL", "beam-1.0")
KEY = os.environ.get("API_KEY") or open(os.environ.get("API_KEY_FILE", os.path.expanduser("~/beam-serve/api_key"))).read().strip()
T = open(os.environ.get("CORPUS", os.path.join(os.environ.get("RECIPE", os.path.expanduser("~/beam-serve/recipe")), "bench/corpus.txt"))).read()
REPS = int(os.environ.get("REPS", "3")); OUT = os.environ.get("OUT", "")
MAXLEN = int(os.environ.get("MAX_MODEL_LEN", "131072"))
W = "alpha beta gamma delta epsilon zeta eta theta iota kappa system network memory kernel buffer thread packet cluster tensor matrix".split()
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
HDR = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}


def prose(seed, n):   # bench-sbs.py prompt
    r = random.Random(seed)
    return ("Reference document:\n" + " ".join(r.choice(W) for _ in range(int(n * 0.95))) +
            "\n\nIgnore the document above. Write a detailed technical explanation of how speculative decoding works in LLM inference.")


def code(seed, n):    # bench-fixed.py prompt
    off = (seed * 104729) % (len(T) - int(n * 3.7)); return T[off:off + int(n * 3.6)] + "\n\nExplain what this code does, step by step."


KIND = {"prose": prose, "code": code}


def req(p, mt, ignore_eos=True):
    body = {"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": mt, "temperature": 0,
            "ignore_eos": ignore_eos, "reasoning_effort": "low"}
    r = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=HDR)
    t = time.perf_counter(); d = json.load(urllib.request.urlopen(r, timeout=3600)); return time.perf_counter() - t, d["usage"]


def ttft_stream(p, mt=32):
    body = {"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": mt, "temperature": 0, "stream": True,
            "reasoning_effort": "low", "stream_options": {"include_usage": True}}
    r = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=HDR)
    t = time.perf_counter(); first = None; ptok = None
    with urllib.request.urlopen(r, timeout=3600) as resp:
        for line in resp:
            line = line.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                ptok = d["usage"].get("prompt_tokens")
            for ch in d.get("choices", []):
                dl = ch.get("delta", {})
                if first is None and (dl.get("content") or dl.get("reasoning") or dl.get("reasoning_content")):
                    first = time.perf_counter() - t
    return first, ptok


res = {}
seed0 = int(os.environ.get("SEED") or int(time.time()) % 100000 * 10)   # SEED fixes the prompts (same load across boots)


def decode1(kind, n):
    p = KIND[kind](seed0 + n + (1 if kind == "code" else 2), n); req(p, 16)
    v = []
    for _ in range(REPS):
        t16, _ = req(p, 16); t400, u = req(p, 400); v.append(384 / (t400 - t16))
    return {"median_tps": round(st.median(v), 1), "runs": [round(x, 1) for x in v], "prompt_tokens": u["prompt_tokens"]}


def decode4(kind, n):
    ps = [KIND[kind](seed0 + 500 + 13 * i, n) for i in range(4)]
    for p in ps:
        req(p, 16)

    def run4(mt):
        th = [threading.Thread(target=req, args=(ps[i], mt)) for i in range(4)]; t = time.perf_counter()
        [x.start() for x in th]; [x.join() for x in th]; return time.perf_counter() - t
    v = []
    for _ in range(REPS):
        a = run4(16); b = run4(400); v.append(4 * 384 / (b - a))
    return {"median_agg_tps": round(st.median(v), 1), "runs": [round(x, 1) for x in v]}


def prefill(kind, n):
    v = []
    for i in range(REPS):
        t, u = req(KIND[kind](seed0 + 900 + 31 * i + n, n), 1, ignore_eos=False); v.append((u["prompt_tokens"], t))
    return {"median_tps": round(st.median(pt / t for pt, t in v)), "median_ttft_s": round(st.median(t for _, t in v), 2),
            "runs": [{"prompt_tokens": pt, "s": round(t, 2), "tps": round(pt / t)} for pt, t in v]}


def ttft_short():
    v = [ttft_stream(f"Reply with a one-line greeting for user number {seed0 + i}.")[0] for i in range(5)]
    return {"median_s": round(st.median(v), 3), "runs": [round(x, 3) for x in v]}


def ttft_behind_cold(n=46000):
    out = {}

    def bg():
        out["bg"] = req(prose(seed0 + 7777, n), 1, ignore_eos=False)[0]
    th = threading.Thread(target=bg); th.start(); time.sleep(1.0)
    out["short_ttft_s"] = round(ttft_stream("Reply with one short greeting.")[0], 3); th.join()
    out["cold_prefill_s"] = round(out.pop("bg"), 2); out["cold_tokens"] = n
    return out


def maxctx():
    n = int(MAXLEN * 0.92)   # target prompt tokens (code runs ~3.6 chars/token; keeps the prompt under MAX_MODEL_LEN)
    t, u = req(code(seed0 + 4242, n), 8, ignore_eos=True)
    return {"prompt_tokens": u["prompt_tokens"], "s": round(t, 1), "tps": round(u["prompt_tokens"] / t)}


ROWS = {
    "decode1_prose_2k": lambda: decode1("prose", 2000), "decode1_code_2k": lambda: decode1("code", 2000),
    "decode1_prose_46k": lambda: decode1("prose", 46000), "decode1_code_46k": lambda: decode1("code", 46000),
    "decode4_code_46k": lambda: decode4("code", 46000), "decode4_prose_46k": lambda: decode4("prose", 46000),
    "decode4_code_2k": lambda: decode4("code", 2000),
    "prefill_prose_8k": lambda: prefill("prose", 8000), "prefill_code_8k": lambda: prefill("code", 8000),
    "prefill_prose_46k": lambda: prefill("prose", 46000), "prefill_code_46k": lambda: prefill("code", 46000),
    "ttft_short": ttft_short, "ttft_behind_cold46k": ttft_behind_cold, "maxctx": maxctx,
}
want = [r for r in os.environ.get("ROWS", ",".join(ROWS)).split(",") if r]
req("Say hi.", 4)   # warm-up
for name in want:
    t0 = time.time()
    try:
        res[name] = ROWS[name]()
    except Exception as e:
        res[name] = {"error": str(e)[:300]}
    res[name]["row_wall_s"] = round(time.time() - t0, 1)
    print(name, json.dumps(res[name]), flush=True)
    if OUT:
        json.dump(res, open(OUT, "w"), indent=1)
print("ROWS_DONE", flush=True)
