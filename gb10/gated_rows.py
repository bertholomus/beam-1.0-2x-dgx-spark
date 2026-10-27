#!/usr/bin/env python3
"""gated_rows.py: run gb10/rows.py one row at a time, each only while the server has no other requests.

For a lane that already serves users: no row may mix with their requests, and a row only starts while nobody waits. Before each row: wait until /metrics shows 0 running and 0 waiting requests for QUIET_S
seconds (the row is skipped after GATE_MAX_S). During the row: /metrics every 0.5 s. The row overlapped other traffic when the server
held more requests (running + waiting) than the row has in flight, still held any when the row ended, or finished more requests than
the row sent (vllm:request_success_total, all reasons). An overlapped row is run again, up to RETRIES more times. Each row's DFlash2
counters (before vs after) give its acceptance; the sum over the clean decode rows goes to ACCEPT_OUT.
Env: ROWS (comma list), REPS (3), OUT (rows json), ACCEPT_OUT (acceptance json), QUIET_S (20), GATE_MAX_S (1800), RETRIES (2),
PORT, API_KEY_FILE; the rest of the environment passes to rows.py."""
import json, os, subprocess, sys, tempfile, threading, time, urllib.request

PORT = os.environ.get("PORT", "8000")
KEY = os.environ.get("API_KEY") or open(os.environ.get("API_KEY_FILE", os.path.expanduser("~/beam-serve/api_key"))).read().strip()
ROWS = [r for r in os.environ.get("ROWS", "").split(",") if r]
REPS = int(os.environ.get("REPS", "3")); OUT = os.environ.get("OUT", ""); ACCEPT_OUT = os.environ.get("ACCEPT_OUT", "")
QUIET_S = float(os.environ.get("QUIET_S", "20")); GATE_MAX_S = float(os.environ.get("GATE_MAX_S", "1800"))
RETRIES = int(os.environ.get("RETRIES", "2"))
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting", "vllm:request_success": "done",
         "vllm:spec_decode_num_drafts": "drafts", "vllm:spec_decode_num_accepted_tokens": "accepted",
         "vllm:spec_decode_num_draft_tokens": "drafted"}


def sent(row):   # (requests the row sends, incl. rows.py's "Say hi." warm-up; most it has in flight at once)
    if row.startswith("decode1_"): return 2 + 2 * REPS, 1
    if row.startswith("decode4_"): return 5 + 8 * REPS, 4
    if row.startswith("prefill_"): return 1 + REPS, 1
    if row == "ttft_short": return 6, 1
    if row == "ttft_behind_cold46k": return 3, 2
    raise SystemExit(f"unknown row {row}")


def metrics():
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/metrics", headers={"Authorization": f"Bearer {KEY}"})
    m = dict.fromkeys(NAMES.values(), 0.0)
    for line in urllib.request.urlopen(r, timeout=10).read().decode().splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, val = line.rsplit(" ", 1); base = name.split("{", 1)[0]
        base = base[:-6] if base.endswith("_total") else base
        if base in NAMES:
            m[NAMES[base]] += float(val)
    return m


def log(msg):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def wait_quiet():
    t0 = time.time(); since = None
    while time.time() - t0 < GATE_MAX_S:
        m = metrics()
        if m["running"] == 0 and m["waiting"] == 0:
            since = since or time.time()
            if time.time() - since >= QUIET_S:
                return round(time.time() - t0)
        else:
            since = None
        time.sleep(1)
    return None


def run_row(row):
    n_sent, n_flight = sent(row)
    fd, tmp = tempfile.mkstemp(suffix=".json"); os.close(fd)
    peak = [0.0]; stop = threading.Event()

    def watch():
        while not stop.is_set():
            try:
                m = metrics(); peak[0] = max(peak[0], m["running"] + m["waiting"])
            except Exception:
                pass
            stop.wait(0.5)
    m0 = metrics(); th = threading.Thread(target=watch, daemon=True); th.start()
    rc = subprocess.run([sys.executable, os.path.join(HERE, "rows.py")], env=dict(os.environ, ROWS=row, OUT=tmp)).returncode
    stop.set(); th.join()
    for _ in range(10):   # the gauges update with the engine's next step: give them up to 5 s to settle
        m1 = metrics()
        if m1["running"] + m1["waiting"] == 0:
            break
        time.sleep(0.5)
    try:
        res = json.load(open(tmp)).get(row) or {"error": f"rows.py rc={rc}, no result"}
    except Exception:
        res = {"error": f"rows.py rc={rc}, no result"}
    os.unlink(tmp)
    done = round(m1["done"] - m0["done"]); left = m1["running"] + m1["waiting"]
    g = {"overlap": peak[0] > n_flight or left > 0 or done > n_sent, "peak_requests_in_server": peak[0], "own_in_flight": n_flight,
         "requests_finished": done, "requests_sent": n_sent, "left_in_server": left,
         "spec": {k: round(m1[k] - m0[k]) for k in ("drafts", "accepted", "drafted")}}
    if g["spec"]["drafts"]:
        g["mean_acceptance_length"] = round(1 + g["spec"]["accepted"] / g["spec"]["drafts"], 3)
    return res, g


results = {}; acc = dict(drafts=0, accepted=0, drafted=0); clean = []
for row in ROWS:
    for attempt in range(1, RETRIES + 2):
        waited = wait_quiet()
        if waited is None:
            log(f"{row}: never {QUIET_S:.0f} s without other requests within {GATE_MAX_S:.0f} s, skipped")
            results[row] = {"skipped": "server busy"}; break
        res, g = run_row(row); g.update(attempt=attempt, waited_for_quiet_s=waited)
        log(f"{row} attempt {attempt}: {json.dumps(res)} gate {json.dumps(g)}")
        results[row] = dict(res, gate=g)
        if not g["overlap"]:
            clean.append(row)
            if row.startswith("decode"):
                for k in acc:
                    acc[k] += g["spec"][k]
            break
    if OUT:
        json.dump(results, open(OUT, "w"), indent=1)
if ACCEPT_OUT and acc["drafts"]:
    a = {"drafts": acc["drafts"], "accepted_tokens": acc["accepted"], "draft_tokens": acc["drafted"],
         "mean_acceptance_length": round(1 + acc["accepted"] / acc["drafts"], 3),
         "draft_acceptance_rate": round(acc["accepted"] / acc["drafted"], 3) if acc["drafted"] else None,
         "rows": [r for r in clean if r.startswith("decode")]}
    json.dump(a, open(ACCEPT_OUT, "w")); log(f"acceptance over the clean decode rows {json.dumps(a)}")
log(f"GATED_ROWS_DONE clean {len(clean)}/{len(ROWS)}: {','.join(clean)}")
