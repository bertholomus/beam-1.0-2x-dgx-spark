#!/usr/bin/env python3
"""hfrange.py URL OUT SIZE [CONN]: download one large public file with CONN parallel HTTP range requests into a preallocated file.
Resumable: progress per segment in OUT.seg (bytes done per segment). Used for AtomicChat's BF16 KLD base for Beam."""
import json, os, sys, threading, time, urllib.request

url, out, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
conn = int(sys.argv[4]) if len(sys.argv) > 4 else 8
seg = -(-size // conn)
state = out + ".seg"
done = json.load(open(state)) if os.path.exists(state) else [0] * conn
fd = os.open(out, os.O_RDWR | os.O_CREAT, 0o644)
os.ftruncate(fd, size)
lock = threading.Lock()


def work(i):
    a, b = i * seg, min(size, (i + 1) * seg)
    for attempt in range(20):
        pos = a + done[i]
        if pos >= b:
            return
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-{b - 1}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                while pos < b:
                    buf = r.read(min(8 << 20, b - pos))
                    if not buf:
                        break
                    os.pwrite(fd, buf, pos)
                    pos += len(buf)
                    with lock:
                        done[i] = pos - a
        except Exception as e:
            print(f"seg {i} attempt {attempt}: {e}", flush=True)
            time.sleep(min(60, 2 ** attempt))


ts = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(conn)]
for t in ts:
    t.start()
t0 = time.time(); last = sum(done)
while any(t.is_alive() for t in ts):
    time.sleep(30)
    with lock:
        cur = sum(done)
        json.dump(done, open(state, "w"))
    print(f"{time.strftime('%H:%M:%S')} {cur / 1e9:.1f}/{size / 1e9:.1f} GB {(cur - last) / 30e6:.0f} MB/s", flush=True)
    last = cur
os.fsync(fd); os.close(fd)
json.dump(done, open(state, "w"))
ok = sum(done) == size
print("DOWNLOAD_DONE" if ok else f"INCOMPLETE {sum(done)}/{size}", flush=True)
sys.exit(0 if ok else 1)
