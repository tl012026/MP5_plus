"""Download missing Minecraft assets into the Gradle cache.

Skips files that already exist with the right size, retries each file up to
MAX_RETRY times, prints failures and continues. Safe to rerun.

Usage: python3 tools/fetch_mc_assets.py [path/to/assets/indexes/<version>.json]
"""
import hashlib
import json
import os
import socket
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_RETRY = 3
TIMEOUT = 120      # seconds per request; large sound files need a long timeout
WORKERS = 4
FORCE_IPV4 = True  # IPv6 downloads can hang on some HPC networks

if FORCE_IPV4:
    _orig = socket.getaddrinfo
    socket.getaddrinfo = lambda host, port, family=0, *a, **k: _orig(host, port, socket.AF_INET, *a, **k)

gradle_home = os.environ.get("GRADLE_USER_HOME", os.path.expanduser("~/.gradle"))
if len(sys.argv) > 1:
    idx = sys.argv[1]
else:
    cands = []
    for base in (f"{gradle_home}/caches/minecraft/assets/indexes",
                 f"{gradle_home}/caches/forge_gradle/assets/indexes"):
        if os.path.isdir(base):
            cands += [os.path.join(base, f) for f in os.listdir(base) if f.endswith(".json")]
    if not cands:
        sys.exit("Asset index not found. Run the MineDojo build once, or pass the index path.")
    idx = sorted(cands)[0]

root = os.path.dirname(os.path.dirname(idx))  # .../assets
objs = list(json.load(open(idx))["objects"].values())
print(f"index: {idx}\n{len(objs)} files -> {root}/objects", flush=True)


def fetch(item):
    h, size = item["hash"], item["size"]
    path = f"{root}/objects/{h[:2]}/{h}"
    if os.path.exists(path) and os.path.getsize(path) == size:
        return "skip", None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = f"https://resources.download.minecraft.net/{h[:2]}/{h}"
    err = None
    for _ in range(MAX_RETRY):
        try:
            data = urllib.request.urlopen(url, timeout=TIMEOUT).read()
            if hashlib.sha1(data).hexdigest() != h:
                raise ValueError("sha1 mismatch")
            with open(path + ".part", "wb") as f:
                f.write(data)
            os.replace(path + ".part", path)
            return "ok", None
        except Exception as e:  # noqa: BLE001
            err = e
    return "fail", f"{url}  ({err})"


stats = {"skip": 0, "ok": 0, "fail": 0}
fails = []
with ThreadPoolExecutor(WORKERS) as ex:
    futures = [ex.submit(fetch, o) for o in objs]
    for i, fut in enumerate(as_completed(futures), 1):
        status, msg = fut.result()
        stats[status] += 1
        if msg:
            fails.append(msg)
            print(f"[fail] {msg}", flush=True)
        if i % 50 == 0 or i == len(objs):
            print(f"{i}/{len(objs)}  existing={stats['skip']} downloaded={stats['ok']} failed={stats['fail']}", flush=True)

print("all assets present" if not fails else f"{len(fails)} failed; rerun this script")
sys.exit(1 if fails else 0)
