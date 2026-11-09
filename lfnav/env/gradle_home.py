"""Copy a read-only Gradle master cache onto node-local disk.

This module is stdlib-only so compute-node hosts can run it as a script
before singularity starts. Jobs must not write back to the master.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path


def dir_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def copy_dest(job_id, worker_id=None, tmpdir=None) -> Path:
    base = tmpdir or os.environ.get("SLURM_TMPDIR") or "/tmp"
    job = str(job_id)
    if worker_id is None or str(worker_id) == "":
        return Path(base) / f"gradle_{job}"
    return Path(base) / f"gradle_{job}_w{worker_id}"


def copy_gradle_home(src, dst) -> dict:
    src = Path(src)
    dst = Path(dst)
    if not src.is_dir():
        raise FileNotFoundError(f"Gradle master is missing: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for root, dirs, files in os.walk(src):
        dirs[:] = [name for name in dirs if name != "daemon"]
        rel = Path(root).relative_to(src)
        dest_dir = dst / rel
        dest_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            if name.endswith(".lock"):
                continue
            shutil.copy2(Path(root) / name, dest_dir / name)
    elapsed = time.perf_counter() - t0
    size = dir_bytes(dst)
    print(
        f"gradle copy src={src} dst={dst} elapsed_s={elapsed:.3f} size_bytes={size}",
        flush=True,
    )
    return {"src": str(src), "dst": str(dst), "elapsed_s": elapsed, "size_bytes": size}


def lock_path(job_id=None) -> Path:
    job = job_id or os.environ.get("SLURM_JOB_ID") or "local"
    home = os.environ.get("GRADLE_USER_HOME")
    if home:
        parent = Path(home).resolve().parent
    else:
        parent = Path(os.environ.get("SLURM_TMPDIR") or "/tmp")
    return parent / f"gradle_make_{job}.lock"


class EnvCreateLock:
    def __init__(self, path):
        self.path = Path(path)
        self.fd = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None
        return False


def worker_copies_enabled(cfg=None) -> bool:
    env = os.environ.get("GRADLE_WORKER_COPIES", "").strip().lower()
    if env in ("1", "true", "yes"):
        return True
    if env in ("0", "false", "no"):
        return False
    if cfg:
        return bool((cfg.get("gradle") or {}).get("worker_copies", False))
    return False


@contextmanager
def env_create_guard(cfg=None):
    """Serialize minedojo.make unless each worker has its own Gradle copy."""
    if worker_copies_enabled(cfg):
        yield
        return
    with EnvCreateLock(lock_path()):
        yield


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["copy"])
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args(argv)
    copy_gradle_home(args.src, args.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
