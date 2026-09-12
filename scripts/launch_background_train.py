#!/usr/bin/env python3
"""Launch background YOLO training as a SINGLE detached process.

Single-instance guarantee: an exclusive lock file plus a recorded trainer PID
prevents two training processes from ever writing to the same ultralytics run
dir. Root-cause fix for the duplicate concurrent trainers that corrupted the
previous drishti-ss_yolov8n_e30 run (4 writers, interrupted at epoch 5).

Usage:
    python scripts/launch_background_train.py             # new 30-epoch run
    python scripts/launch_background_train.py --resume    # resume last.pt
    python scripts/launch_background_train.py --status    # is training running?
"""
from __future__ import annotations

import argparse
import msvcrt
import os
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
python_exe = project_root / ".venv" / "Scripts" / "python.exe"
train_script = project_root / "scripts" / "train_yolo.py"
log_file = project_root / "scripts" / "train_full_log.txt"
lock_file = project_root / "scripts" / ".training.lock"


def _pid_alive(pid: int) -> bool:
    """Windows-safe liveness check via tasklist (no external deps)."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True, text=True, timeout=15,
        ).stdout
        return f" {pid} " in out or out.rstrip().endswith(str(pid))
    except Exception:
        return False


def _trainer_running() -> int | None:
    """Return PID of a live trainer if the pidfile records one."""
    if not lock_file.is_file():
        return None
    raw = lock_file.read_text().strip()
    if not raw.isdigit():
        return None
    pid = int(raw)
    return pid if _pid_alive(pid) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume", action="store_true", help="resume interrupted run from last.pt")
    ap.add_argument("--status", action="store_true", help="only report whether training is running")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="drishti-ss_yolov8n_e30")
    ap.add_argument("--model", default="yolov8n.pt", help="base weights (e.g. yolov8s.pt for the capacity probe)")
    args = ap.parse_args()

    # Read recorded trainer PID *before* locking (the lock itself blocks reads).
    live = _trainer_running()

    # Exclusive launch lock (prevents two launchers racing at the same moment).
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = open(lock_file, "a+")  # noqa: SIM115 — released below in every path
    try:
        msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print(f"REFUSED: another launcher holds {lock_file.name}.")
        return 2

    try:
        if args.status:
            print(f"training running: {'YES (pid ' + str(live) + ')' if live else 'no'}")
            return 0
        if live:
            print(f"REFUSED: trainer already running (pid {live}).")
            print("Multiple concurrent trainers corrupt the run dir — that bug killed the previous run.")
            return 2

        if args.resume:
            ckpt = project_root / "models" / "weights" / args.name / "weights" / "last.pt"
            if not ckpt.is_file():
                print(f"ERROR: resume checkpoint not found: {ckpt}")
                return 1
            cmd = [str(python_exe), str(train_script), "--resume", str(ckpt)]
        else:
            cmd = [
                str(python_exe), str(train_script),
                "--epochs", str(args.epochs),
                "--batch", str(args.batch),
                "--name", args.name,
                "--model", args.model,
            ]

        print(f"Launching: {' '.join(cmd)}")
        print(f"Log: {log_file}")
        log_handle = open(log_file, "w")  # fresh log per launch
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        # Record trainer PID so future launches can detect it after we exit.
        # Must go through the already-open locked handle (write_text would collide).
        lock_fd.seek(0)
        lock_fd.truncate()
        lock_fd.write(str(proc.pid))
        lock_fd.flush()
        print(f"Training started (PID: {proc.pid}). Monitor: tail -f {log_file}")
        return 0
    finally:
        try:
            msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        lock_fd.close()


if __name__ == "__main__":
    raise SystemExit(main())
