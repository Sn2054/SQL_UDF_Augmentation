#!/usr/bin/env python3
"""Cross-process, cross-host advisory lock for the shared results/*.xlsx files.

results/baseline_cost_estimation.xlsx and results/augmented_cost_estimation.xlsx
are read-modify-written (load workbook, add/replace one row, save the whole
file) by update_baseline_cost_estimation.py and update_augmented_cost_estimation.py.
Two run_code.sh jobs finishing at the same moment -- including from different
hosts, since results/ lives on a shared /mnt/shared mount -- can otherwise
race on that save and silently clobber each other's row (or, since
openpyxl.save() is not atomic, corrupt the workbook if both writes land at
once).

This uses a lock *directory* rather than flock()/fcntl byte-range locks:
os.mkdir() is a single atomic RPC on NFS, whereas flock-style locking needs a
working NFS lock manager (rpc.statd/rpc.lockd) that shared mounts don't
always provide reliably across hosts.
"""

from __future__ import annotations

import contextlib
import os
import socket
import time

STALE_LOCK_SECONDS = 600  # a lock older than this is assumed to be from a dead/killed process


@contextlib.contextmanager
def locked(path: str, timeout: float = 120.0, poll_interval: float = 0.2):
    """Hold an exclusive lock associated with ``path`` for the duration of the block."""
    lock_dir = f"{path}.lock"
    owner_file = os.path.join(lock_dir, "owner")
    deadline = time.monotonic() + timeout

    while True:
        try:
            os.mkdir(lock_dir)
        except FileExistsError:
            _clear_if_stale(lock_dir, owner_file)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for lock on {path} ({lock_dir})")
            time.sleep(poll_interval)
            continue
        else:
            break

    try:
        with open(owner_file, "w") as fh:
            fh.write(f"{socket.gethostname()} pid={os.getpid()} time={time.time()}\n")
        yield
    finally:
        try:
            os.remove(owner_file)
        except OSError:
            pass
        try:
            os.rmdir(lock_dir)
        except OSError:
            pass


def _clear_if_stale(lock_dir: str, owner_file: str) -> None:
    try:
        age = time.time() - os.path.getmtime(owner_file)
    except OSError:
        return
    if age <= STALE_LOCK_SECONDS:
        return
    try:
        os.remove(owner_file)
        os.rmdir(lock_dir)
    except OSError:
        pass
