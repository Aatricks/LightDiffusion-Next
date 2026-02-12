"""LightDiffusion Settings persistence and history store.

This module provides a small JSON-backed settings store used for:
- persisting the last used seed (previously include/last_seed.txt)
- maintaining a short history of saved generation settings (for UI)

Design notes:
- Store file defaults to './include/settings_store.json'. Override via
  environment variable LD_SETTINGS_STORE_PATH for tests or custom locations.
- migrate_from_last_seed_txt() will import the legacy include/last_seed.txt
  if present and then remove the legacy file.
- Reads/writes are atomic (write to temp file then replace).

This is intentionally lightweight (no DB dependency) and robust to failure.
"""
from __future__ import annotations

import json
import os
import time
import tempfile
import uuid
from typing import Any, Dict, List, Optional


def _get_store_path() -> str:
    env = os.environ.get("LD_SETTINGS_STORE_PATH")
    if env:
        return env
    return os.path.join(os.getcwd(), "include", "settings_store.json")


def _read_store() -> Dict[str, Any]:
    path = _get_store_path()
    try:
        if not os.path.exists(path):
            return {"last_seed": None, "history": []}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Corrupt/invalid file -> return sane default (do not raise)
        return {"last_seed": None, "history": []}


def _write_store(data: Dict[str, Any]) -> None:
    path = _get_store_path()
    ddir = os.path.dirname(path)
    os.makedirs(ddir, exist_ok=True)
    # atomic write
    fd, tmp = tempfile.mkstemp(prefix="settings_store_", dir=ddir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


# Public API ---------------------------------------------------------------

def get_last_seed() -> Optional[int]:
    """Return the persisted last-seed or None if not set."""
    data = _read_store()
    seed = data.get("last_seed")
    return int(seed) if seed is not None else None


def set_last_seed(seed: int) -> None:
    """Persist the provided seed (int).

    Also ensures the store file exists and retains existing history.
    """
    try:
        d = _read_store()
        d["last_seed"] = int(seed)
        _write_store(d)
    except Exception:
        # Best-effort only; never raise for caller convenience
        pass


def get_history() -> List[Dict[str, Any]]:
    """Return the settings history (most-recent-first)."""
    data = _read_store()
    hist = data.get("history") or []
    # Ensure sensible return type
    return list(reversed(hist)) if hist else []


def append_snapshot(snapshot: Dict[str, Any], max_len: int = 64) -> Dict[str, Any]:
    """Append a snapshot to the history and return the stored entry.

    The incoming `snapshot` should be a dict (typically containing a
    `settings` key). We enrich it with `id` and `ts` fields.
    """
    try:
        d = _read_store()
        hist = d.get("history") or []
        entry = {
            "id": uuid.uuid4().hex[:12],
            "ts": int(time.time()),
            **snapshot,
        }
        # Store newest at the end for compact append logic
        hist.append(entry)
        # Trim oldest if exceeding max_len
        if len(hist) > max_len:
            hist = hist[-max_len:]
        d["history"] = hist
        _write_store(d)
        return entry
    except Exception:
        return {"id": uuid.uuid4().hex[:12], "ts": int(time.time()), **snapshot}


def migrate_from_last_seed_txt() -> Optional[int]:
    """Migrate legacy `include/last_seed.txt` into the JSON store.

    Migration target directory is derived from the store path so tests can
    override the store location with LD_SETTINGS_STORE_PATH and provide a
    corresponding legacy file next to it.
    """
    store_path = _get_store_path()
    basedir = os.path.dirname(store_path)
    legacy = os.path.join(basedir, "last_seed.txt")
    if not os.path.exists(legacy):
        return None
    try:
        with open(legacy, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            if not raw:
                return None
            seed = int(raw)
        set_last_seed(seed)
        try:
            os.remove(legacy)
        except Exception:
            pass
        return seed
    except Exception:
        return None
