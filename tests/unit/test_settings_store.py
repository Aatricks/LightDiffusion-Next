import importlib
import os

import pytest


def test_set_and_get_last_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("LD_SETTINGS_STORE_PATH", str(tmp_path / "settings_store.json"))
    # Import SettingsStore directly from file to avoid package-level side-effects
    import importlib.util
    spec = importlib.util.spec_from_file_location('settings_store_module', os.path.join(os.getcwd(), 'src', 'Core', 'SettingsStore.py'))
    SS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(SS)

    assert SS.get_last_seed() is None
    SS.set_last_seed(424242)
    assert SS.get_last_seed() == 424242


def test_append_and_get_history(tmp_path, monkeypatch):
    monkeypatch.setenv("LD_SETTINGS_STORE_PATH", str(tmp_path / "settings_store.json"))
    # Import module directly from file
    import importlib.util
    spec = importlib.util.spec_from_file_location('settings_store_module', os.path.join(os.getcwd(), 'src', 'Core', 'SettingsStore.py'))
    SS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(SS)

    assert SS.get_history() == []
    entry = SS.append_snapshot({"settings": {"prompt": "x", "width": 64}})
    hist = SS.get_history()
    assert isinstance(hist, list) and len(hist) == 1
    assert hist[0]["settings"]["prompt"] == "x"
    assert "id" in hist[0] and "ts" in hist[0]


def test_preferences_default_and_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("LD_SETTINGS_STORE_PATH", str(tmp_path / "settings_store.json"))
    import importlib.util
    spec = importlib.util.spec_from_file_location('settings_store_module', os.path.join(os.getcwd(), 'src', 'Core', 'SettingsStore.py'))
    SS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(SS)

    assert SS.get_preferences() == {"torch_compile": False, "vae_autotune": False}

    stored = SS.set_preferences({"torch_compile": True, "vae_autotune": True})
    assert stored == {"torch_compile": True, "vae_autotune": True}
    assert SS.get_preferences() == {"torch_compile": True, "vae_autotune": True}


def test_migrate_from_last_seed_txt(tmp_path, monkeypatch):
    store_path = tmp_path / "settings_store.json"
    monkeypatch.setenv("LD_SETTINGS_STORE_PATH", str(store_path))

    # Create legacy last_seed.txt in same directory
    basedir = store_path.parent
    basedir.mkdir(parents=True, exist_ok=True)
    legacy = basedir / "last_seed.txt"
    legacy.write_text("999999")

    # Import module directly from file
    import importlib.util
    spec = importlib.util.spec_from_file_location('settings_store_module', os.path.join(os.getcwd(), 'src', 'Core', 'SettingsStore.py'))
    SS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(SS)

    res = SS.migrate_from_last_seed_txt()
    assert res == 999999
    assert SS.get_last_seed() == 999999
    assert not legacy.exists()
