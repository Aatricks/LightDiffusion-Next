"""
Unit tests for the HistoryManager module.

Tests cover:
- Seed sanitization edge cases
- History entry normalization
- JSON serialization/deserialization
- Deduplication logic
- Backup rotation
- Search/filter functionality
"""

import os
import json
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock, mock_open
from dataclasses import asdict

import sys
# Use direct import to avoid loading heavy src __init__.py dependencies
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "HistoryManager",
    os.path.join(os.path.dirname(__file__), "../../src/FileManaging/HistoryManager.py")
)
HistoryManagerModule = importlib.util.module_from_spec(spec)
sys.modules["HistoryManager"] = HistoryManagerModule
spec.loader.exec_module(HistoryManagerModule)

sanitize_seed_for_display = HistoryManagerModule.sanitize_seed_for_display
HistoryEntry = HistoryManagerModule.HistoryEntry
HistoryManager = HistoryManagerModule.HistoryManager
_parse_float_safe = HistoryManagerModule._parse_float_safe
_parse_int_safe = HistoryManagerModule._parse_int_safe


class TestSanitizeSeedForDisplay(unittest.TestCase):
    """Test cases for seed sanitization function."""
    
    def test_none_input(self):
        """None should return None."""
        self.assertIsNone(sanitize_seed_for_display(None))
    
    def test_integer_input(self):
        """Integers should be converted to strings."""
        self.assertEqual(sanitize_seed_for_display(12345), "12345")
        self.assertEqual(sanitize_seed_for_display(0), "0")
        self.assertEqual(sanitize_seed_for_display(-123), "-123")
    
    def test_float_input(self):
        """Floats should be converted to int strings."""
        self.assertEqual(sanitize_seed_for_display(12345.0), "12345")
        self.assertEqual(sanitize_seed_for_display(12345.7), "12345")
    
    def test_simple_string(self):
        """Simple numeric strings should be preserved."""
        self.assertEqual(sanitize_seed_for_display("12345"), "12345")
        self.assertEqual(sanitize_seed_for_display("  12345  "), "12345")
    
    def test_tensor_dump_extraction(self):
        """Tensor dumps should extract numeric content."""
        result = sanitize_seed_for_display("tensor(123456789)")
        self.assertEqual(result, "123456789")
    
    def test_array_dump_extraction(self):
        """Array-like content should extract numeric content."""
        result = sanitize_seed_for_display("[1, 2, 3, seed=987654321]")
        self.assertEqual(result, "987654321")
    
    def test_very_long_string_extraction(self):
        """Very long strings should try to extract a seed."""
        long_string = "x" * 300 + "12345678" + "y" * 100
        result = sanitize_seed_for_display(long_string)
        self.assertEqual(result, "12345678")
    
    def test_multiline_string_extraction(self):
        """Multiline strings should try to extract a seed."""
        multiline = "line1\nline2\nseed=98765432\nline3"
        result = sanitize_seed_for_display(multiline)
        self.assertEqual(result, "98765432")
    
    def test_no_numeric_in_garbage(self):
        """Garbage without numeric content should return None."""
        self.assertIsNone(sanitize_seed_for_display("tensor(abc, def)"))
    
    def test_short_numeric_not_extracted(self):
        """Numeric tokens shorter than 4 digits should not be extracted from dumps."""
        result = sanitize_seed_for_display("tensor(123)")
        self.assertIsNone(result)
    
    def test_empty_string(self):
        """Empty string should return None."""
        self.assertIsNone(sanitize_seed_for_display(""))
        self.assertIsNone(sanitize_seed_for_display("   "))


class TestParseFunctions(unittest.TestCase):
    """Test helper parsing functions."""
    
    def test_parse_float_safe_valid(self):
        """Valid float inputs."""
        self.assertEqual(_parse_float_safe(3.14), 3.14)
        self.assertEqual(_parse_float_safe("3.14"), 3.14)
        self.assertEqual(_parse_float_safe(42), 42.0)
    
    def test_parse_float_safe_with_suffix(self):
        """Float strings with 's' suffix (like '10.5s')."""
        self.assertEqual(_parse_float_safe("10.5s"), 10.5)
        self.assertEqual(_parse_float_safe("3.2s"), 3.2)
    
    def test_parse_float_safe_invalid(self):
        """Invalid float inputs should return None."""
        self.assertIsNone(_parse_float_safe(None))
        self.assertIsNone(_parse_float_safe("abc"))
        self.assertIsNone(_parse_float_safe([1, 2, 3]))
    
    def test_parse_int_safe_valid(self):
        """Valid int inputs."""
        self.assertEqual(_parse_int_safe(42), 42)
        self.assertEqual(_parse_int_safe("42"), 42)
        self.assertEqual(_parse_int_safe(3.7), 3)
    
    def test_parse_int_safe_invalid(self):
        """Invalid int inputs should return None."""
        self.assertIsNone(_parse_int_safe(None))
        self.assertIsNone(_parse_int_safe("abc"))


class TestHistoryEntry(unittest.TestCase):
    """Test HistoryEntry dataclass."""
    
    def test_to_dict(self):
        """Entry should serialize to dictionary."""
        entry = HistoryEntry(
            timestamp="2024-01-01 12:00:00",
            image_path="/path/to/image.png",
            prompt="test prompt",
            seed="12345"
        )
        d = entry.to_dict()
        self.assertEqual(d["timestamp"], "2024-01-01 12:00:00")
        self.assertEqual(d["prompt"], "test prompt")
        self.assertEqual(d["seed"], "12345")
    
    def test_from_dict(self):
        """Entry should deserialize from dictionary."""
        data = {
            "timestamp": "2024-01-01 12:00:00",
            "image_path": "/path/to/image.png",
            "prompt": "test prompt",
            "extra_field": "ignored"
        }
        entry = HistoryEntry.from_dict(data)
        self.assertEqual(entry.timestamp, "2024-01-01 12:00:00")
        self.assertEqual(entry.prompt, "test prompt")
    
    def test_default_values(self):
        """Entry should have sensible defaults."""
        entry = HistoryEntry(timestamp="", image_path="")
        self.assertEqual(entry.prompt, "")
        self.assertIsNone(entry.width)
        self.assertEqual(entry.png_metadata, {})


class TestHistoryManager(unittest.TestCase):
    """Test HistoryManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.history_file = os.path.join(self.temp_dir, "test_history.json")
        self.manager = HistoryManager(history_file=self.history_file)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_empty_history(self):
        """Loading non-existent file should return empty list."""
        entries = self.manager.load()
        self.assertEqual(entries, [])
    
    def test_save_and_load(self):
        """Saved entries should be loadable."""
        entry = HistoryEntry(
            timestamp="2024-01-01 12:00:00",
            image_path="/path/to/image.png",
            prompt="test"
        )
        self.manager.save([entry])
        
        # Force cache invalidation
        loaded = self.manager.load(use_cache=False)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].prompt, "test")
    
    def test_deduplication_by_path(self):
        """Duplicate image paths should be deduplicated."""
        data = [
            {"timestamp": "2024-01-01", "image_path": "/path/a.png", "prompt": "first"},
            {"timestamp": "2024-01-02", "image_path": "/path/a.png", "prompt": "duplicate"},
            {"timestamp": "2024-01-03", "image_path": "/path/b.png", "prompt": "second"},
        ]
        with open(self.history_file, "w") as f:
            json.dump(data, f)
        
        entries = self.manager.load(use_cache=False)
        self.assertEqual(len(entries), 2)
        paths = [e.image_path for e in entries]
        self.assertIn("/path/a.png", paths)
        self.assertIn("/path/b.png", paths)
    
    def test_max_entries_limit(self):
        """History should be limited to MAX_HISTORY_ENTRIES."""
        data = [
            {"timestamp": f"2024-01-{i:02d}", "image_path": f"/path/{i}.png"}
            for i in range(150)
        ]
        with open(self.history_file, "w") as f:
            json.dump(data, f)
        
        entries = self.manager.load(use_cache=False)
        self.assertEqual(len(entries), 100)
    
    def test_cache_behavior(self):
        """Cache should be used when valid."""
        entry = HistoryEntry(timestamp="2024-01-01", image_path="/path/a.png")
        self.manager.save([entry])
        
        # First load populates cache
        self.manager.load()
        
        # Modify file time to simulate no change
        original_mtime = self.manager._cache_mtime
        
        # Second load should use cache
        entries = self.manager.load(use_cache=True)
        self.assertEqual(len(entries), 1)
    
    def test_add_entry(self):
        """Adding an entry should prepend to history."""
        entry1 = HistoryEntry(timestamp="2024-01-01", image_path="/path/a.png")
        entry2 = HistoryEntry(timestamp="2024-01-02", image_path="/path/b.png")
        
        self.manager.save([entry1])
        self.manager.add_entry(entry2)
        
        entries = self.manager.load(use_cache=False)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].image_path, "/path/b.png")
    
    def test_delete_entry(self):
        """Deleting an entry should remove it from history."""
        entry1 = HistoryEntry(timestamp="2024-01-01", image_path="/path/a.png")
        entry2 = HistoryEntry(timestamp="2024-01-02", image_path="/path/b.png")
        
        self.manager.save([entry1, entry2])
        self.manager.delete_entry(0)
        
        entries = self.manager.load(use_cache=False)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].image_path, "/path/b.png")
    
    def test_clear_history(self):
        """Clearing should remove all entries."""
        entries = [
            HistoryEntry(timestamp=f"2024-01-{i:02d}", image_path=f"/path/{i}.png")
            for i in range(5)
        ]
        self.manager.save(entries)
        self.manager.clear(delete_files=False)
        
        loaded = self.manager.load(use_cache=False)
        self.assertEqual(len(loaded), 0)
    
    def test_search_by_keyword(self):
        """Search should filter by keyword in prompt."""
        entries = [
            HistoryEntry(timestamp="2024-01-01", image_path="/a.png", prompt="cat photo"),
            HistoryEntry(timestamp="2024-01-02", image_path="/b.png", prompt="dog photo"),
            HistoryEntry(timestamp="2024-01-03", image_path="/c.png", prompt="cat and dog"),
        ]
        self.manager.save(entries)
        
        results = self.manager.search(keyword="cat")
        self.assertEqual(len(results), 2)
        
        results = self.manager.search(keyword="dog")
        self.assertEqual(len(results), 2)
        
        results = self.manager.search(keyword="bird")
        self.assertEqual(len(results), 0)
    
    def test_search_by_model_type(self):
        """Search should filter by model type."""
        entries = [
            HistoryEntry(timestamp="2024-01-01", image_path="/a.png", model_type="SD15"),
            HistoryEntry(timestamp="2024-01-02", image_path="/b.png", model_type="SDXL"),
            HistoryEntry(timestamp="2024-01-03", image_path="/c.png", model_type="SD15"),
        ]
        self.manager.save(entries)
        
        results = self.manager.search(model_type="SD15")
        self.assertEqual(len(results), 2)
        
        results = self.manager.search(model_type="SDXL")
        self.assertEqual(len(results), 1)
    
    def test_search_by_date_range(self):
        """Search should filter by date range."""
        entries = [
            HistoryEntry(timestamp="2024-01-15 10:00:00", image_path="/a.png"),
            HistoryEntry(timestamp="2024-02-15 10:00:00", image_path="/b.png"),
            HistoryEntry(timestamp="2024-03-15 10:00:00", image_path="/c.png"),
        ]
        self.manager.save(entries)
        
        results = self.manager.search(date_from="2024-02-01")
        self.assertEqual(len(results), 2)
        
        results = self.manager.search(date_to="2024-02-01")
        self.assertEqual(len(results), 1)
        
        results = self.manager.search(date_from="2024-02-01", date_to="2024-02-28")
        self.assertEqual(len(results), 1)
    
    def test_get_model_types(self):
        """Should return unique model types."""
        entries = [
            HistoryEntry(timestamp="2024-01-01", image_path="/a.png", model_type="SD15"),
            HistoryEntry(timestamp="2024-01-02", image_path="/b.png", model_type="SDXL"),
            HistoryEntry(timestamp="2024-01-03", image_path="/c.png", model_type="SD15"),
            HistoryEntry(timestamp="2024-01-04", image_path="/d.png"),
        ]
        self.manager.save(entries)
        
        types = self.manager.get_model_types()
        self.assertEqual(sorted(types), ["SD15", "SDXL"])


class TestBackupRotation(unittest.TestCase):
    """Test backup rotation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.history_file = os.path.join(self.temp_dir, "test_history.json")
        self.backup_dir = os.path.join(self.temp_dir, ".history_backups")
        
        # Patch BACKUP_DIR using the imported module reference
        self.backup_dir_patcher = patch.object(
            HistoryManagerModule, 'BACKUP_DIR',
            self.backup_dir
        )
        self.backup_dir_patcher.start()

        
        self.manager = HistoryManager(history_file=self.history_file)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.backup_dir_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_backup_created_on_save(self):
        """Backup should be created when saving."""
        # First save
        entry = HistoryEntry(timestamp="2024-01-01", image_path="/a.png")
        self.manager.save([entry])
        
        # Second save should create backup
        entry2 = HistoryEntry(timestamp="2024-01-02", image_path="/b.png")
        self.manager.save([entry, entry2])
        
        backup_file = os.path.join(self.backup_dir, "history_backup_1.json")
        self.assertTrue(os.path.exists(backup_file))
    
    def test_restore_from_backup(self):
        """Should restore from backup when main file is corrupt."""
        # Create valid backup
        os.makedirs(self.backup_dir, exist_ok=True)
        backup_data = [{"timestamp": "2024-01-01", "image_path": "/backup.png"}]
        backup_file = os.path.join(self.backup_dir, "history_backup_1.json")
        with open(backup_file, "w") as f:
            json.dump(backup_data, f)
        
        # Create corrupt main file
        with open(self.history_file, "w") as f:
            f.write("not valid json {{{")
        
        entries = self.manager.load(use_cache=False)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].image_path, "/backup.png")


if __name__ == "__main__":
    unittest.main()
