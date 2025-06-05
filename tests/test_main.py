"""
Tests for the main module.
"""
import pytest
from pathlib import Path
from topic_sorting.main import load_data


def test_load_data(tmp_path):
    # Create a temporary JSON file
    test_data = {"test": "data"}
    test_file = tmp_path / "test.json"
    test_file.write_text('{"test": "data"}')
    
    # Test loading the data
    loaded_data = load_data(str(test_file))
    assert loaded_data == test_data 