#!/usr/bin/env python3
"""PP-OCRv6 size selection and per-size dictionary resolution.

The 2026-09-11 rebuild ships one dictionary per size (rec_v6_{t,s,m}_dict.txt)
because tiny uses a 6906-class head while small/medium use 18710. Earlier
packages shipped a single ppocrv6_dict.txt, which must keep working.
"""
import importlib
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


@pytest.fixture
def svc(monkeypatch):
    monkeypatch.delenv("V6_MODEL_SIZE", raising=False)
    import ocr_service
    return importlib.reload(ocr_service)


@pytest.mark.parametrize("value,expected", [
    (None, "m"), ("m", "m"), ("s", "s"), ("t", "t"),
    ("T", "t"), ("tiny", "m"), ("", "m"),
])
def test_size_selection(monkeypatch, svc, value, expected):
    if value is None:
        monkeypatch.delenv("V6_MODEL_SIZE", raising=False)
    else:
        monkeypatch.setenv("V6_MODEL_SIZE", value)
    assert svc.get_v6_model_size() == expected


def test_per_size_dict_preferred(tmp_path, svc):
    (tmp_path / "rec_v6_t_dict.txt").write_text("a\n")
    (tmp_path / "ppocrv6_dict.txt").write_text("b\n")
    assert svc.resolve_v6_dict(tmp_path, "t").name == "rec_v6_t_dict.txt"


def test_falls_back_to_shared_dict(tmp_path, svc):
    (tmp_path / "ppocrv6_dict.txt").write_text("b\n")
    assert svc.resolve_v6_dict(tmp_path, "m").name == "ppocrv6_dict.txt"


def test_missing_dict_returns_shared_path_for_error_reporting(tmp_path, svc):
    # Nothing on disk: the shared name is returned so the caller raises a
    # single clear "dictionary not found" error instead of a confusing one.
    assert svc.resolve_v6_dict(tmp_path, "s").name == "ppocrv6_dict.txt"
