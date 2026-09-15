#!/usr/bin/env python3
"""CPU/GPU path model naming per OCR version.

The NPU path loads .dxnn files by hand, but the CPU/GPU path asks paddleocr for
official model *names*. Those names carry the version, so OCR_VERSION has to be
read here too - otherwise selecting v6 silently serves PP-OCRv5 weights.
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
    for var in ("OCR_VERSION", "V6_MODEL_SIZE", "USE_MOBILE", "MODEL_SIZE"):
        monkeypatch.delenv(var, raising=False)
    import ocr_service
    return importlib.reload(ocr_service)


# v5 is no longer the default - it has to be asked for. See test_v6_default.py
# for what an unset OCR_VERSION now resolves to.
def test_v5_server(monkeypatch, svc):
    monkeypatch.setenv("OCR_VERSION", "v5")
    paths = svc.get_model_paths()
    assert paths["det_model_name"] == "PP-OCRv5_server_det"
    assert paths["rec_model_name"] == "PP-OCRv5_server_rec"
    assert paths["model_type"] == "server"


def test_v5_mobile(monkeypatch, svc):
    monkeypatch.setenv("OCR_VERSION", "v5")
    monkeypatch.setenv("USE_MOBILE", "true")
    paths = svc.get_model_paths()
    assert paths["det_model_name"] == "PP-OCRv5_mobile_det"
    assert paths["model_type"] == "mobile"


@pytest.mark.parametrize("size,expected", [
    ("m", "medium"), ("s", "small"), ("t", "tiny"),
])
def test_v6_uses_ppocrv6_names(monkeypatch, svc, size, expected):
    monkeypatch.setenv("OCR_VERSION", "v6")
    monkeypatch.setenv("V6_MODEL_SIZE", size)
    paths = svc.get_model_paths()
    assert paths["det_model_name"] == f"PP-OCRv6_{expected}_det"
    assert paths["rec_model_name"] == f"PP-OCRv6_{expected}_rec"
    assert paths["model_type"] == expected


def test_v6_defaults_to_medium(monkeypatch, svc):
    monkeypatch.setenv("OCR_VERSION", "v6")
    assert svc.get_model_paths()["det_model_name"] == "PP-OCRv6_medium_det"


def test_names_are_returned_even_when_not_downloaded_yet(monkeypatch, svc, tmp_path):
    # paddleocr downloads official models on demand. Returning None for a model
    # that simply is not cached yet makes it fall back to its own default, which
    # is how selecting v6 ended up serving v5.
    monkeypatch.setenv("OCR_VERSION", "v6")
    monkeypatch.setattr(svc.Path, "home", lambda: tmp_path)
    paths = svc.get_model_paths()
    assert paths["det_model_name"] == "PP-OCRv6_medium_det"
    assert paths["det_model_dir"] is None      # not cached -> let paddleocr fetch it
