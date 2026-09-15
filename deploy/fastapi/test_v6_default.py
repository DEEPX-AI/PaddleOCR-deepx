#!/usr/bin/env python3
"""PP-OCRv6 is the default version.

model_selection already defaults to v6, but ocr_service kept its own 'v5'
fallback. Anything that bypasses model_selection - `uvicorn ocr_service:app`,
importing the module, a request arriving before OCR_VERSION is set - silently
got v5 while run.sh gave v6.
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


def test_default_version_is_v6(svc):
    assert svc.get_ocr_version() == "v6"


def test_default_matches_model_selection(svc):
    from model_selection import DEFAULT_VERSION
    assert svc.get_ocr_version() == DEFAULT_VERSION


def test_v5_is_still_selectable(monkeypatch, svc):
    monkeypatch.setenv("OCR_VERSION", "v5")
    assert svc.get_ocr_version() == "v5"


@pytest.mark.parametrize("bad", ["v7", "", "six", "V6X"])
def test_unrecognised_values_fall_back_to_the_default(monkeypatch, svc, bad):
    monkeypatch.setenv("OCR_VERSION", bad)
    assert svc.get_ocr_version() == "v6"


def test_cpu_path_defaults_to_v6_models(svc):
    assert svc.get_model_paths()["det_model_name"] == "PP-OCRv6_medium_det"


def test_missing_v6_models_point_at_the_downloader(svc, tmp_path, monkeypatch):
    # v5's message names local_deepx_setup.sh, but the actual downloader for
    # the .dxnn files is deepx/setup.sh. Selecting v6 on a host that has not
    # fetched them must say how to get them.
    msg = svc.v6_models_missing_message(tmp_path / "v6")
    assert "setup.sh" in msg
    assert "OCR_VERSION=v5" in msg
