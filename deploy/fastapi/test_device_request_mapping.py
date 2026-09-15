#!/usr/bin/env python3
"""How a request names its compute device, and how stage timings report it.

Two regressions from the `deepx` -> `device` migration are pinned here.

1. `deepx: false` stopped meaning "not the NPU". It was mapped to None
   (auto-select), and auto-select prefers the NPU, so the v5 online demo's
   CPU radio button ran on the NPU - and the response said so. The whole
   point of that demo is a CPU/NPU comparison.

2. `ocrStages` kept reading the deprecated flag while every neighbouring
   field had moved to the resolved device. A client that sends
   `device: "npu"` (and no `deepx`) got detection and recognition timings of
   0 while `perPage` - computed from the same totals a few lines below -
   reported the real numbers.
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ocr_service  # noqa: E402

resolve_request_device = ocr_service.resolve_request_device


# ------------------------------------------------------- device mapping ---

def test_explicit_device_wins_over_the_deprecated_flag():
    assert resolve_request_device("cpu", True) == "cpu"
    assert resolve_request_device("gpu", False) == "gpu"
    assert resolve_request_device("npu", False) == "npu"


def test_deepx_true_still_means_npu():
    assert resolve_request_device(None, True) == "npu"


def test_deepx_false_means_cpu_not_auto_select():
    """The regression: False must not fall through to auto-select.

    Auto-select walks npu > gpu > cpu, so returning None here hands the NPU
    to a caller that explicitly asked for anything but.
    """
    assert resolve_request_device(None, False) == "cpu"


def test_neither_field_auto_selects():
    assert resolve_request_device(None, None) is None


# -------------------------------------------------------- stage timings ---

TOTALS = {"doc_ori": 10.0, "doc_uv": 20.0, "det": 300.0, "cls": 40.0, "rec": 500.0}
ALL_ON = {"docOrientation": True, "docUnwarping": True, "textlineOrientation": True}
ALL_OFF = {"docOrientation": False, "docUnwarping": False, "textlineOrientation": False}


@pytest.fixture
def build():
    return ocr_service.build_stage_timings


def test_npu_reports_measured_stage_times(build):
    stages = build("npu", TOTALS, ALL_ON)
    assert stages["detectionMs"] == 300.0
    assert stages["recognitionMs"] == 500.0
    assert stages["docOrientationMs"] == 10.0
    assert stages["docUnwarpingMs"] == 20.0
    assert stages["textlineOrientationMs"] == 40.0


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_cpu_and_gpu_report_zero_because_they_are_not_instrumented(build, device):
    stages = build(device, TOTALS, ALL_ON)
    assert stages["detectionMs"] == 0
    assert stages["recognitionMs"] == 0


def test_a_stage_that_was_switched_off_reads_null_not_zero(build):
    """null = did not run, 0 = ran but was not measured. Different answers."""
    stages = build("npu", TOTALS, ALL_OFF)
    assert stages["docOrientationMs"] is None
    assert stages["docUnwarpingMs"] is None
    assert stages["textlineOrientationMs"] is None
    # Detection and recognition always run, so they are never null.
    assert stages["detectionMs"] == 300.0
    assert stages["recognitionMs"] == 500.0


def test_detection_and_recognition_do_not_depend_on_the_deprecated_flag(build):
    """The exact regression: device='npu' with no `deepx` field at all."""
    stages = build("npu", TOTALS, ALL_OFF)
    assert stages["detectionMs"] != 0
    assert stages["recognitionMs"] != 0


def test_pages_divides_the_totals_for_per_page_averages(build):
    stages = build("npu", TOTALS, ALL_ON, pages=4)
    assert stages["detectionMs"] == 75.0
    assert stages["recognitionMs"] == 125.0
    assert stages["docOrientationMs"] == 2.5


def test_zero_pages_does_not_divide_by_zero(build):
    stages = build("npu", TOTALS, ALL_ON, pages=0)
    assert stages["detectionMs"] == 300.0


def test_the_two_blocks_agree_on_which_stages_are_null(build):
    """ocrStages and perPage drifting apart is what produced the bug."""
    totals_view = build("npu", TOTALS, ALL_OFF)
    per_page_view = build("npu", TOTALS, ALL_OFF, pages=3)
    nulls = lambda d: {k for k, v in d.items() if v is None}
    assert nulls(totals_view) == nulls(per_page_view)
