#!/usr/bin/env python3
"""Per-request device selection: cpu | gpu | npu.

Selection is per request, not per server. That is the point of the rewrite:
USE_GPU was a process-wide environment variable while the NPU was chosen per
request, so "this server uses the NPU, but run this one on the GPU" could not
be expressed at all.
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from device_selection import (  # noqa: E402
    DEVICES, DeviceError, available_devices, probe, report, resolve_device,
)

ALL = {"cpu": True, "gpu": True, "npu": True}


def test_priority_order_is_npu_gpu_cpu():
    assert DEVICES == ("npu", "gpu", "cpu")


@pytest.mark.parametrize("want", ["cpu", "gpu", "npu"])
def test_explicit_available_device_is_honoured(want):
    assert resolve_device(want, ALL) == want


@pytest.mark.parametrize("caps,expected", [
    ({"cpu": True, "gpu": True, "npu": True}, "npu"),
    ({"cpu": True, "gpu": True, "npu": False}, "gpu"),
    ({"cpu": True, "gpu": False, "npu": False}, "cpu"),
    ({"cpu": True, "gpu": False, "npu": True}, "npu"),
])
def test_auto_selection_walks_the_priority_order(caps, expected):
    assert resolve_device(None, caps) == expected


def test_explicit_unavailable_device_raises_instead_of_falling_back():
    # Silently downgrading npu -> cpu would hide a misconfigured deployment.
    with pytest.raises(DeviceError) as e:
        resolve_device("npu", {"cpu": True, "gpu": False, "npu": False})
    assert "npu" in str(e.value).lower()


def test_error_names_the_devices_that_are_available():
    with pytest.raises(DeviceError) as e:
        resolve_device("gpu", {"cpu": True, "gpu": False, "npu": True})
    msg = str(e.value)
    assert "cpu" in msg and "npu" in msg


@pytest.mark.parametrize("bad", ["tpu", "GPU0", "", "npu0"])
def test_unknown_device_names_are_rejected(bad):
    with pytest.raises(DeviceError):
        resolve_device(bad, ALL)


def test_device_names_are_case_insensitive():
    assert resolve_device("NPU", ALL) == "npu"
    assert resolve_device(" gpu ", ALL) == "gpu"


def test_cpu_is_always_available():
    assert available_devices()["cpu"] is True


def test_probe_explains_why_a_device_is_unavailable():
    info = probe()
    for name in DEVICES:
        assert name in info
        assert "available" in info[name]
        if not info[name]["available"]:
            # An operator has to be able to act on this.
            assert info[name]["reason"], f"{name} unavailable with no reason"


def test_gpu_distinguishes_cpu_only_build_from_absent_hardware():
    # This machine has an RTX 5060 Ti but a CPU-only paddle wheel. Saying
    # "no GPU" would send the user hunting for hardware they already have.
    info = probe()["gpu"]
    if not info["available"]:
        assert info["reason"], "gpu unavailable with no reason"
        assert info.get("hint"), "gpu unavailable with no fix hint"


def test_report_is_printable_text():
    text = report()
    assert isinstance(text, str) and text.strip()
    for name in DEVICES:
        assert name in text
