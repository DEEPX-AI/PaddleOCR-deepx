#!/usr/bin/env python3
"""Per-request compute device selection: cpu | gpu | npu.

Selection is per request. The previous scheme could not express "this server
uses the NPU, but run this request on the GPU": USE_GPU was a process-wide
environment variable while the NPU was picked per request via a `deepx` flag,
so the two axes never met.

Omit the device and the best available one is used, in the order npu > gpu >
cpu. Name a device explicitly and it is either used or the request fails --
quietly downgrading npu to cpu would hide a broken deployment behind results
that merely look slow.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Auto-selection priority, best first.
DEVICES = ("npu", "gpu", "cpu")

HERE = Path(__file__).resolve().parent
PADDLEX_MODELS = Path.home() / ".paddlex" / "official_models"


class DeviceError(Exception):
    """Unusable or unknown device, with a message meant for the caller."""


# --------------------------------------------------------------- probing ---

def _probe_gpu():
    """GPU availability, separating a CPU-only build from absent hardware.

    These need different fixes and look identical from paddle alone: a machine
    with a real GPU and a CPU-only wheel reports exactly what a machine with no
    GPU reports. Saying "no GPU" would send the user hunting for hardware they
    already have, so nvidia-smi is consulted for the hardware side.
    """
    has_hw = False
    hw_desc = ""
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10)
            names = [n.strip() for n in out.stdout.splitlines() if n.strip()]
            has_hw, hw_desc = bool(names), ", ".join(names)
        except Exception:
            pass

    try:
        import paddle
    except Exception as exc:
        return {"available": False,
                "reason": f"paddle is not importable ({type(exc).__name__})",
                "hint": "pip install -r requirements.txt"}

    if not paddle.is_compiled_with_cuda():
        if has_hw:
            return {"available": False,
                    "reason": f"GPU present ({hw_desc}) but paddlepaddle is a CPU-only build",
                    "hint": "./local_setup.sh --gpu   (installs paddlepaddle-gpu)"}
        return {"available": False,
                "reason": "no NVIDIA GPU detected and paddlepaddle is a CPU-only build",
                "hint": "install an NVIDIA GPU + driver, then ./local_setup.sh --gpu"}

    try:
        count = paddle.device.cuda.device_count()
    except Exception as exc:
        return {"available": False,
                "reason": f"CUDA build present but device query failed ({type(exc).__name__})",
                "hint": "check the NVIDIA driver (nvidia-smi)"}

    if count < 1:
        return {"available": False,
                "reason": "paddlepaddle has CUDA support but no CUDA device is visible",
                "hint": "check the NVIDIA driver / CUDA_VISIBLE_DEVICES"}
    return {"available": True, "detail": f"{count} CUDA device(s)"
            + (f" - {hw_desc}" if hw_desc else "")}


def _probe_npu(model_dir=None):
    """NPU availability: the dx_engine runtime AND compiled .dxnn models."""
    try:
        import dx_engine  # noqa: F401
    except Exception:
        return {"available": False,
                "reason": "dx_engine runtime is not installed",
                "hint": "./local_deepx_setup.sh --dx_rt /path/to/dx_rt"}

    base = Path(model_dir) if model_dir else HERE / "deepx" / "engine" / "model_files"
    if not base.exists():
        return {"available": False,
                "reason": f"no NPU model directory at {base}",
                "hint": "cd deepx && ./setup.sh"}

    present = [d.name for d in base.iterdir()
               if d.is_dir() and any(d.glob("*.dxnn"))]
    if not present:
        return {"available": False,
                "reason": f"no .dxnn models under {base}",
                "hint": "cd deepx && ./setup.sh"}
    return {"available": True, "detail": "model sets: " + ", ".join(sorted(present))}


def probe():
    """Availability of every device, with a reason and fix when unavailable."""
    return {
        "cpu": {"available": True, "detail": "always available"},
        "gpu": _probe_gpu(),
        "npu": _probe_npu(),
    }


def available_devices():
    return {name: info["available"] for name, info in probe().items()}


# -------------------------------------------------------------- resolving ---

def resolve_device(requested, caps=None):
    """Pick the device for one request.

    requested=None -> best available in DEVICES order.
    requested=name -> that device, or DeviceError if it is unavailable.
    """
    if caps is None:
        caps = available_devices()

    if requested is not None:
        name = str(requested).strip().lower()
        if name not in DEVICES:
            raise DeviceError(
                f"Unknown device {requested!r}. Valid devices: "
                + " | ".join(DEVICES) + " (or omit to auto-select)")
        if not caps.get(name):
            usable = [d for d in DEVICES if caps.get(d)]
            raise DeviceError(
                f"Device {name!r} is not available on this deployment. "
                f"Available: {', '.join(usable) if usable else 'none'}. "
                f"Run ./run.sh --sanity-check to see why.")
        return name

    for name in DEVICES:
        if caps.get(name):
            return name
    raise DeviceError("No compute device is available, not even cpu.")


# ---------------------------------------------------------------- report ---

def _version(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


def report():
    """Human-readable environment check for `run.sh --sanity-check`."""
    info = probe()
    out = ["", "PaddleOCR-deepx environment check", ""]

    out.append(" Installation")
    for mod, label in (("paddleocr", "paddleocr"), ("paddle", "paddlepaddle"),
                       ("dx_engine", "dx_engine")):
        ver = _version(mod)
        if ver is None:
            out.append(f"   {label:<18} -         not installed")
        elif mod == "paddle":
            try:
                import paddle
                build = "CUDA build" if paddle.is_compiled_with_cuda() else "CPU-only build"
            except Exception:
                build = ""
            out.append(f"   {label:<18} {ver:<9} OK  ({build})")
        else:
            out.append(f"   {label:<18} {ver:<9} OK")

    out += ["", " Devices"]
    for name in DEVICES:
        d = info[name]
        if d["available"]:
            out.append(f"   [ok] {name:<4} {d.get('detail', '')}")
        else:
            out.append(f"   [--] {name:<4} {d.get('reason', '')}")
            if d.get("hint"):
                out.append(f"            -> {d['hint']}")

    out += ["", " Models"]
    out.append("   CPU/GPU (paddleocr official models)")
    if PADDLEX_MODELS.exists():
        cached = sorted(p.name for p in PADDLEX_MODELS.iterdir() if p.is_dir())
        ocr = [c for c in cached if c.startswith("PP-OCR")]
        out.append(f"     cached: {', '.join(ocr) if ocr else '(none yet)'}")
    else:
        out.append("     cache empty - models download on first use")

    out.append("   NPU (.dxnn)")
    base = HERE / "deepx" / "engine" / "model_files"
    if base.exists():
        for d in sorted(base.iterdir()):
            if d.is_dir():
                n = len(list(d.glob("*.dxnn")))
                out.append(f"     {d.name:<16} {n} file(s)" if n else f"     {d.name:<16} empty")
    else:
        out.append("     not downloaded -> cd deepx && ./setup.sh")

    usable = [d for d in DEVICES if info[d]["available"]]
    out += ["", f" Verdict: usable devices - {', '.join(usable)}", ""]
    return "\n".join(out)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Device availability check")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    print(report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
