#!/usr/bin/env python3
"""PP-OCRv5 vs PP-OCRv6 end-to-end accuracy / FPS benchmark on the DEEPX DX-M1 NPU.

Ground truth comes from deepx/images/labels.json ({filename: [{text, bbox}, ...]}).
Accuracy is 1 - normalized edit distance between the whitespace-stripped
concatenation of the recognized lines and of the ground-truth lines, which is the
same end-to-end measure used in CSP-1527.

Each configuration runs in its OWN process (--single) because every OCR version
loads its own set of InferenceEngine instances onto the NPU; loading v5 (11
models) and both v6 sizes into one process risks exhausting NPU memory.

usage:
  python benchmark_npu.py                      # v5 + v6/s + v6/m, one process each
  python benchmark_npu.py --single v5          # one configuration in this process
  python benchmark_npu.py --single v6 --size m
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

IMAGES_DIR = HERE / "deepx" / "images"
LABELS_JSON = IMAGES_DIR / "labels.json"
ENV_DEEPX = HERE / ".env.deepx"


def load_dxrt_env():
    """Apply the DX-RT runtime settings from .env.deepx.

    Without DXRT_TASK_MAX_LOAD the v5 server set (2 det + 6 rec + 3 aux models)
    exhausts NPU memory at load time:
      [DXRT][Error] Failed to register memory cache for task 12
    """
    if not ENV_DEEPX.exists():
        return
    for line in ENV_DEEPX.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


# ---------------------------------------------------------------- accuracy ---

def normalize(text):
    """Strip whitespace so line-break differences do not count as errors."""
    return "".join(ch for ch in text if not ch.isspace())


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def accuracy(pred, gt):
    p, g = normalize(pred), normalize(gt)
    if not g:
        return 1.0 if not p else 0.0
    return max(0.0, 1.0 - edit_distance(p, g) / len(g))


def load_labels():
    raw = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    labels = {}
    for fname, lines in raw.items():
        if isinstance(lines, list):
            labels[fname] = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in lines
            )
        else:
            labels[fname] = str(lines)
    return labels


# --------------------------------------------------------------- one config ---

def run_single(version, size, model_dir=None):
    """Benchmark one (version, size) configuration in THIS process."""
    import cv2

    load_dxrt_env()
    os.environ["OCR_VERSION"] = version
    os.environ["V6_MODEL_SIZE"] = size
    os.environ["SETUP_NPU"] = "true"
    if model_dir:
        os.environ["V6_MODEL_DIR"] = model_dir
    else:
        os.environ.pop("V6_MODEL_DIR", None)

    import ocr_service

    ocr_service._npu_models = None
    ocr_service._npu_ocr_sync_instance = None
    ocr_service._npu_ocr_async_instance = None

    ocr = ocr_service.get_npu_ocr_instance(sync=True, use_textline_orientation=True)

    labels = load_labels()
    images = sorted(p for p in IMAGES_DIR.glob("*.png"))

    accs, times, line_counts = [], [], []
    per_image = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        t0 = time.time()
        result = ocr(img)
        elapsed = time.time() - t0

        rec_results = result[2] if isinstance(result, tuple) and len(result) >= 3 else []
        text = "".join(r.get("text", "") for r in rec_results)
        acc = accuracy(text, labels.get(img_path.name, ""))

        times.append(elapsed)
        accs.append(acc)
        line_counts.append(len(rec_results))
        per_image.append({"image": img_path.name, "accuracy": round(100 * acc, 2),
                          "lines": len(rec_results), "ms": round(1000 * elapsed, 1)})

    total_time = sum(times) or 1e-9
    return {
        "version": version,
        "size": size if version == "v6" else "-",
        "model_dir": model_dir or "(default v6)",
        "images": len(times),
        "accuracy_pct": round(100 * sum(accs) / len(accs), 2) if accs else 0.0,
        "fps": round(len(times) / total_time, 2),
        "ms_per_image": round(1000 * total_time / len(times), 1) if times else 0.0,
        "avg_lines": round(sum(line_counts) / len(line_counts), 1) if line_counts else 0.0,
        "per_image": per_image,
    }


# ------------------------------------------------------------------- driver ---

CONFIGS = [("v5", "-"), ("v6", "s"), ("v6", "m")]


def print_table(rows):
    header = f"| {'version':<7} | {'size':<4} | {'imgs':>4} | {'accuracy(%)':>11} | {'FPS':>6} | {'ms/img':>7} | {'lines':>5} |"
    sep = "|" + "|".join("-" * (len(part) ) for part in header.split("|")[1:-1]) + "|"
    print(header)
    print(sep)
    for r in rows:
        print(f"| {r['version']:<7} | {r['size']:<4} | {r['images']:>4} | "
              f"{r['accuracy_pct']:>11} | {r['fps']:>6} | {r['ms_per_image']:>7} | "
              f"{r['avg_lines']:>5} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", choices=["v5", "v6"],
                    help="run one configuration in this process (used by the driver)")
    ap.add_argument("--size", default="m", choices=["s", "m"])
    ap.add_argument("--model-dir", default=None,
                    help="override the v6 model directory (V6_MODEL_DIR)")
    ap.add_argument("--out", default=str(HERE / "benchmark_results.json"))
    args = ap.parse_args()

    if args.single:
        result = run_single(args.single, args.size, args.model_dir)
        print("BENCHMARK_JSON " + json.dumps(result, ensure_ascii=False))
        return 0

    rows = []
    for version, size in CONFIGS:
        size_arg = size if size != "-" else "m"
        print(f"\n### running {version}/{size} in a separate process ###", flush=True)
        proc = subprocess.run(
            [sys.executable, __file__, "--single", version, "--size", size_arg],
            cwd=str(HERE), capture_output=True, text=True,
        )
        line = next((ln for ln in proc.stdout.splitlines()
                     if ln.startswith("BENCHMARK_JSON ")), None)
        if line is None:
            print(f"!!! {version}/{size} produced no result (exit {proc.returncode})")
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            continue
        rows.append(json.loads(line[len("BENCHMARK_JSON "):]))

    if not rows:
        print("no configuration produced results")
        return 1

    print()
    print_table(rows)
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    print(f"\nsaved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
