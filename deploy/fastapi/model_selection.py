#!/usr/bin/env python3
"""OCR model version/size resolution, shared by run.sh and ocr_service.py.

PP-OCRv5 and PP-OCRv6 name their variants on different axes:

  v5  mobile | server          deployment target (edge vs server)
  v6  tiny | small | medium    model scale

The letters collide misleadingly - v6 "s" is *small*, not server, and v6 "m" is
*medium*, not mobile - so each version only accepts its own names. Keeping the
rules in one module means run.sh and ocr_service.py cannot drift apart:
ocr_service.py imports it, run.sh shells out to `--resolve`.

Run directly to resolve a selection for a shell:

    python model_selection.py --resolve --ocr-version v6 --model-size small
    OCR_VERSION=v6
    V6_MODEL_SIZE=s
    USE_MOBILE=false
"""
import os
import sys

SIZES = {
    "v5": ("mobile", "server"),
    "v6": ("tiny", "small", "medium"),
}
DEFAULT_VERSION = "v6"
DEFAULT_SIZE = {"v5": "server", "v6": "medium"}

# Shown while choosing, so the trade-off is visible at the prompt.
VERSION_BLURB = {
    "v6": "PP-OCRv6  (default, recommended)",
    "v5": "PP-OCRv5  (previous generation)",
}
SIZE_BLURB = {
    "v6": {
        "medium": "accuracy 86.5%   1.34 FPS   largest",
        "small":  "accuracy 85.3%   2.39 FPS",
        "tiny":   "accuracy 78.0%   4.95 FPS   smallest",
    },
    "v5": {
        "server": "accuracy 88.5%   1.19 FPS   largest",
        "mobile": "smaller and faster, lower accuracy",
    },
}

# Menu order puts the default first so the recommended choice is always "1",
# matching the version menu. SIZES stays in ascending-scale order because it is
# the canonical list used for validation and error messages.
MENU_ORDER = {
    "v5": ("server", "mobile"),
    "v6": ("medium", "small", "tiny"),
}

# v6 abbreviations. Deliberately no "server"/"mobile" aliases here: mapping
# v6 "server" onto a scale name would hide a real mistake.
_V6_ALIASES = {"t": "tiny", "s": "small", "m": "medium"}


class SelectionError(Exception):
    """Invalid or incomplete model selection, with a message meant for the user."""


def _usage_hint():
    return (
        "\nUsage:\n"
        "  --ocr-version v6 --model-size medium|small|tiny\n"
        "  --ocr-version v5 --model-size server|mobile\n"
        "\nOmit both to choose interactively."
    )


def normalize_size(version, size):
    """Canonical size name for a version, accepting v6 abbreviations."""
    if version not in SIZES:
        raise SelectionError(f"Unknown OCR version: {version!r} (expected v5 or v6)")
    key = str(size).strip().lower()
    if version == "v6":
        key = _V6_ALIASES.get(key, key)
    if key not in SIZES[version]:
        valid = " | ".join(SIZES[version])
        raise SelectionError(
            f"Invalid model size {size!r} for {version}. Valid sizes: {valid}"
            + _usage_hint()
        )
    return key


def _choose(prompt, title, options, blurbs, default):
    """Numbered menu; empty input takes the default, junk re-asks."""
    while True:
        lines = [f"\n{title}"]
        for i, opt in enumerate(options, 1):
            mark = "  (default)" if opt == default else ""
            lines.append(f"  {i}) {opt:<8} {blurbs.get(opt, '')}{mark}")
        sys.stderr.write("\n".join(lines) + "\n")
        answer = (prompt(f"Select [{options.index(default) + 1}]: ") or "").strip()
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        low = answer.lower()
        if low in options:
            return low
        sys.stderr.write(f"  '{answer}' is not one of the choices.\n")


def resolve(version=None, size=None, *, isatty=None, prompt=input):
    """Resolve (version, size).

    Both given      -> normalized and returned.
    Exactly one     -> SelectionError naming the missing option.
    Neither, TTY    -> interactive menus (Enter takes the documented default).
    Neither, no TTY -> SelectionError; the caller must state the model.
    """
    version = (version or "").strip().lower() or None
    size = (size or "").strip() or None

    if version and version not in SIZES:
        raise SelectionError(
            f"Unknown OCR version: {version!r} (expected v5 or v6)" + _usage_hint()
        )

    if version and size:
        return version, normalize_size(version, size)

    if version and not size:
        valid = " | ".join(SIZES[version])
        raise SelectionError(
            f"--ocr-version {version} was given but --model-size is missing.\n"
            f"Valid sizes for {version}: {valid}" + _usage_hint()
        )

    if size and not version:
        raise SelectionError(
            f"--model-size {size!r} was given but --ocr-version is missing.\n"
            f"The size names differ per version, so the version must be explicit."
            + _usage_hint()
        )

    if isatty is None:
        isatty = sys.stdin.isatty()

    if not isatty:
        # Never guess. A detached container or CI job that starts on an
        # unstated model is worse than one that refuses to start: the operator
        # has no way to tell which weights served a request.
        raise SelectionError(
            "No model selected and no terminal available to ask on.\n"
            "Specify the model explicitly, either as options:\n"
            f"  --ocr-version {DEFAULT_VERSION} --model-size {DEFAULT_SIZE[DEFAULT_VERSION]}\n"
            "or as environment variables (useful for Docker):\n"
            f"  OCR_VERSION={DEFAULT_VERSION}  MODEL_SIZE={DEFAULT_SIZE[DEFAULT_VERSION]}"
            + _usage_hint()
        )

    versions = ("v6", "v5")
    version = _choose(prompt, "OCR model version", versions, VERSION_BLURB, DEFAULT_VERSION)
    size = _choose(prompt, f"{version.upper()} model size", list(MENU_ORDER[version]),
                   SIZE_BLURB[version], DEFAULT_SIZE[version])
    sys.stderr.write(
        f"\n-> starting with {version} / {size}\n"
        f"   To skip this next time:  --ocr-version {version} --model-size {size}\n\n"
    )
    return version, size


def to_env(version, size):
    """Environment variables ocr_service.py actually reads.

    MODEL_SIZE carries the canonical size name so a resolved selection can be
    re-resolved downstream (run.sh exports these, then ocr_service.py resolves
    again in its own process). Without it the child would see a version with no
    size and reject the very selection its parent just made.
    """
    env = {"OCR_VERSION": version, "MODEL_SIZE": size}
    if version == "v6":
        env["V6_MODEL_SIZE"] = {"tiny": "t", "small": "s", "medium": "m"}[size]
    else:
        env["USE_MOBILE"] = "true" if size == "mobile" else "false"
    return env


def apply_to_env(version, size):
    os.environ.update(to_env(version, size))


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--resolve", action="store_true",
                    help="print the resolved selection as KEY=VALUE lines")
    ap.add_argument("--ocr-version", choices=["v5", "v6"])
    ap.add_argument("--model-size")
    ap.add_argument("--no-interactive", action="store_true")
    args = ap.parse_args(argv)

    interactive = (sys.stdin.isatty() and not args.no_interactive
                   and os.getenv("OCR_NONINTERACTIVE") != "1")
    # CLI wins over environment; ENV is how Docker injects the choice.
    version = args.ocr_version or os.getenv("OCR_VERSION")
    size = args.model_size or os.getenv("MODEL_SIZE")
    try:
        version, size = resolve(version, size, isatty=interactive)
    except SelectionError as exc:
        sys.stderr.write(f"\nERROR: {exc}\n\n")
        return 2

    for key, value in to_env(version, size).items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
