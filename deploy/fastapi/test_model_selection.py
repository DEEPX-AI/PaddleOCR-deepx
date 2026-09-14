#!/usr/bin/env python3
"""Model version/size resolution rules shared by run.sh and ocr_service.py.

PP-OCRv5 sizes are deployment targets (mobile/server); PP-OCRv6 sizes are model
scales (tiny/small/medium). The letters overlap misleadingly - v6 "s" is small,
not server - so each version only accepts its own names.
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from model_selection import (  # noqa: E402
    DEFAULT_SIZE, DEFAULT_VERSION, SIZES, SelectionError,
    apply_to_env, normalize_size, resolve,
)


def test_size_names_per_version():
    assert SIZES["v5"] == ("mobile", "server")
    assert SIZES["v6"] == ("tiny", "small", "medium")


def test_defaults():
    assert DEFAULT_VERSION == "v6"
    assert DEFAULT_SIZE["v5"] == "server"
    assert DEFAULT_SIZE["v6"] == "medium"


@pytest.mark.parametrize("given,expected", [
    ("medium", "medium"), ("m", "medium"),
    ("small", "small"), ("s", "small"),
    ("tiny", "tiny"), ("t", "tiny"),
    ("MEDIUM", "medium"), ("  small  ", "small"),
])
def test_v6_size_accepts_full_names_and_abbreviations(given, expected):
    assert normalize_size("v6", given) == expected


@pytest.mark.parametrize("given,expected", [
    ("server", "server"), ("mobile", "mobile"), ("SERVER", "server"),
])
def test_v5_size_names(given, expected):
    assert normalize_size("v5", given) == expected


def test_v5_rejects_v6_size_and_says_what_is_valid():
    with pytest.raises(SelectionError) as e:
        normalize_size("v5", "tiny")
    msg = str(e.value)
    assert "v5" in msg and "mobile" in msg and "server" in msg


def test_v6_rejects_v5_size():
    # "s" is small for v6, but "server" must not silently become small.
    with pytest.raises(SelectionError):
        normalize_size("v6", "server")


def test_both_given_never_prompts():
    called = []
    got = resolve("v6", "small", isatty=True, prompt=lambda p: called.append(p) or "1")
    assert got == ("v6", "small")
    assert called == []


def test_version_only_errors_with_hint_naming_the_missing_option():
    with pytest.raises(SelectionError) as e:
        resolve("v6", None, isatty=True)
    msg = str(e.value)
    assert "--model-size" in msg
    assert "medium" in msg and "small" in msg and "tiny" in msg


def test_size_only_errors_with_hint_naming_the_missing_option():
    with pytest.raises(SelectionError) as e:
        resolve(None, "small", isatty=True)
    assert "--ocr-version" in str(e.value)


def test_non_tty_with_nothing_given_is_an_explicit_error():
    # Never guess silently: a detached container must say what it is missing
    # rather than start on an unstated model. Docker injects the choice via ENV.
    with pytest.raises(SelectionError) as e:
        resolve(None, None, isatty=False)
    msg = str(e.value)
    assert "--ocr-version" in msg and "--model-size" in msg
    assert "OCR_VERSION" in msg and "MODEL_SIZE" in msg   # env alternative


def test_defaults_are_menu_defaults_not_silent_fallbacks():
    # Pressing Enter at the prompt still yields the documented defaults.
    assert resolve(None, None, isatty=True, prompt=lambda _: "") == ("v6", "medium")


def test_tty_with_nothing_given_prompts_for_both():
    # Menus list the default first: version 2 = v5, then v5 size 2 = mobile.
    answers = iter(["2", "2"])
    assert resolve(None, None, isatty=True, prompt=lambda _: next(answers)) == ("v5", "mobile")


def test_menus_list_the_default_first():
    from model_selection import MENU_ORDER
    for version, order in MENU_ORDER.items():
        assert order[0] == DEFAULT_SIZE[version], version
        assert sorted(order) == sorted(SIZES[version]), version


def test_empty_answers_take_the_defaults():
    assert resolve(None, None, isatty=True, prompt=lambda _: "") == ("v6", "medium")


def test_prompt_reasks_on_invalid_choice():
    answers = iter(["9", "zzz", "1", "3"])   # junk, junk, v6, tiny
    assert resolve(None, None, isatty=True, prompt=lambda _: next(answers)) == ("v6", "tiny")


def test_apply_to_env_sets_what_ocr_service_reads(monkeypatch):
    monkeypatch.delenv("OCR_VERSION", raising=False)
    monkeypatch.delenv("V6_MODEL_SIZE", raising=False)
    monkeypatch.delenv("USE_MOBILE", raising=False)
    import os
    apply_to_env("v6", "small")
    assert os.environ["OCR_VERSION"] == "v6"
    assert os.environ["V6_MODEL_SIZE"] == "s"      # ocr_service reads t/s/m
    # Canonical name too, so a child process can re-resolve the same choice.
    assert os.environ["MODEL_SIZE"] == "small"
    apply_to_env("v5", "mobile")
    assert os.environ["OCR_VERSION"] == "v5"
    assert os.environ["USE_MOBILE"] == "true"
    apply_to_env("v5", "server")
    assert os.environ["USE_MOBILE"] == "false"
