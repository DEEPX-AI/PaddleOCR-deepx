#!/usr/bin/env python3
"""Unit tests for the PP-OCRv6 crop-splitting recognition path.

v6 ships a single 48x240 recognition model (30 CTC timesteps), so a wide text
line must be cut into overlapping tiles and merged. These tests cover the split
geometry only — no NPU required.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

DEEPX = Path(__file__).resolve().parent / "deepx"
for p in (str(DEEPX), str(DEEPX / "engine")):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine.paddleocr import split_crop_for_recognition  # noqa: E402
from engine.utils import merge_recognition_results  # noqa: E402

V6_RATIO = 240 / 48  # 5.0


def _crop(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_crop_at_target_ratio_is_not_split():
    assert len(split_crop_for_recognition(_crop(48, 240), V6_RATIO)) == 1


def test_crop_just_under_threshold_is_not_split():
    # 1.3 * 5.0 = 6.5 -> ratio 6.0 must stay whole
    assert len(split_crop_for_recognition(_crop(48, 288), V6_RATIO)) == 1


def test_long_crop_is_split_into_expected_tile_count():
    tiles = split_crop_for_recognition(_crop(48, 48 * 35), V6_RATIO)  # ratio 35
    assert len(tiles) == 7  # ceil(35 / 5)


def test_tiles_keep_full_height_and_overlap():
    w = 48 * 35
    tiles = split_crop_for_recognition(_crop(48, w), V6_RATIO, overlap_ratio=0.1)
    assert all(t.shape[0] == 48 for t in tiles)
    # every tile except the last carries the 10% overlap
    step = w / 7
    assert tiles[0].shape[1] == pytest.approx(step * 1.1, abs=2)
    # tiles together cover more than the original width (overlap)
    assert sum(t.shape[1] for t in tiles) > w


def test_degenerate_crops_return_original():
    assert len(split_crop_for_recognition(_crop(48, 0), V6_RATIO)) == 1
    assert len(split_crop_for_recognition(_crop(0, 240), V6_RATIO)) == 1
    assert len(split_crop_for_recognition(_crop(48, 240), 0)) == 1


def test_merge_reassembles_overlapping_tile_texts():
    # merge_recognition_results drops the overlapping prefix when it matches
    merged_text, merged_score = merge_recognition_results(
        [("HELLOWO", 0.9), ("WORLD00", 0.8)], overlap_ratio=0.0)
    assert merged_text.startswith("HELLOWO")
    assert 0.8 <= merged_score <= 0.9


def test_merge_of_single_tile_is_identity():
    text, score = merge_recognition_results([("ABC", 0.77)], overlap_ratio=0.1)
    assert text == "ABC"
    assert score == pytest.approx(0.77)
