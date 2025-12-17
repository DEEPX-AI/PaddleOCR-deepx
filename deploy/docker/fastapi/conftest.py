#!/usr/bin/env python3
"""
pytest configuration file for PaddleOCR FastAPI test suite

This file is automatically loaded by pytest before running tests.
It defines custom command line options and shared fixtures.
"""

import pytest


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--deepx",
        action="store_true",
        default=False,
        help="Use DEEPX NPU for inference (default: false, uses CPU)"
    )
    parser.addoption(
        "--sync",
        action="store_true",
        default=False,
        help="Use sync NPU PaddleOcr instead of async AsyncPipelineOCR (default: false, uses async)"
    )


@pytest.fixture(scope="session")
def use_deepx(request):
    """Fixture to get the --deepx flag value"""
    return request.config.getoption("--deepx")


@pytest.fixture(scope="session")
def use_sync(request):
    """Fixture to get the --sync flag value"""
    return request.config.getoption("--sync")


@pytest.fixture(scope="session")
def backend_name(use_deepx, use_sync):
    """Fixture to get backend name for output directory"""
    if use_deepx:
        return "deepx-npu-sync" if use_sync else "deepx-npu-async"
    return "cpu"
