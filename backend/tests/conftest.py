"""Shared fixtures for the backend test suite.

Run from backend/ (uv is used, per CLAUDE.md; pytest is not a project dependency):

    uv run --with pytest python -m pytest tests

Tests marked `live` call the real Anthropic API and need network access and a valid
ANTHROPIC_API_KEY. They are skipped unless RUN_LIVE=1 is set.

The vector-store tests build a temporary ChromaDB from ../docs. The embedding model
(all-MiniLM-L6-v2) is loaded through sentence-transformers, which checks huggingface.co
on load; set HF_HUB_OFFLINE=1 to use the local cache without that check.
"""
import dataclasses
import os
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BACKEND_DIR.parent / "docs"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as config_module  # noqa: E402
from rag_system import RAGSystem  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: calls the real Anthropic API (needs network and ANTHROPIC_API_KEY; set RUN_LIVE=1)"
    )


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_LIVE") == "1":
        return
    skip_live = pytest.mark.skip(reason="live test; set RUN_LIVE=1 to call the real Anthropic API")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture
def fake_store():
    from helpers import FakeStore
    return FakeStore()


@pytest.fixture(scope="session")
def built_rag_system(tmp_path_factory):
    """A RAGSystem backed by a fresh temporary ChromaDB populated from ../docs.

    The app's own backend/chroma_db is never touched.
    """
    chroma_dir = tmp_path_factory.mktemp("chroma")
    cfg = dataclasses.replace(
        config_module.Config(),
        CHROMA_PATH=str(chroma_dir),
        ANTHROPIC_API_KEY="test-key",
    )
    rag = RAGSystem(cfg)
    courses, chunks = rag.add_course_folder(str(DOCS_DIR), clear_existing=False)
    assert courses == 4, f"expected 4 courses from {DOCS_DIR}, loaded {courses}"
    assert chunks > 0
    return rag
