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
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BACKEND_DIR.parent / "docs"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as config_module  # noqa: E402
from api_test_app import create_test_app  # noqa: E402
from rag_system import RAGSystem  # noqa: E402
from session_manager import SessionManager  # noqa: E402


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


@pytest.fixture
def fake_rag():
    """Stands in for RAGSystem in API tests: no ChromaDB, no Claude calls.

    session_manager is real, so session creation and deletion behave as they do in the app.
    query() and get_course_analytics() return harmless defaults; tests override them as needed.
    """
    rag = MagicMock(name="RAGSystem")
    rag.session_manager = SessionManager(max_history=2)
    rag.query.return_value = ("A test answer.", [])
    rag.get_course_analytics.return_value = {"total_courses": 0, "course_titles": []}
    return rag


@pytest.fixture
def frontend_dir(tmp_path):
    """A small stand-in for ../frontend, so static serving can be tested without the real files."""
    site = tmp_path / "frontend"
    site.mkdir()
    (site / "index.html").write_text(
        "<!DOCTYPE html><html><head><title>Course Materials</title></head>"
        "<body><div id=\"app\"></div></body></html>",
        encoding="utf-8",
    )
    (site / "app.js").write_text("console.log('test asset');", encoding="utf-8")
    return site


@pytest.fixture
def api_client(fake_rag, frontend_dir):
    """A TestClient for the API routes, wired to fake_rag and the stand-in frontend."""
    return TestClient(create_test_app(fake_rag, frontend_dir))


@pytest.fixture
def sample_course_analytics():
    """Analytics in the shape RAGSystem.get_course_analytics() returns."""
    return {
        "total_courses": 2,
        "course_titles": [
            "Building Towards Computer Use with Anthropic",
            "MCP: Build Rich-Context AI Apps with Anthropic",
        ],
    }


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
