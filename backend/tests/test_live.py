"""Live checks against the real Anthropic API. Skipped unless RUN_LIVE=1.

These are the checks that match the real "Query failed" symptom: they fail with the
actual error the app would raise (for example a 404 for a retired model, or a
connection error from TLS certificate verification).
"""
import anthropic
import pytest

from config import config as app_config

pytestmark = pytest.mark.live


def test_configured_model_is_accepted_by_anthropic():
    client = anthropic.Anthropic(api_key=app_config.ANTHROPIC_API_KEY, max_retries=0)
    client.messages.create(
        model=app_config.ANTHROPIC_MODEL,
        max_tokens=5,
        messages=[{"role": "user", "content": "ping"}],
    )


def test_live_content_question_end_to_end(built_rag_system, monkeypatch):
    client = anthropic.Anthropic(api_key=app_config.ANTHROPIC_API_KEY, max_retries=0)
    monkeypatch.setattr(built_rag_system.ai_generator, "client", client)
    built_rag_system.tool_manager.reset_sources()

    answer, sources = built_rag_system.query("What is MCP?")

    assert answer.strip()
    assert sources, "a course question should return citations"
