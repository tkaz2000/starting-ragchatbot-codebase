"""Part 3: evaluate how RAGSystem.query handles content-related questions.

These tests use the real ChromaDB built from ../docs (needs the embedding model) and a
scripted Anthropic API, so they show what happens inside one request without spending tokens.
"""
import anthropic
import pytest

from helpers import ScriptedAnthropic, api_error, text_reply, tool_reply


@pytest.fixture(autouse=True)
def _clear_tool_sources(built_rag_system):
    # The RAGSystem is session-scoped; make sure no sources leak in from an earlier test.
    built_rag_system.tool_manager.reset_sources()
    yield
    built_rag_system.tool_manager.reset_sources()


@pytest.fixture
def use_script(built_rag_system, monkeypatch):
    """Route the RAGSystem's Claude calls through a scripted API. Returns a function taking the script."""
    def _use(script):
        api = ScriptedAnthropic(script)
        monkeypatch.setattr(built_rag_system.ai_generator, "client", api.client())
        return api
    return _use


CONTENT_CALL = tool_reply("search_course_content", {"query": "what is MCP", "course_name": "MCP"})


def test_content_question_is_answered_from_retrieved_course_text(built_rag_system, use_script):
    api = use_script([CONTENT_CALL, text_reply("MCP is an open protocol.")])
    answer, sources = built_rag_system.query("What is MCP?")

    assert answer == "MCP is an open protocol."
    # The retrieved chunks, not just the answer, must reach the model.
    tool_result = api.requests[1]["messages"][2]["content"][0]["content"]
    assert "[MCP: Build Rich-Context" in tool_result
    # The UI gets citations with labels and lesson links.
    assert sources
    assert all(s["label"].startswith("MCP: Build Rich-Context") for s in sources)
    assert all(s["url"] and s["url"].startswith("https://") for s in sources)


def test_query_is_wrapped_in_course_materials_prompt(built_rag_system, use_script):
    api = use_script([text_reply("ok")])
    built_rag_system.query("What is MCP?")
    assert api.requests[0]["messages"] == [
        {"role": "user", "content": "Answer this question about course materials: What is MCP?"}
    ]


def test_general_question_skips_search_and_returns_no_sources(built_rag_system, use_script):
    api = use_script([text_reply("Paris.")])
    answer, sources = built_rag_system.query("What is the capital of France?")
    assert answer == "Paris."
    assert sources == []
    assert len(api.requests) == 1
    assert "tools" in api.requests[0]


def test_sources_do_not_carry_over_to_the_next_query(built_rag_system, use_script):
    use_script([CONTENT_CALL, text_reply("MCP answer")])
    _, first_sources = built_rag_system.query("What is MCP?")
    assert first_sources

    use_script([text_reply("Paris.")])
    _, second_sources = built_rag_system.query("What is the capital of France?")
    assert second_sources == []


def test_conversation_history_is_sent_on_the_next_turn(built_rag_system, use_script):
    session_id = built_rag_system.session_manager.create_session()

    use_script([text_reply("A1")])
    built_rag_system.query("Q1", session_id)

    api = use_script([text_reply("A2")])
    built_rag_system.query("Q2", session_id)

    system = api.requests[0]["system"]
    assert "User: Q1" in system
    assert "Assistant: A1" in system


def test_session_records_the_raw_question_and_answer(built_rag_system, use_script):
    session_id = built_rag_system.session_manager.create_session()
    use_script([CONTENT_CALL, text_reply("MCP answer")])
    built_rag_system.query("What is MCP?", session_id)

    history = built_rag_system.session_manager.sessions[session_id]
    assert [(m.role, m.content) for m in history] == [
        ("user", "What is MCP?"),
        ("assistant", "MCP answer"),
    ]


def test_api_failure_propagates_so_the_endpoint_returns_500(built_rag_system, use_script):
    # app.py turns any exception here into HTTP 500, which script.js shows as "Query failed".
    use_script([api_error(500, "boom")])
    with pytest.raises(anthropic.APIStatusError):
        built_rag_system.query("What is MCP?")


def test_failed_tool_turn_does_not_leak_sources_into_the_next_query(built_rag_system, use_script):
    # BUG CHECK: sources are only cleared on the success path of RAGSystem.query. If the
    # second Claude call fails after the search ran, the next query returns stale citations.
    use_script([CONTENT_CALL, api_error(500, "boom")])
    with pytest.raises(anthropic.APIStatusError):
        built_rag_system.query("What is MCP?")

    use_script([text_reply("Hello!")])
    _, sources = built_rag_system.query("Hi there")
    assert sources == []
