"""Part 2: evaluate whether AIGenerator calls CourseSearchTool (and the other tools) correctly.

The real anthropic SDK is used, with its HTTP transport scripted (see helpers.ScriptedAnthropic),
so these tests inspect the exact request bodies the generator sends. No network or embedding model needed.
"""
import anthropic
import pytest

from ai_generator import AIGenerator
from helpers import ScriptedAnthropic, api_error, make_results, text_reply, tool_reply
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager

OUTLINE = {
    "title": "MCP: Build Rich-Context AI Apps",
    "course_link": "https://example.com/mcp",
    "lessons": [{"lesson_number": 0, "lesson_title": "Introduction"},
                {"lesson_number": 1, "lesson_title": "Why MCP"}],
}


@pytest.fixture
def tool_manager(fake_store):
    tm = ToolManager()
    tm.register_tool(CourseSearchTool(fake_store))
    tm.register_tool(CourseOutlineTool(fake_store))
    return tm


def make_generator(script):
    gen = AIGenerator(api_key="test-key", model="test-model")
    api = ScriptedAnthropic(script)
    gen.client = api.client()
    return gen, api


def ask(gen, tool_manager, query, history=None):
    return gen.generate_response(
        query=query,
        conversation_history=history,
        tools=tool_manager.get_tool_definitions(),
        tool_manager=tool_manager,
    )


# ---------------------------------------------------------------------------
# What the first request looks like
# ---------------------------------------------------------------------------

def test_first_request_offers_both_tools_with_auto_choice(tool_manager):
    gen, api = make_generator([text_reply("ok")])
    ask(gen, tool_manager, "What is MCP?")
    req = api.requests[0]
    assert {t["name"] for t in req["tools"]} == {"search_course_content", "get_course_outline"}
    assert req["tool_choice"] == {"type": "auto"}


def test_request_uses_configured_model_and_system_prompt(tool_manager):
    gen, api = make_generator([text_reply("ok")])
    ask(gen, tool_manager, "What is MCP?")
    req = api.requests[0]
    assert req["model"] == "test-model"
    assert req["system"] == AIGenerator.SYSTEM_PROMPT
    assert req["messages"] == [{"role": "user", "content": "What is MCP?"}]


def test_conversation_history_is_appended_to_system_prompt(tool_manager):
    gen, api = make_generator([text_reply("ok")])
    ask(gen, tool_manager, "And the next one?", history="User: hi\nAssistant: hello")
    assert api.requests[0]["system"].endswith("Previous conversation:\nUser: hi\nAssistant: hello")


# ---------------------------------------------------------------------------
# Routing: when does the model's answer lead to a tool call?
# ---------------------------------------------------------------------------

def test_general_question_answers_in_one_call_without_searching(tool_manager, fake_store):
    gen, api = make_generator([text_reply("Paris.")])
    answer = ask(gen, tool_manager, "What is the capital of France?")
    assert answer == "Paris."
    assert len(api.requests) == 1
    assert fake_store.search_calls == []


def test_content_question_runs_search_with_the_models_arguments(tool_manager, fake_store):
    fake_store.next_results = make_results([("chunk", {"course_title": "MCP", "lesson_number": 2, "chunk_index": 0})])
    gen, api = make_generator([
        tool_reply("search_course_content", {"query": "how does MCP transport work", "course_name": "MCP", "lesson_number": 2}),
        text_reply("final answer"),
    ])
    answer = ask(gen, tool_manager, "How does MCP transport work in lesson 2 of the MCP course?")
    assert answer == "final answer"
    assert fake_store.search_calls == [
        {"query": "how does MCP transport work", "course_name": "MCP", "lesson_number": 2}
    ]


def test_outline_question_routes_to_outline_tool(tool_manager, fake_store):
    fake_store.outline = OUTLINE
    gen, api = make_generator([
        tool_reply("get_course_outline", {"course_name": "MCP"}),
        text_reply("outline answer"),
    ])
    assert ask(gen, tool_manager, "What lessons are in the MCP course?") == "outline answer"
    assert fake_store.search_calls == []
    tool_result = api.requests[1]["messages"][2]["content"][0]
    assert tool_result["content"].startswith("Course: MCP: Build Rich-Context AI Apps")


# ---------------------------------------------------------------------------
# The tool-use round trip: second request must carry the tool result
# ---------------------------------------------------------------------------

def test_second_round_request_still_offers_tools(tool_manager, fake_store):
    # After one tool round, Claude may still call another tool, so tools stay on the next request.
    fake_store.next_results = make_results([("c", {"course_title": "MCP", "lesson_number": 1, "chunk_index": 0})])
    gen, api = make_generator([
        tool_reply("search_course_content", {"query": "q"}),
        text_reply("done"),
    ])
    ask(gen, tool_manager, "anything about MCP")
    assert {t["name"] for t in api.requests[1]["tools"]} == {"search_course_content", "get_course_outline"}


def test_two_sequential_rounds_feed_the_first_result_into_the_second_search(tool_manager, fake_store):
    # The example from the spec: read lesson 1's title from the outline, then search for that topic.
    fake_store.outline = OUTLINE
    fake_store.next_results = make_results([("Other course text", {"course_title": "Other", "lesson_number": 3, "chunk_index": 0})])
    gen, api = make_generator([
        tool_reply("get_course_outline", {"course_name": "MCP"}, tool_id="toolu_round1"),
        tool_reply("search_course_content", {"query": "Why MCP"}, tool_id="toolu_round2"),
        text_reply("Other covers the same topic."),
    ])

    answer = ask(gen, tool_manager, "Find a course about the same topic as lesson 1 of MCP")

    assert answer == "Other covers the same topic."
    assert fake_store.search_calls == [{"query": "Why MCP", "course_name": None, "lesson_number": None}]
    assert len(api.requests) == 3
    # The final request carries both rounds, in order, each result paired with its tool_use id
    messages = api.requests[2]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[2]["content"][0]["tool_use_id"] == "toolu_round1"
    assert messages[2]["content"][0]["content"].startswith("Course: MCP: Build Rich-Context AI Apps")
    assert messages[4]["content"][0]["tool_use_id"] == "toolu_round2"
    assert messages[4]["content"][0]["content"] == "[Other - Lesson 3]\nOther course text"


def test_round_cap_stops_tool_use_and_the_last_request_offers_no_tools(tool_manager, fake_store):
    fake_store.outline = OUTLINE
    fake_store.next_results = make_results([("c", {"course_title": "MCP", "lesson_number": 1, "chunk_index": 0})])
    gen, api = make_generator([
        tool_reply("get_course_outline", {"course_name": "MCP"}, tool_id="toolu_1"),
        tool_reply("search_course_content", {"query": "q"}, tool_id="toolu_2"),
        text_reply("final answer"),
    ])
    assert ask(gen, tool_manager, "question") == "final answer"
    assert len(fake_store.search_calls) == 1
    assert len(api.requests) == 3
    assert "tools" not in api.requests[2]


def test_tool_request_after_the_round_cap_is_not_executed(tool_manager, fake_store):
    # If the final request still returns a tool_use block, it must not run, and the reply falls back to text.
    fake_store.outline = OUTLINE
    fake_store.next_results = make_results([("c", {"course_title": "MCP", "lesson_number": 1, "chunk_index": 0})])
    gen, api = make_generator([
        tool_reply("get_course_outline", {"course_name": "MCP"}, tool_id="toolu_1"),
        tool_reply("search_course_content", {"query": "q"}, tool_id="toolu_2"),
        tool_reply("search_course_content", {"query": "third"}, tool_id="toolu_3"),
    ])
    answer = ask(gen, tool_manager, "question")
    assert isinstance(answer, str) and answer.strip()
    assert len(fake_store.search_calls) == 1
    assert len(api.requests) == 3


def test_stops_when_a_reply_has_no_tool_use(tool_manager, fake_store):
    fake_store.next_results = make_results([("c", {"course_title": "MCP", "lesson_number": 1, "chunk_index": 0})])
    gen, api = make_generator([
        tool_reply("search_course_content", {"query": "q"}),
        text_reply("answer after one round"),
    ])
    assert ask(gen, tool_manager, "question") == "answer after one round"
    assert len(api.requests) == 2
    assert len(fake_store.search_calls) == 1


def test_not_found_result_does_not_stop_the_loop(tool_manager, fake_store):
    # "No course found" is a normal result, not a failure, so Claude can retry in round two.
    fake_store.outline = None
    fake_store.next_results = make_results([("c", {"course_title": "MCP", "lesson_number": 1, "chunk_index": 0})])
    gen, api = make_generator([
        tool_reply("get_course_outline", {"course_name": "Mcp typo"}, tool_id="toolu_1"),
        tool_reply("search_course_content", {"query": "q"}, tool_id="toolu_2"),
        text_reply("done"),
    ])
    assert ask(gen, tool_manager, "question") == "done"
    assert len(fake_store.search_calls) == 1
    assert len(api.requests) == 3


def test_tool_exception_stops_the_loop_and_reports_an_error_result(tool_manager):
    gen, api = make_generator([
        tool_reply("search_course_content", {"query": "x", "not_a_parameter": 1}),
        text_reply("I could not search for that."),
    ])
    assert ask(gen, tool_manager, "question") == "I could not search for that."
    assert len(api.requests) == 2
    assert "tools" not in api.requests[1]
    result = api.requests[1]["messages"][2]["content"][0]
    assert result["is_error"] is True
    assert "error" in result["content"].lower()


def test_unknown_tool_name_stops_the_loop_and_is_marked_as_error(tool_manager):
    gen, api = make_generator([tool_reply("web_search", {"query": "x"}), text_reply("I cannot search the web.")])
    assert ask(gen, tool_manager, "question") == "I cannot search the web."
    assert "tools" not in api.requests[1]
    result = api.requests[1]["messages"][2]["content"][0]
    assert result["content"] == "Tool 'web_search' not found"
    assert result["is_error"] is True


def test_empty_tool_output_is_replaced_with_a_placeholder(fake_store):
    class EmptyTool:
        def get_tool_definition(self):
            return {"name": "empty_tool", "description": "returns nothing",
                    "input_schema": {"type": "object", "properties": {}}}

        def execute(self, **kwargs):
            return ""

    tm = ToolManager()
    tm.register_tool(EmptyTool())
    gen, api = make_generator([tool_reply("empty_tool", {}), text_reply("ok")])
    ask(gen, tm, "question")
    result = api.requests[1]["messages"][2]["content"][0]
    assert result["content"].strip() != ""


def test_system_prompt_no_longer_limits_to_one_tool_call():
    assert "One tool call per query maximum" not in AIGenerator.SYSTEM_PROMPT
    assert "Up to two rounds of tool calls" in AIGenerator.SYSTEM_PROMPT


def test_requests_allow_1000_output_tokens(tool_manager):
    gen, api = make_generator([text_reply("ok")])
    ask(gen, tool_manager, "question")
    assert api.requests[0]["max_tokens"] == 1000


def test_follow_up_request_carries_tool_use_and_matching_tool_result(tool_manager, fake_store):
    fake_store.next_results = make_results([
        ("Chunk text", {"course_title": "MCP", "lesson_number": 2, "chunk_index": 0}),
    ])
    gen, api = make_generator([
        tool_reply("search_course_content", {"query": "q", "course_name": "MCP"}, tool_id="toolu_abc"),
        text_reply("done"),
    ])
    ask(gen, tool_manager, "question")

    messages = api.requests[1]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]

    assistant_block = messages[1]["content"][0]
    assert assistant_block["type"] == "tool_use"
    assert assistant_block["id"] == "toolu_abc"
    assert assistant_block["name"] == "search_course_content"
    assert assistant_block["input"] == {"query": "q", "course_name": "MCP"}

    result_block = messages[2]["content"][0]
    assert result_block["type"] == "tool_result"
    assert result_block["tool_use_id"] == "toolu_abc"
    assert result_block["content"] == "[MCP - Lesson 2]\nChunk text"


def test_follow_up_request_keeps_the_system_prompt(tool_manager, fake_store):
    fake_store.next_results = make_results([("c", {"course_title": "MCP", "lesson_number": 1, "chunk_index": 0})])
    gen, api = make_generator([tool_reply("search_course_content", {"query": "q"}), text_reply("done")])
    ask(gen, tool_manager, "question", history="User: earlier")
    assert api.requests[1]["system"] == api.requests[0]["system"]


def test_every_tool_use_in_a_reply_gets_a_tool_result(tool_manager, fake_store):
    fake_store.next_results = make_results([("c", {"course_title": "MCP", "lesson_number": 1, "chunk_index": 0})])
    fake_store.outline = OUTLINE
    two_calls = {
        "id": "msg_test", "type": "message", "role": "assistant", "model": "test-model",
        "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "search_course_content", "input": {"query": "q"}},
            {"type": "tool_use", "id": "toolu_2", "name": "get_course_outline", "input": {"course_name": "MCP"}},
        ],
        "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    gen, api = make_generator([two_calls, text_reply("done")])
    ask(gen, tool_manager, "question")
    results = api.requests[1]["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["toolu_1", "toolu_2"]


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

def test_unknown_tool_name_is_reported_to_the_model(tool_manager):
    gen, api = make_generator([tool_reply("web_search", {"query": "x"}), text_reply("I cannot search the web.")])
    assert ask(gen, tool_manager, "question") == "I cannot search the web."
    assert api.requests[1]["messages"][2]["content"][0]["content"] == "Tool 'web_search' not found"


def test_tool_exception_is_reported_to_the_model_not_raised(tool_manager):
    # BUG CHECK: a bad argument from the model (e.g. a key not in the schema) makes
    # execute(**input) raise TypeError. Nothing catches it, so the whole request fails
    # and the API returns 500, which the frontend shows as "Query failed".
    gen, api = make_generator([
        tool_reply("search_course_content", {"query": "x", "not_a_parameter": 1}),
        text_reply("I could not search for that."),
    ])
    assert ask(gen, tool_manager, "question") == "I could not search for that."
    result = api.requests[1]["messages"][2]["content"][0]["content"]
    assert "error" in result.lower()


def test_api_error_on_first_call_propagates_to_caller(tool_manager):
    gen, _ = make_generator([api_error(500, "boom")])
    with pytest.raises(anthropic.APIStatusError):
        ask(gen, tool_manager, "question")


def test_api_error_on_follow_up_call_propagates_to_caller(tool_manager, fake_store):
    fake_store.next_results = make_results([("c", {"course_title": "MCP", "lesson_number": 1, "chunk_index": 0})])
    gen, _ = make_generator([tool_reply("search_course_content", {"query": "q"}), api_error(500, "boom")])
    with pytest.raises(anthropic.APIStatusError):
        ask(gen, tool_manager, "question")
