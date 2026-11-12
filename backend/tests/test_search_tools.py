"""Part 1: evaluate the output of CourseSearchTool.execute (and its sibling tools).

Unit tests use FakeStore and need no embedding model. Integration tests at the bottom
use the real ChromaDB built from ../docs and need the embedding model to load.
"""
import re

import pytest

from helpers import make_results
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults


# ---------------------------------------------------------------------------
# Unit: CourseSearchTool.execute formatting, filters, errors
# ---------------------------------------------------------------------------

def test_result_has_course_and_lesson_header(fake_store):
    fake_store.next_results = make_results([
        ("Chunk A text", {"course_title": "MCP Course", "lesson_number": 2, "chunk_index": 0}),
    ])
    out = CourseSearchTool(fake_store).execute(query="anything")
    assert out == "[MCP Course - Lesson 2]\nChunk A text"


def test_header_omits_lesson_when_metadata_has_none(fake_store):
    fake_store.next_results = make_results([
        ("Intro text", {"course_title": "MCP Course", "lesson_number": None, "chunk_index": 0}),
    ])
    out = CourseSearchTool(fake_store).execute(query="anything")
    assert out == "[MCP Course]\nIntro text"


def test_multiple_results_keep_rank_order_and_are_separated(fake_store):
    fake_store.next_results = make_results([
        ("first", {"course_title": "C", "lesson_number": 1, "chunk_index": 0}),
        ("second", {"course_title": "C", "lesson_number": 2, "chunk_index": 1}),
    ])
    out = CourseSearchTool(fake_store).execute(query="anything")
    assert out == "[C - Lesson 1]\nfirst\n\n[C - Lesson 2]\nsecond"


def test_query_and_filters_are_passed_to_store(fake_store):
    fake_store.next_results = make_results([("x", {"course_title": "C", "lesson_number": 1})])
    CourseSearchTool(fake_store).execute(query="how do tools work", course_name="MCP", lesson_number=1)
    assert fake_store.search_calls == [
        {"query": "how do tools work", "course_name": "MCP", "lesson_number": 1}
    ]


def test_lesson_zero_filter_is_passed_to_store(fake_store):
    fake_store.next_results = make_results([("x", {"course_title": "C", "lesson_number": 0})])
    CourseSearchTool(fake_store).execute(query="intro", lesson_number=0)
    assert fake_store.search_calls[0]["lesson_number"] == 0


def test_store_error_is_returned_verbatim(fake_store):
    fake_store.next_results = SearchResults.empty("No course found matching 'zzz'")
    out = CourseSearchTool(fake_store).execute(query="q", course_name="zzz")
    assert out == "No course found matching 'zzz'"


def test_empty_results_message_names_the_filters(fake_store):
    fake_store.next_results = SearchResults(documents=[], metadata=[], distances=[])
    out = CourseSearchTool(fake_store).execute(query="q", course_name="MCP", lesson_number=2)
    assert out == "No relevant content found in course 'MCP' in lesson 2."


def test_empty_results_message_mentions_lesson_zero(fake_store):
    # BUG CHECK: execute() tests `if lesson_number:`, which is False for 0, so the
    # filter is silently dropped from the message.
    fake_store.next_results = SearchResults(documents=[], metadata=[], distances=[])
    out = CourseSearchTool(fake_store).execute(query="q", lesson_number=0)
    assert "lesson 0" in out


# ---------------------------------------------------------------------------
# Unit: source tracking (drives the citations shown in the UI)
# ---------------------------------------------------------------------------

def test_source_uses_lesson_link_when_available(fake_store):
    fake_store.next_results = make_results([("t", {"course_title": "MCP", "lesson_number": 2, "chunk_index": 0})])
    fake_store.lesson_links[("MCP", 2)] = "https://example.com/lesson-2"
    tool = CourseSearchTool(fake_store)
    tool.execute(query="q")
    assert tool.last_sources == [{"label": "MCP - Lesson 2", "url": "https://example.com/lesson-2"}]


def test_source_falls_back_to_course_link(fake_store):
    fake_store.next_results = make_results([("t", {"course_title": "MCP", "lesson_number": 2, "chunk_index": 0})])
    fake_store.course_links["MCP"] = "https://example.com/course"
    tool = CourseSearchTool(fake_store)
    tool.execute(query="q")
    assert tool.last_sources == [{"label": "MCP - Lesson 2", "url": "https://example.com/course"}]


def test_source_url_is_none_when_link_is_not_http(fake_store):
    fake_store.next_results = make_results([("t", {"course_title": "MCP", "lesson_number": 2, "chunk_index": 0})])
    fake_store.lesson_links[("MCP", 2)] = "javascript:alert(1)"
    tool = CourseSearchTool(fake_store)
    tool.execute(query="q")
    assert tool.last_sources == [{"label": "MCP - Lesson 2", "url": None}]


def test_lesson_zero_source_uses_its_own_link(fake_store):
    fake_store.next_results = make_results([("t", {"course_title": "MCP", "lesson_number": 0, "chunk_index": 0})])
    fake_store.lesson_links[("MCP", 0)] = "https://example.com/lesson-0"
    tool = CourseSearchTool(fake_store)
    tool.execute(query="q")
    assert tool.last_sources == [{"label": "MCP - Lesson 0", "url": "https://example.com/lesson-0"}]


def test_source_without_lesson_is_labelled_by_course(fake_store):
    fake_store.next_results = make_results([("t", {"course_title": "MCP", "lesson_number": None, "chunk_index": 0})])
    fake_store.course_links["MCP"] = "https://example.com/course"
    tool = CourseSearchTool(fake_store)
    tool.execute(query="q")
    assert tool.last_sources == [{"label": "MCP", "url": "https://example.com/course"}]


def test_multiple_chunks_from_one_lesson_give_one_source(fake_store):
    fake_store.next_results = make_results([
        ("a", {"course_title": "MCP", "lesson_number": 2, "chunk_index": 0}),
        ("b", {"course_title": "MCP", "lesson_number": 2, "chunk_index": 1}),
    ])
    tool = CourseSearchTool(fake_store)
    tool.execute(query="q")
    assert len(tool.last_sources) == 1


# ---------------------------------------------------------------------------
# Unit: CourseOutlineTool
# ---------------------------------------------------------------------------

OUTLINE = {
    "title": "MCP: Build Rich-Context AI Apps",
    "course_link": "https://example.com/mcp",
    "lessons": [
        {"lesson_number": 0, "lesson_title": "Introduction"},
        {"lesson_number": 1, "lesson_title": "Why MCP"},
    ],
}


def test_outline_lists_title_link_and_lessons_in_order(fake_store):
    fake_store.outline = OUTLINE
    out = CourseOutlineTool(fake_store).execute(course_name="MCP")
    assert out == (
        "Course: MCP: Build Rich-Context AI Apps\n"
        "Course Link: https://example.com/mcp\n"
        "Lessons (2):\n"
        "- Lesson 0: Introduction\n"
        "- Lesson 1: Why MCP"
    )


def test_outline_without_course_link_says_not_available(fake_store):
    fake_store.outline = {**OUTLINE, "course_link": None}
    out = CourseOutlineTool(fake_store).execute(course_name="MCP")
    assert "Course Link: Not available" in out


def test_outline_not_found_message(fake_store):
    fake_store.outline = None
    assert CourseOutlineTool(fake_store).execute(course_name="nope") == "No course found matching 'nope'"


# ---------------------------------------------------------------------------
# Unit: ToolManager registration, schemas and source handoff
# ---------------------------------------------------------------------------

def test_both_tools_register_with_valid_schemas(fake_store):
    tm = ToolManager()
    tm.register_tool(CourseSearchTool(fake_store))
    tm.register_tool(CourseOutlineTool(fake_store))
    defs = {d["name"]: d for d in tm.get_tool_definitions()}
    assert set(defs) == {"search_course_content", "get_course_outline"}
    for d in defs.values():
        schema = d["input_schema"]
        assert schema["type"] == "object"
        assert set(schema.get("required", [])) <= set(schema["properties"])


def test_unknown_tool_returns_not_found_text(fake_store):
    tm = ToolManager()
    tm.register_tool(CourseSearchTool(fake_store))
    assert tm.execute_tool("web_search", query="x") == "Tool 'web_search' not found"


def test_sources_are_returned_then_reset(fake_store):
    fake_store.next_results = make_results([("t", {"course_title": "MCP", "lesson_number": 1, "chunk_index": 0})])
    tm = ToolManager()
    tm.register_tool(CourseSearchTool(fake_store))
    tm.execute_tool("search_course_content", query="q")
    assert tm.get_last_sources() == [{"label": "MCP - Lesson 1", "url": None}]
    tm.reset_sources()
    assert tm.get_last_sources() == []


# ---------------------------------------------------------------------------
# Integration: real ChromaDB built from ../docs (needs the embedding model)
# ---------------------------------------------------------------------------

@pytest.fixture
def real_search_tool(built_rag_system):
    return CourseSearchTool(built_rag_system.vector_store)


@pytest.fixture
def real_outline_tool(built_rag_system):
    return CourseOutlineTool(built_rag_system.vector_store)


def _headers(out: str):
    return re.findall(r"^\[(.*)\]$", out, flags=re.MULTILINE)


def test_real_content_query_returns_formatted_results(real_search_tool):
    out = real_search_tool.execute(query="What is MCP?")
    assert not out.startswith("No ")
    assert _headers(out), f"expected '[course - Lesson N]' headers, got: {out[:200]!r}"


def test_real_partial_course_name_restricts_results_to_that_course(real_search_tool):
    out = real_search_tool.execute(query="how do MCP servers expose tools", course_name="MCP")
    headers = _headers(out)
    assert headers
    assert all(h.startswith("MCP: Build Rich-Context") for h in headers), headers


@pytest.mark.parametrize("partial, expected_prefix", [
    ("Computer Use", "Building Towards Computer Use"),
    ("Prompt Compression", "Prompt Compression and Query Optimization"),
    ("Advanced Retrieval", "Advanced Retrieval for AI with Chroma"),
])
def test_real_partial_course_names_resolve_to_the_right_course(real_search_tool, partial, expected_prefix):
    out = real_search_tool.execute(query="overview", course_name=partial)
    headers = _headers(out)
    assert headers, out[:200]
    assert all(h.startswith(expected_prefix) for h in headers), headers


def test_real_lesson_filter_returns_only_that_lesson(real_search_tool):
    out = real_search_tool.execute(query="introduction", course_name="MCP", lesson_number=1)
    headers = _headers(out)
    assert headers
    assert all(h.endswith("- Lesson 1") for h in headers), headers


def test_real_lesson_zero_filter_returns_lesson_zero(real_search_tool):
    out = real_search_tool.execute(query="introduction", course_name="MCP", lesson_number=0)
    headers = _headers(out)
    assert headers, out[:200]
    assert all(h.endswith("- Lesson 0") for h in headers), headers


def test_real_unknown_course_name_is_reported_not_silently_matched(real_search_tool):
    # DESIGN CHECK: _resolve_course_name takes the nearest catalog entry with no
    # similarity threshold, so a made-up course name currently maps to some real course.
    out = real_search_tool.execute(query="what is it", course_name="Quantum Basket Weaving")
    assert out.startswith("No course found matching"), out[:200]


def test_real_sources_point_at_lesson_links(real_search_tool):
    real_search_tool.execute(query="What is MCP?", course_name="MCP")
    sources = real_search_tool.last_sources
    assert sources
    assert all(s["label"].startswith("MCP: Build Rich-Context") for s in sources)
    assert all(s["url"] and s["url"].startswith("https://") for s in sources), sources


def test_real_outline_lesson_count_matches_listed_lessons(real_outline_tool):
    out = real_outline_tool.execute(course_name="MCP")
    count = int(re.search(r"Lessons \((\d+)\)", out).group(1))
    assert count > 0
    assert len(re.findall(r"^- Lesson \d+: ", out, flags=re.MULTILINE)) == count
