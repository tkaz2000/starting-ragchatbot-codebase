"""Test doubles shared by the backend test modules."""
import json
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import httpx

from vector_store import SearchResults


def make_results(items: List[Tuple[str, Dict[str, Any]]]) -> SearchResults:
    """Build SearchResults from (document, metadata) pairs."""
    return SearchResults(
        documents=[doc for doc, _ in items],
        metadata=[meta for _, meta in items],
        distances=[0.1] * len(items),
    )


class FakeStore:
    """In-memory stand-in for VectorStore, so CourseSearchTool/CourseOutlineTool can be tested alone."""

    def __init__(self):
        self.next_results: SearchResults = SearchResults(documents=[], metadata=[], distances=[])
        self.search_calls: List[Dict[str, Any]] = []
        self.lesson_links: Dict[Tuple[str, int], str] = {}
        self.course_links: Dict[str, str] = {}
        self.outline: Optional[Dict[str, Any]] = None

    def search(self, query, course_name=None, lesson_number=None, limit=None):
        self.search_calls.append({"query": query, "course_name": course_name, "lesson_number": lesson_number})
        return self.next_results

    def get_lesson_link(self, course_title, lesson_number):
        return self.lesson_links.get((course_title, lesson_number))

    def get_course_link(self, course_title):
        return self.course_links.get(course_title)

    def get_course_outline(self, course_name):
        return self.outline


class ScriptedAnthropic:
    """Stands in for the Anthropic Messages API.

    The real anthropic SDK is used, but its HTTP transport is replaced, so the request
    bodies the code produces are recorded in `requests` and the replies are scripted.
    Each scripted item is a message dict (HTTP 200) or a (status_code, error_body) tuple.
    """

    def __init__(self, script: List[Any]):
        self.script = list(script)
        self.requests: List[Dict[str, Any]] = []

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        if not self.script:
            return httpx.Response(599, json={"type": "error", "error": {
                "type": "api_error", "message": "test made more API calls than scripted"}})
        item = self.script.pop(0)
        if isinstance(item, tuple):
            status, body = item
            return httpx.Response(status, json=body)
        return httpx.Response(200, json=item)

    def client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(
            api_key="test-key",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(self._handle)),
        )


def api_error(status: int = 500, message: str = "simulated API failure") -> Tuple[int, Dict[str, Any]]:
    return status, {"type": "error", "error": {"type": "api_error", "message": message}}


def _message(content: List[Dict[str, Any]], stop_reason: str) -> Dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "test-model",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def text_reply(text: str) -> Dict[str, Any]:
    return _message([{"type": "text", "text": text}], "end_turn")


def tool_reply(name: str, tool_input: Dict[str, Any], tool_id: str = "toolu_test_1") -> Dict[str, Any]:
    return _message([{"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}], "tool_use")
