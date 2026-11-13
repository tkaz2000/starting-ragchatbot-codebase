"""API endpoint tests for the routes in backend/app.py.

The routes run through api_test_app.create_test_app with a fake RAGSystem (see the api_client
fixture), so these tests make no Claude calls and open no ChromaDB.
"""
import pytest


class TestQueryEndpoint:
    def test_returns_answer_sources_and_session_id(self, api_client, fake_rag):
        fake_rag.query.return_value = (
            "MCP is an open protocol.",
            [{"label": "MCP: Build Rich-Context AI Apps - Lesson 1", "url": "https://example.com/mcp/1"}],
        )

        resp = api_client.post("/api/query", json={"query": "what is MCP", "session_id": "session_7"})

        assert resp.status_code == 200
        assert resp.json() == {
            "answer": "MCP is an open protocol.",
            "sources": [{"label": "MCP: Build Rich-Context AI Apps - Lesson 1", "url": "https://example.com/mcp/1"}],
            "session_id": "session_7",
        }
        fake_rag.query.assert_called_once_with("what is MCP", "session_7")

    @pytest.mark.parametrize("body", [{"query": "hello"}, {"query": "hello", "session_id": None},
                                      {"query": "hello", "session_id": ""}])
    def test_creates_session_when_none_given(self, api_client, fake_rag, body):
        resp = api_client.post("/api/query", json=body)

        session_id = resp.json()["session_id"]
        assert resp.status_code == 200
        assert session_id
        assert session_id in fake_rag.session_manager.sessions
        assert fake_rag.query.call_args.args == ("hello", session_id)

    def test_each_request_without_session_gets_its_own_session(self, api_client):
        first = api_client.post("/api/query", json={"query": "one"}).json()["session_id"]
        second = api_client.post("/api/query", json={"query": "two"}).json()["session_id"]

        assert first != second

    def test_follow_up_reuses_the_session_returned_by_the_first_request(self, api_client, fake_rag):
        session_id = api_client.post("/api/query", json={"query": "first"}).json()["session_id"]

        resp = api_client.post("/api/query", json={"query": "follow up", "session_id": session_id})

        assert resp.json()["session_id"] == session_id
        assert fake_rag.query.call_args.args == ("follow up", session_id)

    def test_source_without_url_is_returned_with_null_url(self, api_client, fake_rag):
        fake_rag.query.return_value = ("Answer.", [{"label": "Some Course - Lesson 2", "url": None}])

        resp = api_client.post("/api/query", json={"query": "q"})

        assert resp.json()["sources"] == [{"label": "Some Course - Lesson 2", "url": None}]

    def test_answer_without_sources_returns_empty_list(self, api_client, fake_rag):
        fake_rag.query.return_value = ("General answer.", [])

        resp = api_client.post("/api/query", json={"query": "q"})

        assert resp.json()["sources"] == []

    @pytest.mark.parametrize("body", [{}, {"session_id": "session_1"}])
    def test_missing_query_is_rejected_before_reaching_rag(self, api_client, fake_rag, body):
        resp = api_client.post("/api/query", json=body)

        assert resp.status_code == 422
        fake_rag.query.assert_not_called()

    def test_non_json_body_is_rejected(self, api_client, fake_rag):
        resp = api_client.post("/api/query", content=b"not json", headers={"Content-Type": "application/json"})

        assert resp.status_code == 422
        fake_rag.query.assert_not_called()

    def test_rag_failure_returns_500_with_error_detail(self, api_client, fake_rag):
        fake_rag.query.side_effect = RuntimeError("Anthropic API unavailable")

        resp = api_client.post("/api/query", json={"query": "q"})

        assert resp.status_code == 500
        assert resp.json() == {"detail": "Anthropic API unavailable"}


class TestCoursesEndpoint:
    def test_returns_course_count_and_titles(self, api_client, fake_rag, sample_course_analytics):
        fake_rag.get_course_analytics.return_value = sample_course_analytics

        resp = api_client.get("/api/courses")

        assert resp.status_code == 200
        assert resp.json() == {
            "total_courses": 2,
            "course_titles": sample_course_analytics["course_titles"],
        }

    def test_empty_catalog_returns_zero_and_empty_list(self, api_client):
        resp = api_client.get("/api/courses")

        assert resp.status_code == 200
        assert resp.json() == {"total_courses": 0, "course_titles": []}

    def test_extra_analytics_fields_are_not_exposed(self, api_client, fake_rag):
        fake_rag.get_course_analytics.return_value = {
            "total_courses": 1,
            "course_titles": ["Only Course"],
            "internal_debug": "should not leak",
        }

        resp = api_client.get("/api/courses")

        assert set(resp.json()) == {"total_courses", "course_titles"}

    def test_storage_failure_returns_500_with_error_detail(self, api_client, fake_rag):
        fake_rag.get_course_analytics.side_effect = RuntimeError("ChromaDB unreachable")

        resp = api_client.get("/api/courses")

        assert resp.status_code == 500
        assert resp.json() == {"detail": "ChromaDB unreachable"}


class TestDeleteSessionEndpoint:
    def test_deletes_existing_session(self, api_client, fake_rag):
        session_id = fake_rag.session_manager.create_session()

        resp = api_client.delete(f"/api/sessions/{session_id}")

        assert resp.status_code == 200
        assert resp.json() == {"deleted": True}
        assert session_id not in fake_rag.session_manager.sessions

    def test_unknown_session_is_reported_not_deleted(self, api_client):
        resp = api_client.delete("/api/sessions/session_does_not_exist")

        assert resp.status_code == 200
        assert resp.json() == {"deleted": False}


class TestStaticFrontend:
    def test_root_serves_index_html(self, api_client):
        resp = api_client.get("/")

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert "<title>Course Materials</title>" in resp.text

    def test_static_assets_are_served(self, api_client):
        resp = api_client.get("/app.js")

        assert resp.status_code == 200
        assert "test asset" in resp.text

    def test_static_responses_disable_caching(self, api_client):
        resp = api_client.get("/app.js")

        assert resp.headers["cache-control"] == "no-cache, no-store, must-revalidate"
        assert resp.headers["pragma"] == "no-cache"
        assert resp.headers["expires"] == "0"

    def test_missing_asset_returns_404(self, api_client):
        resp = api_client.get("/does-not-exist.js")

        assert resp.status_code == 404

    def test_api_routes_are_not_shadowed_by_static_mount(self, api_client, fake_rag, sample_course_analytics):
        fake_rag.get_course_analytics.return_value = sample_course_analytics

        resp = api_client.get("/api/courses")

        assert resp.status_code == 200
        assert resp.json()["total_courses"] == 2
