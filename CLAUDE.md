# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Dependencies are managed with `uv` (Python 3.13, see `.python-version` and `pyproject.toml`).

```bash
uv sync                                  # install dependencies
./run.sh                                 # start the app (Git Bash on Windows)
cd backend && uv run uvicorn app:app --reload --port 8000   # equivalent manual start
```

The app serves the UI at http://localhost:8000 and the API docs at http://localhost:8000/docs.

The server **must be started from `backend/`**. Paths are relative: `app.py` loads course files from `../docs` and mounts `../frontend`, and `config.py` writes ChromaDB to `./chroma_db`, which resolves to `backend/chroma_db/`.

There is no test suite, linter, or formatter configured in this repository. Don't assume `pytest` or `ruff` exist; check before running them.

Environment: `ANTHROPIC_API_KEY` is read from a `.env` file. The root `.env` works because `load_dotenv()` searches parent directories. Copy `.env.example` as a starting point.

## Architecture

### Startup ingestion
On startup, `app.py` calls `RAGSystem.add_course_folder("../docs", clear_existing=False)`. Each `.pdf`, `.docx`, or `.txt` file is parsed and chunked, and then added to ChromaDB. A course is skipped if its title is already in the `course_catalog` collection, so re-running the server does not duplicate data. Editing an existing course file has no effect until the data is cleared (`clear_existing=True` wipes both collections).

**Only `.txt` files are actually parsed.** `DocumentProcessor.read_file()` reads every file as UTF-8 text, so a PDF or DOCX in `docs/` will be ingested as garbage. Adding real PDF or DOCX support means changing `document_processor.py`.

### Document format contract
`DocumentProcessor.process_course_document()` depends on a fixed layout: `Course Title:`, `Course Link:`, and `Course Instructor:` on the first lines, then `Lesson N: <title>` headers, with an optional `Lesson Link:` line directly after each header. Lesson text is everything in between. Files that don't match still load, but the title falls back to the first line or the filename.

Chunks are sentence-based, up to 800 characters, with about 100 characters of overlap. The first chunk of each lesson is prefixed `Lesson N content:`. The last lesson in a file uses a different prefix (`Course <title> Lesson N content:`). Keep this in mind when changing chunk text, because the prefix becomes part of the embedded text.

### Two stores, one embedding model
`VectorStore` keeps two ChromaDB collections, both embedded with `all-MiniLM-L6-v2`:
- `course_catalog`: one record per course, keyed by title. Its metadata carries the instructor, link, and a `lessons_json` string. Course-name filters are resolved here by semantic match, so partial names work.
- `course_content`: one record per chunk, with `course_title`, `lesson_number`, and `chunk_index` metadata. Searches use a `where` filter built from the resolved course title and lesson number.

### Query path (`RAGSystem.query`)
1. `app.py` `POST /api/query` creates a session if the request has none.
2. `RAGSystem.query` fetches history from `SessionManager`, wraps the question in a prompt, and calls `AIGenerator.generate_response` with the `search_course_content` tool.
3. Claude decides whether to search. If it does, `_handle_tool_execution` runs the tool and makes a **second** Claude call with tools removed, so the model must answer from the results. Each query gets at most one tool round trip.
4. `RAGSystem` pulls the source labels from the tool, clears them, and saves the raw question and answer to the session.

Tools are registered through `ToolManager`. To add a tool, implement `Tool` (`get_tool_definition` and `execute`) and register it in `RAGSystem.__init__`. `AIGenerator._handle_tool_execution` runs any `tool_use` block it receives, but its second call is never given tools, so it cannot chain a second search.

### State and its limits
- Sessions live in memory in `SessionManager` and are lost on restart. The browser keeps its `session_id`, so after a restart the server silently starts an empty session under that ID.
- Sources are stored on the single `CourseSearchTool` instance (`last_sources`), which is shared across all requests. Concurrent queries can read or clear each other's sources. Returning sources per call would fix this.
- History is trimmed to `MAX_HISTORY * 2` messages. It is formatted as plain text and appended to the system prompt, not sent as message turns.

### Frontend
`frontend/script.js` uses relative `/api` URLs. It sends `{query, session_id}` to `POST /api/query` and renders the Markdown answer with `marked`. It also shows the `sources` array in a collapsible section. Course stats come from `GET /api/courses`. The static files are served by FastAPI, so there is no separate frontend build step.

### Configuration
All tunable values are in `backend/config.py` (`Config` dataclass): model name, embedding model, chunk size and overlap, `MAX_RESULTS`, `MAX_HISTORY`, and the ChromaDB path. The Claude model is currently `claude-sonnet-4-20250514`.

### Per instructor - running python
#use uv to run Python files.