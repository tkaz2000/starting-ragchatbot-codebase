from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""
    
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "What to search for in the course content"
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)"
                    }
                },
                "required": ["query"]
            }
        }
    
    def execute(self, query: str, course_name: Optional[str] = None, lesson_number: Optional[int] = None) -> str:
        """
        Execute the search tool with given parameters.
        
        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter
            
        Returns:
            Formatted search results or error message
        """
        
        # Use the vector store's unified search interface
        results = self.store.search(
            query=query,
            course_name=course_name,
            lesson_number=lesson_number
        )
        
        # Handle errors
        if results.error:
            return results.error
        
        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number is not None:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."
        
        # Format and return results
        return self._format_results(results)
    
    def _resolve_url(self, course_title: str, lesson_num: Optional[int]) -> Optional[str]:
        """Return the lesson video link, falling back to the course link; None if neither is a valid http(s) URL"""
        url = None
        if lesson_num is not None:
            url = self.store.get_lesson_link(course_title, lesson_num)
        if not url:
            url = self.store.get_course_link(course_title)
        if url and url.startswith(("http://", "https://")):
            return url
        return None

    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        # Sources for the UI ({"label": str, "url": str | None}) accumulate across searches
        # within one query. Duplicates are dropped by (label, url).
        seen = {(s["label"], s["url"]) for s in self.last_sources}
        
        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get('course_title', 'unknown')
            lesson_num = meta.get('lesson_number')
            
            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"
            
            # Track source for the UI, with a link to the lesson (or course) video
            label = course_title
            if lesson_num is not None:
                label += f" - Lesson {lesson_num}"
            url = self._resolve_url(course_title, lesson_num)
            if (label, url) not in seen:
                seen.add((label, url))
                self.last_sources.append({"label": label, "url": url})
            
            formatted.append(f"{header}\n{doc}")
        
        return "\n\n".join(formatted)

class CourseOutlineTool(Tool):
    """Tool for returning a course's outline: title, link, and numbered lesson list"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": "Get the outline of a course: its title, course link, and every lesson's number and title",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')"
                    }
                },
                "required": ["course_name"]
            }
        }

    def execute(self, course_name: str) -> str:
        """
        Return the outline for the best-matching course.

        Args:
            course_name: Course title or partial title

        Returns:
            Formatted outline or a not-found message
        """
        outline = self.store.get_course_outline(course_name)
        if outline is None:
            return f"No course found matching '{course_name}'"

        lessons = outline["lessons"]
        lines = [
            f"Course: {outline['title']}",
            f"Course Link: {outline['course_link'] or 'Not available'}",
            f"Lessons ({len(lessons)}):"
        ]
        for lesson in lessons:
            lines.append(f"- Lesson {lesson['lesson_number']}: {lesson['lesson_title']}")
        return "\n".join(lines)

class ToolManager:
    """Manages available tools for the AI"""
    
    def __init__(self):
        self.tools = {}
    
    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    
    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"
        
        return self.tools[tool_name].execute(**kwargs)
    
    def has_tool(self, tool_name: str) -> bool:
        """Check whether a tool with this name is registered"""
        return tool_name in self.tools

    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources') and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, 'last_sources'):
                tool.last_sources = []