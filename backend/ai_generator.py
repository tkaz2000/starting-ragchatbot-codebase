import anthropic
from typing import List, Optional, Dict, Any, Tuple

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Maximum number of sequential tool-call rounds per user query
    MAX_TOOL_ROUNDS = 2

    # Returned when the model produces no text at all
    FALLBACK_TEXT = "I wasn't able to produce an answer. Please try rephrasing your question."

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Tools:
- **search_course_content**: Use **only** for questions about specific course content or detailed educational materials
- **get_course_outline**: Use for questions about a course's outline, structure, or list of lessons. Pass the course name only
- **Up to two rounds of tool calls per query**: after a tool result, call a tool again only if that result is not enough to answer. For example, get a course outline to find a lesson title, then search for that title. Each later call should use what the earlier results showed
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Course Outline Responses:
- When answering an outline question, include the course title, the course link, and the number and title of every lesson, in lesson order
- Use the outline data exactly as returned by the tool

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **Course outline questions**: Call get_course_outline, then answer
- **Multi-part or comparison questions**: Gather what each part needs with tool calls, then answer everything together
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 1000
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Claude may call tools in up to MAX_TOOL_ROUNDS sequential rounds. Each round is a
        separate API request, so Claude can use earlier results to decide the next call.
        The loop ends when Claude replies without tool calls, when a tool call fails, or
        when the round cap is reached. Then one final request is made with no tools, so
        Claude must answer from the results it already has.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]
        response = self._create(system_content, messages, tools)

        rounds_used = 0
        while True:
            if not tool_manager or not self._tool_uses(response):
                return self._text_of(response)

            rounds_used += 1
            messages.append({"role": "assistant", "content": response.content})
            tool_results, failed = self._run_tools(response, tool_manager)
            messages.append({"role": "user", "content": tool_results})

            # Stop after a failed tool call or after the last allowed round
            if failed or rounds_used >= self.MAX_TOOL_ROUNDS:
                break
            response = self._create(system_content, messages, tools)

        # Final answer with no tools offered
        final_response = self._create(system_content, messages, tools=None)
        return self._text_of(final_response)

    def _create(self, system_content: str, messages: List[Dict[str, Any]], tools: Optional[List]):
        """Send one request to Claude. Tools are offered only when provided."""
        params = {
            **self.base_params,
            "system": system_content,
            "messages": messages
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = {"type": "auto"}
        return self.client.messages.create(**params)

    def _tool_uses(self, response) -> List[Any]:
        """Return the tool_use blocks in a response. The blocks are the source of truth, not stop_reason."""
        return [block for block in response.content if block.type == "tool_use"]

    def _run_tools(self, response, tool_manager) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Execute every tool_use block in a response.

        Every block gets a tool_result with its id, even when another block failed.
        A failure is an exception or an unknown tool name. Results that merely say
        nothing was found are not failures.

        Returns:
            (tool_result blocks, whether any call failed)
        """
        tool_results = []
        failed = False
        for block in self._tool_uses(response):
            is_error = False
            if not tool_manager.has_tool(block.name):
                output = f"Tool '{block.name}' not found"
                is_error = True
            else:
                try:
                    output = tool_manager.execute_tool(block.name, **block.input)
                except Exception as e:
                    # Report the failure to Claude instead of failing the whole request
                    output = f"Tool error: {e}"
                    is_error = True

            if not output:
                output = "The tool returned no results."

            result = {"type": "tool_result", "tool_use_id": block.id, "content": output}
            if is_error:
                result["is_error"] = True
                failed = True
            tool_results.append(result)
        return tool_results, failed

    def _text_of(self, response) -> str:
        """Join the text blocks of a response, or return a fallback if there are none."""
        text = "".join(block.text for block in response.content if block.type == "text")
        return text if text.strip() else self.FALLBACK_TEXT
