"""ReAct agent: the LLM alternates between reasoning and calling tools until it can answer."""
import json
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Tuple

import config
from agent.llm_client import GroqLLMClient, LLMError
from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

logger = logging.getLogger(__name__)

MAX_TOOL_RESULT_CHARS = 6000  # keep tool outputs from blowing up the context window

SYSTEM_PROMPT = """You are an investment research assistant for a professional asset manager.
Today's date is {today}.

You answer questions by reasoning step by step and calling tools to gather facts:
- get_stock_data: live prices, valuation ratios and performance for a ticker
- search_news: recent news headlines about companies, sectors or the macro environment
- search_documents: the internal document library (annual reports, prospectuses)
- calculate: any arithmetic - never do math in your head

Rules:
1. Before each tool call, briefly state what you need and why.
2. Use as many tool calls as necessary, but do not repeat identical calls.
3. If a tool returns an error, adapt (different ticker/query or another tool) or explain the limitation.
4. Base your final answer only on tool results. Cite sources (ticker data, article source, document + page).
5. Be concise and factual. This is research support, not investment advice."""


@dataclass
class AgentStep:
    step: int
    thought: Optional[str]
    tool: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]


@dataclass
class AgentResult:
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    stopped_reason: str = "final_answer"  # final_answer | max_steps | llm_error


class ResearchAgent:
    def __init__(self, llm: Optional[GroqLLMClient] = None,
                 tools: Optional[Dict[str, Callable[..., Dict[str, Any]]]] = None,
                 tool_schemas: Optional[List[Dict[str, Any]]] = None,
                 max_steps: Optional[int] = None):
        self.llm = llm or GroqLLMClient()
        self.tools = tools if tools is not None else TOOL_FUNCTIONS
        self.tool_schemas = tool_schemas if tool_schemas is not None else TOOL_SCHEMAS
        self.max_steps = max_steps or config.AGENT_MAX_STEPS

    # ---- tool execution -------------------------------------------------
    def _execute_tool(self, name: str, raw_arguments: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run one tool call. Never raises: every failure becomes an error dict the LLM can read."""
        try:
            arguments = json.loads(raw_arguments or "{}")
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            return {}, {"success": False, "error": f"Invalid tool arguments ({exc}): {raw_arguments!r}"}

        func = self.tools.get(name)
        if func is None:
            return arguments, {"success": False, "error": f"Unknown tool '{name}'. Available: {list(self.tools)}"}
        try:
            result = func(**arguments)
        except TypeError as exc:  # wrong / missing parameters
            result = {"success": False, "error": f"Bad parameters for {name}: {exc}"}
        except Exception as exc:  # tools should catch their own errors; this is the safety net
            logger.exception("Tool %s crashed", name)
            result = {"success": False, "error": f"Tool {name} failed unexpectedly: {exc}"}
        if not isinstance(result, dict):
            result = {"success": True, "result": result}
        return arguments, result

    @staticmethod
    def _serialize(result: Dict[str, Any]) -> str:
        text = json.dumps(result, default=str, ensure_ascii=False)
        if len(text) > MAX_TOOL_RESULT_CHARS:
            text = text[:MAX_TOOL_RESULT_CHARS] + "... [truncated]"
        return text

    # ---- ReAct loop -----------------------------------------------------
    def run(self, question: str) -> AgentResult:
        logger.info("=" * 70)
        logger.info("QUESTION: %s", question)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(today=date.today().isoformat())},
            {"role": "user", "content": question},
        ]
        steps: List[AgentStep] = []

        for iteration in range(1, self.max_steps + 1):
            logger.info("-" * 70)
            logger.info("ITERATION %d/%d - asking LLM what to do next", iteration, self.max_steps)
            try:
                message = self.llm.chat(messages, tools=self.tool_schemas)
            except LLMError as exc:
                logger.error("LLM call failed: %s", exc)
                return AgentResult(answer=f"Sorry, the language model call failed: {exc}",
                                   steps=steps, stopped_reason="llm_error")

            tool_calls = getattr(message, "tool_calls", None) or []
            thought = (message.content or "").strip() or None

            # No tool requested -> the model has produced its final answer.
            if not tool_calls:
                logger.info("FINAL ANSWER after %d tool call(s)", len(steps))
                return AgentResult(answer=thought or "(The model returned an empty answer.)", steps=steps)

            if thought:
                logger.info("THOUGHT: %s", thought)

            # Echo the assistant's tool-call message back so the model sees its own actions.
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                name = tc.function.name
                logger.info("ACTION: %s(%s)", name, tc.function.arguments)
                arguments, result = self._execute_tool(name, tc.function.arguments)
                status = "OK" if result.get("success", True) else f"ERROR - {result.get('error')}"
                logger.info("OBSERVATION: %s -> %s", name, status)
                logger.debug("OBSERVATION DATA: %s", self._serialize(result))
                steps.append(AgentStep(len(steps) + 1, thought, name, arguments, result))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": self._serialize(result),
                })

        # Step budget exhausted: ask for a best-effort answer without tools.
        logger.warning("Reached max steps (%d) - requesting final answer without tools", self.max_steps)
        messages.append({"role": "user", "content": "You have reached the tool-call limit. "
                                                    "Give your best final answer now using the information gathered."})
        try:
            message = self.llm.chat(messages, tools=None)
            answer = (message.content or "").strip() or "(No answer produced.)"
        except LLMError as exc:
            answer = f"Stopped after {self.max_steps} steps and the final LLM call failed: {exc}"
        return AgentResult(answer=answer, steps=steps, stopped_reason="max_steps")
