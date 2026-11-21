"""Tests for the ReAct loop with a scripted fake LLM (no Groq calls)."""
import json
from types import SimpleNamespace

from agent.agent import ResearchAgent
from agent.llm_client import LLMError


def tool_call(call_id, name, arguments):
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=args))


def assistant(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


class ScriptedLLM:
    """Returns pre-defined assistant messages in order and records what it was sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_agent(llm, tools, max_steps=5):
    schemas = [{"type": "function", "function": {"name": n, "parameters": {}}} for n in tools]
    return ResearchAgent(llm=llm, tools=tools, tool_schemas=schemas, max_steps=max_steps)


def test_multi_step_react_loop():
    stock = lambda ticker, period="1mo": {"success": True, "ticker": ticker, "current_price": 200.0}  # noqa: E731
    calc = lambda expression: {"success": True, "result": 210.0}  # noqa: E731
    llm = ScriptedLLM([
        assistant("I need Apple's price.", [tool_call("c1", "get_stock_data", {"ticker": "AAPL"})]),
        assistant("Now compute +5%.", [tool_call("c2", "calculate", {"expression": "200 * 1.05"})]),
        assistant("Apple trades at 200 USD; +5% would be 210 USD."),
    ])
    result = make_agent(llm, {"get_stock_data": stock, "calculate": calc}).run("Apple +5%?")

    assert result.stopped_reason == "final_answer"
    assert "210" in result.answer
    assert [s.tool for s in result.steps] == ["get_stock_data", "calculate"]
    # The 3rd LLM call must contain both tool observations, linked by tool_call_id.
    tool_msgs = [m for m in llm.calls[2]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]
    assert json.loads(tool_msgs[0]["content"])["current_price"] == 200.0


def test_tool_error_is_passed_back_to_llm():
    def broken(**_kwargs):
        raise RuntimeError("API down")

    llm = ScriptedLLM([
        assistant(None, [tool_call("c1", "search_news", {"query": "Apple"})]),
        assistant("News is currently unavailable."),
    ])
    result = make_agent(llm, {"search_news": broken}).run("Apple news?")

    assert result.stopped_reason == "final_answer"
    assert result.steps[0].result["success"] is False
    assert "API down" in llm.calls[1]["messages"][-1]["content"]


def test_unknown_tool_and_bad_json_do_not_crash():
    llm = ScriptedLLM([
        assistant(None, [tool_call("c1", "does_not_exist", {}), tool_call("c2", "calculate", "{not json")]),
        assistant("Done."),
    ])
    result = make_agent(llm, {"calculate": lambda expression: {"success": True}}).run("x")

    assert result.answer == "Done."
    assert "Unknown tool" in result.steps[0].result["error"]
    assert "Invalid tool arguments" in result.steps[1].result["error"]


def test_wrong_parameters_reported():
    llm = ScriptedLLM([
        assistant(None, [tool_call("c1", "calculate", {"wrong_param": 1})]),
        assistant("ok"),
    ])
    result = make_agent(llm, {"calculate": lambda expression: {"success": True}}).run("x")
    assert "Bad parameters" in result.steps[0].result["error"]


def test_max_steps_forces_final_answer_without_tools():
    loop_call = assistant(None, [tool_call("c", "calculate", {"expression": "1+1"})])
    llm = ScriptedLLM([loop_call, loop_call, assistant("Best effort answer.")])
    result = make_agent(llm, {"calculate": lambda expression: {"success": True, "result": 2}}, max_steps=2).run("x")

    assert result.stopped_reason == "max_steps"
    assert result.answer == "Best effort answer."
    assert llm.calls[-1]["tools"] is None


def test_llm_error_returns_graceful_answer():
    llm = ScriptedLLM([LLMError("Groq API timed out")])
    result = make_agent(llm, {}).run("x")
    assert result.stopped_reason == "llm_error"
    assert "timed out" in result.answer
