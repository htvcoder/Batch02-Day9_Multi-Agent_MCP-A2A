import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from common.llm import get_llm


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def search_case_law(keywords: str) -> str:
    """Find a relevant case by keyword.

    Args:
        keywords: Search keywords
    """
    cases = {
        "breach": "Hadley v. Baxendale (1854) - Consequential damages",
        "negligence": "Donoghue v. Stevenson (1932) - Duty of care",
        "contract": "Carlill v. Carbolic Smoke Ball Co (1893) - Unilateral contract",
    }
    for key, case in cases.items():
        if key in keywords.lower():
            return case
    return "Không tìm thấy án lệ phù hợp"


TOOLS = [search_case_law]

TEST_QUESTIONS = [
    (
        "A supplier breached a delivery contract and caused business losses. "
        "What case law is relevant and what should the injured party consider?"
    ),
]

SYSTEM_PROMPT = (
    "You are a legal ReAct demo agent. "
    "Always call search_case_law once when the user asks for relevant case law. "
    "After the tool result, give the final answer immediately. "
    "Then give a concise final answer under 120 words."
)


def format_message_content(content: object) -> str:
    """Convert message content into printable text."""
    if isinstance(content, str):
        return content
    return str(content)


def print_trace(messages: list[BaseMessage]) -> None:
    """Print the agent message trace for debugging."""
    print("\n[Agent Trace]")
    for index, message in enumerate(messages, start=1):
        message_type = message.__class__.__name__
        print(f"{index}. {message_type}")

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            for tool_call in tool_calls:
                print(f"   Tool call: {tool_call['name']}({tool_call['args']})")

        content = format_message_content(getattr(message, "content", ""))
        if content:
            print(f"   Content: {content}")


def main() -> None:
    print("=" * 70)
    print("STAGE 3: Single Agent (ReAct Loop)")
    print("=" * 70)
    print()
    print("[How it works]")
    print("  1. We create a ReAct agent with create_react_agent()")
    print("  2. The agent decides which tools to call")
    print("  3. We call agent_executor.invoke(...) once per question")
    print("  4. The agent handles tool use and final synthesis itself")
    print()
    print("Tools available:")
    print("  - search_case_law")
    print()
    print("Debug note:")
    print("  - verbose=True is not supported by the installed create_react_agent API")
    print("  - We print result['messages'] to show reasoning steps and tool usage")

    llm = get_llm(temperature=0.3, max_tokens=220)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="create_react_agent has been moved to `langchain.agents`.*",
        )
        agent_executor = create_react_agent(
            model=llm,
            tools=TOOLS,
            prompt=SYSTEM_PROMPT,
        )

    for question in TEST_QUESTIONS:
        print()
        print("=" * 70)
        print(f"Question:\n{question}")
        print("-" * 70)

        result = agent_executor.invoke(
            {
                "messages": [
                    HumanMessage(content=question),
                ]
            },
            config={"recursion_limit": 10},
        )

        messages = result["messages"]
        print_trace(messages)

        final_answer = ""
        for message in reversed(messages):
            if message.__class__.__name__ == "AIMessage" and format_message_content(message.content):
                if not getattr(message, "tool_calls", None):
                    final_answer = format_message_content(message.content)
                    break

        print("\nFinal answer:")
        print(final_answer if final_answer else "(No final answer content returned)")

    print()
    print("-" * 70)
    print("[Improvements over Stage 2]")
    print("  + ReAct agent: no manual tool-call loop in user code")
    print("  + Autonomous: the agent decides whether to call one or more tools")
    print("  + Cleaner API: one agent_executor.invoke(...) call per question")
    print()
    print("[Limitations of Stage 3]")
    print("  - Single agent: one LLM still handles every legal domain")
    print("  - Tool calls remain sequential inside the agent")
    print("  - No multi-agent specialization yet")
    print()
    print("Next: Stage 4 introduces multiple specialized agents.")
    print("=" * 70)


if __name__ == "__main__":
    load_dotenv()
    main()
