"""Stage 4: Multi-Agent System (In-Process).

Multiple specialist agents collaborate inside one Python process.
This uses LangGraph StateGraph plus Send for parallel specialist routing.
"""

import os
import sys
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from common.llm import get_llm


def _last_wins(current: str, new: str) -> str:
    """Reducer that keeps the latest non-empty value."""
    return new if new else current


class State(TypedDict):
    question: str
    law_analysis: Annotated[str, _last_wins]
    tax_analysis: Annotated[str, _last_wins]
    compliance_analysis: Annotated[str, _last_wins]
    privacy_analysis: Annotated[str, _last_wins]
    final_answer: Annotated[str, _last_wins]


TEST_QUESTIONS = [
    (
        "Our company collects customer personal data, may share it with a US partner, "
        "and wants to understand tax, compliance, and GDPR risks. What should we consider?"
    ),
    "A supplier breached a delivery contract. What legal issues should the injured party consider?",
]


def law_agent(state: State) -> dict:
    """Lead legal agent that produces the base legal analysis."""
    print("\n  [law_agent] Running general legal analysis...")
    llm = get_llm(temperature=0.3, max_tokens=450)
    prompt = (
        "You are a senior legal analyst. "
        "Review the question and identify the main legal issues in under 120 words.\n\n"
        f"Question: {state['question']}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"law_analysis": response.content}


def check_routing_node(state: State) -> dict:
    """No-op node used to keep the graph flow explicit before routing."""
    return {}


def check_routing(state: State) -> list[Send]:
    """Dispatch specialist agents in parallel when relevant."""
    question_lower = state["question"].lower()
    tasks: list[Send] = []

    if any(kw in question_lower for kw in ["tax", "irs", "thuế"]):
        tasks.append(Send("tax_agent", state))

    if any(kw in question_lower for kw in ["compliance", "sec", "regulation", "regulatory"]):
        tasks.append(Send("compliance_agent", state))

    if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu", "personal data"]):
        tasks.append(Send("privacy_agent", state))

    if tasks:
        print(
            "  [check_routing] Specialists selected: "
            + ", ".join(send.node for send in tasks)
        )
        return tasks

    print("  [check_routing] No specialist selected; routing directly to aggregate_results")
    return [Send("aggregate_results", state)]


def tax_agent(state: State) -> dict:
    """Agent chuyên về rủi ro thuế và giao dịch xuyên biên giới."""
    print("\n  [tax_agent] Running tax analysis...")
    llm = get_llm(temperature=0.3, max_tokens=350)
    prompt = (
        "You are a tax law specialist. "
        "Analyze only the tax risks in under 90 words. "
        "If the question has no meaningful tax issue, say so clearly.\n\n"
        f"Question: {state['question']}\n"
        f"General legal analysis: {state.get('law_analysis', 'N/A')}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"tax_analysis": response.content}


def compliance_agent(state: State) -> dict:
    """Agent chuyên về regulatory compliance."""
    print("\n  [compliance_agent] Running compliance analysis...")
    llm = get_llm(temperature=0.3, max_tokens=350)
    prompt = (
        "You are a regulatory compliance specialist. "
        "Analyze only compliance and regulatory issues in under 90 words. "
        "If no major compliance issue is present, say so clearly.\n\n"
        f"Question: {state['question']}\n"
        f"General legal analysis: {state.get('law_analysis', 'N/A')}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"compliance_analysis": response.content}


def privacy_agent(state: State) -> dict:
    """Agent chuyên về luật bảo vệ dữ liệu cá nhân."""
    print("\n  [privacy_agent] Running privacy analysis...")
    llm = get_llm(temperature=0.3, max_tokens=350)

    prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.

Câu hỏi gốc: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Hãy phân tích ngắn gọn các vấn đề về privacy, GDPR hoặc dữ liệu cá nhân nếu có.
Nếu câu hỏi không có yếu tố privacy/data/GDPR, hãy nói rõ là không phát hiện vấn đề privacy nổi bật.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}


def aggregate_results(state: State) -> dict:
    """Combine agent outputs into a short demo-friendly final answer."""
    print("\n  [aggregate_results] Building final answer...")
    parts = [
        f"Question: {state['question']}",
        f"General legal analysis: {state.get('law_analysis') or 'No general analysis generated.'}",
    ]

    if state.get("tax_analysis"):
        parts.append(f"Tax analysis: {state['tax_analysis']}")

    if state.get("compliance_analysis"):
        parts.append(f"Compliance analysis: {state['compliance_analysis']}")

    if state.get("privacy_analysis"):
        parts.append(f"Privacy analysis: {state['privacy_analysis']}")

    if not state.get("tax_analysis"):
        parts.append("Tax analysis: no standout tax issue detected.")
    if not state.get("compliance_analysis"):
        parts.append("Compliance analysis: no standout compliance issue detected.")
    if not state.get("privacy_analysis"):
        parts.append("Privacy analysis: no standout privacy issue detected.")

    final_answer = "\n\n".join(parts)
    return {"final_answer": final_answer}


def build_graph():
    """Create the in-process multi-agent graph."""
    graph = StateGraph(State)
    graph.add_node("law_agent", law_agent)
    graph.add_node("check_routing", check_routing_node)
    graph.add_node("tax_agent", tax_agent)
    graph.add_node("compliance_agent", compliance_agent)
    graph.add_node("privacy_agent", privacy_agent)
    graph.add_node("aggregate_results", aggregate_results)

    graph.add_edge(START, "law_agent")
    graph.add_edge("law_agent", "check_routing")
    graph.add_conditional_edges(
        "check_routing",
        check_routing,
        ["tax_agent", "compliance_agent", "privacy_agent", "aggregate_results"],
    )
    graph.add_edge("tax_agent", "aggregate_results")
    graph.add_edge("compliance_agent", "aggregate_results")
    graph.add_edge("privacy_agent", "aggregate_results")
    graph.add_edge("aggregate_results", END)
    return graph.compile()


def maybe_print_mermaid(graph) -> None:
    """Best-effort graph visualization for CLI."""
    try:
        mermaid = graph.get_graph().draw_mermaid()
        print("\n[Mermaid Graph]")
        print(mermaid)
    except Exception:
        print("\n[Mermaid Graph] Skipped in this environment.")


def main() -> None:
    print("=" * 70)
    print("STAGE 4: Multi-Agent System (In-Process)")
    print("=" * 70)
    print()
    print("[How it works]")
    print("  1. law_agent produces a general legal analysis")
    print("  2. check_routing decides which specialists to run")
    print("  3. Specialists are dispatched with Send in parallel")
    print("  4. aggregate_results combines outputs into the final answer")
    print()
    print("[Flow]")
    print("  START -> law_agent -> check_routing -> specialists -> aggregate_results -> END")

    graph = build_graph()
    maybe_print_mermaid(graph)

    questions_to_run = TEST_QUESTIONS[:1]

    for question in questions_to_run:
        print()
        print("=" * 70)
        print(f"Question:\n{question}")
        print("-" * 70)

        result = graph.invoke(
            {
                "question": question,
                "law_analysis": "",
                "tax_analysis": "",
                "compliance_analysis": "",
                "privacy_analysis": "",
                "final_answer": "",
            }
        )

        print("\n[Agent Outputs]")
        print(f"law_analysis:\n{result.get('law_analysis', '')}")
        if result.get("tax_analysis"):
            print(f"\ntax_analysis:\n{result['tax_analysis']}")
        if result.get("compliance_analysis"):
            print(f"\ncompliance_analysis:\n{result['compliance_analysis']}")
        if result.get("privacy_analysis"):
            print(f"\nprivacy_analysis:\n{result['privacy_analysis']}")

        print("\nFinal answer:")
        print(result.get("final_answer", ""))

    print()
    print("-" * 70)
    print("[Notes]")
    print("  - TEST_QUESTIONS includes 2 examples, but this script runs 1 by default to save quota.")
    print("  - Stage 5 remains separate and untouched.")
    print("=" * 70)


if __name__ == "__main__":
    load_dotenv()
    main()
