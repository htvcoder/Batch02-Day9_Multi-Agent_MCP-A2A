"""Tax Agent LangGraph definition."""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from common.llm import get_llm

TAX_SYSTEM_PROMPT = """You are a specialist tax attorney and CPA.

Reply concisely in short bullet points.
Focus on:
- tax exposure
- withholding tax or cross-border tax issues
- penalties
- reporting obligations
- company vs. executive liability

If relevant, mention IRS, DOJ Tax Division, or FinCEN.
Keep the answer under 140 words and end with a short educational-use disclaimer.
"""


def create_graph():
    """Return a compiled LangGraph create_react_agent for tax questions."""
    llm = get_llm(temperature=0.3, max_tokens=400)
    graph = create_react_agent(
        model=llm,
        tools=[],
        prompt=TAX_SYSTEM_PROMPT,
    )
    return graph
