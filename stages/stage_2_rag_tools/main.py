"""Stage 2: LLM + RAG / Tools.

Adds retrieval-augmented generation and tool use to ground LLM responses
in external data. The orchestration is still manual: one tool-calling pass,
then one final LLM answer.
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from common.llm import get_llm

# ---------------------------------------------------------------------------
# Simulated legal knowledge base (in production, this would be a vector store)
# ---------------------------------------------------------------------------

LEGAL_KNOWLEDGE = [
    {
        "id": "ucc_breach",
        "keywords": ["breach", "contract", "remedies", "damages", "ucc"],
        "text": (
            "Under the Uniform Commercial Code (UCC) Article 2, remedies for breach of contract "
            "include: (1) expectation damages - placing the non-breaching party in the position "
            "they would have been in had the contract been performed; (2) consequential damages "
            "for foreseeable losses (Hadley v. Baxendale, 1854); (3) specific performance when "
            "the subject matter is unique; (4) cover damages - the cost of obtaining substitute "
            "performance. The statute of limitations is typically 4 years (UCC 2-725)."
        ),
    },
    {
        "id": "nda_trade_secret",
        "keywords": ["nda", "non-disclosure", "confidential", "trade secret", "agreement"],
        "text": (
            "NDA breaches may trigger both contractual and statutory liability. Under the Defend "
            "Trade Secrets Act (DTSA, 18 U.S.C. 1836), misappropriation of trade secrets can "
            "result in: (1) injunctive relief; (2) actual damages plus unjust enrichment; "
            "(3) exemplary damages up to 2x actual damages for willful misappropriation; "
            "(4) attorney's fees. State Uniform Trade Secrets Act (UTSA) versions provide "
            "additional remedies. Criminal prosecution is possible under the Economic Espionage "
            "Act (18 U.S.C. 1832) with penalties up to $5M for individuals."
        ),
    },
    {
        "id": "dtsa_details",
        "keywords": ["dtsa", "federal", "trade secret", "defend", "statute"],
        "text": (
            "The Defend Trade Secrets Act (2016) created a federal private cause of action for "
            "trade secret misappropriation. Key provisions: (1) ex parte seizure orders in "
            "extraordinary circumstances; (2) 3-year statute of limitations; (3) immunity for "
            "whistleblower disclosures to government officials; (4) employers must notify "
            "employees of whistleblower immunity in any NDA or employment agreement."
        ),
    },
    {
        "id": "liquidated_damages",
        "keywords": ["liquidated", "damages", "penalty", "clause", "contract", "nda"],
        "text": (
            "Liquidated damages clauses in NDAs are enforceable if: (1) actual damages would be "
            "difficult to calculate at the time of contracting; (2) the stipulated amount is a "
            "reasonable estimate of anticipated harm. Courts will void clauses that function as "
            "penalties (Restatement (Second) of Contracts 356). Typical NDA liquidated damages "
            "range from $10,000 to $500,000 depending on the nature of the confidential information."
        ),
    },
    {
        "id": "injunctive_relief",
        "keywords": ["injunction", "restraining", "order", "equitable", "nda", "breach"],
        "text": (
            "Courts routinely grant temporary restraining orders (TROs) and preliminary injunctions "
            "for NDA breaches because: (1) confidential information, once disclosed, cannot be "
            "'un-disclosed' - making monetary damages inadequate; (2) irreparable harm is presumed "
            "for trade secret misappropriation in many jurisdictions. The movant must show "
            "likelihood of success on the merits, irreparable harm, balance of equities, and "
            "public interest (Winter v. Natural Resources Defense Council, 2008)."
        ),
    },
    {
        "id": "labor_law",
        "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
        "text": (
            "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
            "đơn phương chấm dứt hợp đồng trong các trường hợp: (1) người lao động "
            "thường xuyên không hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị "
            "12 tháng chưa khỏi; (3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
        ),
    },
]


def normalize_text(text: str) -> str:
    """Normalize text for simple keyword matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def search_legal_knowledge(query: str) -> str:
    """Tra cứu knowledge base pháp lý theo từ khóa."""
    normalized_query = normalize_text(query)
    scored = []

    for entry in LEGAL_KNOWLEDGE:
        overlap = 0
        for keyword in entry["keywords"]:
            if normalize_text(keyword) in normalized_query:
                overlap += 1
        if overlap > 0:
            scored.append((overlap, entry))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_results = scored[:2]
    if not top_results:
        return "Khong tim thay thong tin phu hop trong knowledge base phap ly."

    return "\n\n".join(
        f"[{entry['id']}] {entry['text']}" for _, entry in top_results
    )


@tool
def calculate_damages(breach_type: str, contract_value: float) -> str:
    """Calculate estimated damages for a contract breach based on type and contract value."""
    breach_type_lower = breach_type.lower()
    if "willful" in breach_type_lower or "intentional" in breach_type_lower:
        multiplier = 2.0
        label = "Willful/intentional breach (2x multiplier under DTSA)"
    elif "negligent" in breach_type_lower:
        multiplier = 1.0
        label = "Negligent breach (1x actual damages)"
    else:
        multiplier = 1.5
        label = "Standard breach (1.5x estimated multiplier)"

    base_damages = contract_value * multiplier
    attorney_fees = contract_value * 0.15
    total = base_damages + attorney_fees

    return (
        f"Damage Estimate:\n"
        f"  Breach type: {label}\n"
        f"  Contract value: ${contract_value:,.2f}\n"
        f"  Estimated damages: ${base_damages:,.2f}\n"
        f"  Attorney's fees (~15%): ${attorney_fees:,.2f}\n"
        f"  Total estimated exposure: ${total:,.2f}"
    )


@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.

    Args:
        case_type: Loại vụ án (contract, tort, property)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định")


TOOLS = [search_legal_knowledge, calculate_damages, check_statute_of_limitations]
TEST_QUESTIONS = [
    "Tôi bị công ty sa thải trước thời hạn hợp đồng lao động thì cần lưu ý những vấn đề pháp lý nào?",
    "What is the statute of limitations for a contract dispute?",
]


def run_question(question: str) -> None:
    """Run one manual tool-calling pass for a single question."""
    print()
    print("=" * 70)
    print(f"Question:\n{question}")
    print("-" * 70)

    llm = get_llm(temperature=0.3, max_tokens=700)
    llm_with_tools = llm.bind_tools(TOOLS)
    tool_map = {tool_fn.name: tool_fn for tool_fn in TOOLS}

    messages = [
        SystemMessage(
            content=(
                "You are a legal expert for a Stage 2 demo. "
                "Use tools when they help. For labor, contract, NDA, or statute questions, "
                "prefer grounded answers using the available tools. "
                "If the user asks about statute of limitations, use check_statute_of_limitations. "
                "If the user asks about legal knowledge or labor issues, use search_legal_knowledge. "
                "Use calculate_damages only for monetary estimate questions. "
                "Keep the final answer concise, clear, and under 220 words."
            )
        ),
        HumanMessage(content=question),
    ]

    print("\n>>> Step 1: Asking LLM with tools bound...\n")
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
        print("No tool call. Direct answer:")
        print(response.content)
        return

    print(f">>> Step 2: Executing {len(response.tool_calls)} tool call(s)...\n")
    for tool_call in response.tool_calls:
        print(f"Tool called: {tool_call['name']}")
        print(f"Tool args: {tool_call['args']}")

        tool_fn = tool_map[tool_call["name"]]
        result = tool_fn.invoke(tool_call["args"])

        print("Tool result:")
        print(result)
        print()

        messages.append(
            ToolMessage(content=result, tool_call_id=tool_call["id"])
        )

    print(">>> Step 3: Asking the LLM for the final grounded answer...\n")
    messages.append(
        HumanMessage(
            content=(
                "Using the tool results above, provide the final answer now. "
                "Do not call any more tools. Keep it concise and practical."
            )
        )
    )
    final_response = llm.invoke(messages)
    print("Final answer:")
    print(final_response.content)


def main() -> None:
    print("=" * 70)
    print("STAGE 2: LLM + RAG / Tools")
    print("=" * 70)
    print()
    print("[How it works]")
    print("  1. We create the LLM with get_llm()")
    print("  2. We bind tools with llm.bind_tools(tools)")
    print("  3. The LLM may request tool calls")
    print("  4. We manually execute tools once")
    print("  5. We send tool results back and get the final answer")
    print()
    print("Tools available:")
    print("  - search_legal_knowledge")
    print("  - calculate_damages")
    print("  - check_statute_of_limitations")

    for question in TEST_QUESTIONS:
        run_question(question)

    print()
    print("-" * 70)
    print("[Improvements over Stage 1]")
    print("  + Grounded: answers can use retrieved legal knowledge")
    print("  + Tool use: can search a knowledge base or check limitation periods")
    print("  + More accurate: retrieval reduces hallucination risk")
    print()
    print("[Limitations of Stage 2]")
    print("  - Manual orchestration: we wrote the tool-call loop ourselves")
    print("  - Single pass: only one round of tool calls")
    print("  - No reasoning loop: the LLM cannot decide to search again if needed")
    print()
    print("Next: Stage 3 wraps this in an autonomous ReAct agent loop.")
    print("=" * 70)


if __name__ == "__main__":
    load_dotenv()
    main()
