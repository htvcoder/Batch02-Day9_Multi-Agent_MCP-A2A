"""Stage 1: Direct LLM Calling.

The simplest way to use an LLM: send messages, get a response.
No tools, no memory, no agents. Just a direct API call.
"""

import os
import sys

# Allow running directly: python stages/stage_1_direct_llm/main.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from common.llm import get_llm

QUESTION = (
    "Mot cong ty don phuong cham dut hop dong dich vu truoc thoi han "
    "thi co the phat sinh nhung rui ro phap ly nao?"
)


def main():
    print("=" * 70)
    print("STAGE 1: Direct LLM Calling")
    print("=" * 70)
    print()
    print("[How it works]")
    print("  1. We get the LLM with get_llm()")
    print("  2. We send a SystemMessage + HumanMessage directly to the LLM")
    print("  3. The LLM responds from its training data only")
    print("  4. No tools, no retrieval, no external knowledge")
    print()
    print(f"Question:\n{QUESTION}")
    print("-" * 70)

    llm = get_llm(max_tokens=1024)

    messages = [
        SystemMessage(
            content=(
                "Ban la mot chuyen gia phap ly. Hay tra loi ro rang, de hieu, "
                "co cau truc ngan gon, neu ra cac rui ro phap ly chinh va luu y "
                "rang day chi la phan tich thong tin chung, khong phai tu van phap ly cu the."
            )
        ),
        HumanMessage(content=QUESTION),
    ]

    print("\n>>> Calling LLM directly with llm.invoke(messages)...\n")
    response = llm.invoke(messages)
    print("Answer:")
    print(response.content)

    print()
    print("-" * 70)
    print("[Limitations of Stage 1]")
    print("  - Stateless: no conversation memory between calls")
    print("  - No tools: cannot search databases or calculate damages")
    print("  - Knowledge cutoff: only knows what was in training data")
    print("  - No grounding: cannot cite specific statutes or current case law")
    print()
    print("Next: Stage 2 adds RAG and tools to ground responses in real data.")
    print("=" * 70)


if __name__ == "__main__":
    load_dotenv()
    main()
