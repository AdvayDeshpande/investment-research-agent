"""CLI entry point.

Usage:
    python main.py "What is the P/E ratio of Apple?"   # single question
    python main.py                                      # interactive chat
    python main.py -v "..."                             # also print raw tool outputs
"""
import argparse
import logging
import sys

from agent import LLMError, ResearchAgent


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # Silence noisy third-party loggers so the agent's reasoning stays readable.
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers", "yfinance", "groq", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def print_answer(result) -> None:
    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result.answer)
    tools_used = ", ".join(s.tool for s in result.steps) or "none"
    print(f"\n[tools used: {tools_used} | stop reason: {result.stopped_reason}]\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Investment Research Agent (ReAct + Groq)")
    parser.add_argument("question", nargs="*", help="Question to ask. Omit for interactive mode.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show raw tool outputs (DEBUG logging)")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum ReAct iterations")
    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        agent = ResearchAgent(max_steps=args.max_steps)
    except LLMError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    if args.question:
        print_answer(agent.run(" ".join(args.question)))
        return 0

    print("Investment Research Agent - type a question, or 'exit' to quit.")
    while True:
        try:
            question = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question:
            print_answer(agent.run(question))
    return 0


if __name__ == "__main__":
    sys.exit(main())
