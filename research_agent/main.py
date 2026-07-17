from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = REPO_ROOT / ".research_agent"


def _friendly_api_error(error: Exception) -> str:
    message = str(error)
    lowered = message.lower()
    if "insufficient_quota" in lowered or "current quota" in lowered:
        return (
            "The OpenAI API account has no available quota or has reached its spend "
            "limit. Check billing at https://platform.openai.com/settings/organization/billing "
            "and limits at https://platform.openai.com/settings/organization/limits. "
            "ChatGPT subscriptions and API billing are separate."
        )
    if "invalid_api_key" in lowered or "authentication" in lowered:
        return "The API key was rejected. Reconfigure OPENAI_API_KEY in .env.local."
    if "model_not_found" in lowered or "does not have access" in lowered:
        return (
            "This API project does not have access to the configured model. Set "
            "AI4MARS_AGENT_MODEL to a model available to the project."
        )
    if "timed out" in lowered or "timeout" in lowered:
        return (
            "The OpenAI API request timed out. Retry once; if it persists, check the "
            "local VPN, proxy, firewall, or network path to api.openai.com."
        )
    return f"The agent request failed: {message}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Talk with the AI4Mars senior ML research advisor."
    )
    parser.add_argument("--ask", help="Ask one question and exit.")
    parser.add_argument(
        "--session",
        default="main",
        help="Conversation name used for persistent local history (default: main).",
    )
    return parser.parse_args()


async def _run() -> None:
    load_dotenv(REPO_ROOT / ".env.local")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing from the environment or .env.local.")

    from agents import Runner, SQLiteSession

    from .agent import research_agent

    args = _parse_args()
    STATE_DIR.mkdir(exist_ok=True)
    session = SQLiteSession(
        session_id=f"ai4mars-{args.session}",
        db_path=STATE_DIR / "conversations.db",
    )

    if args.ask:
        try:
            result = await Runner.run(research_agent, args.ask, session=session)
            print(result.final_output)
        except Exception as error:
            raise SystemExit(_friendly_api_error(error)) from None
        return

    print("AI4Mars Senior ML Researcher")
    print(f"Session: {args.session}")
    print("Type /exit to leave. Your conversation will resume next time.\n")
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question.lower() in {"/exit", "/quit", "exit", "quit"}:
            return
        try:
            result = await Runner.run(research_agent, question, session=session)
            print(f"\nResearcher> {result.final_output}\n")
        except Exception as error:
            print(f"\n{_friendly_api_error(error)}\n")


def main() -> None:
    asyncio.run(_run())
