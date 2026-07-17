from __future__ import annotations

import os
from pathlib import Path

from agents import Agent, WebSearchTool

from .tools import (
    compute_segmentation_metrics,
    estimate_tensor_memory,
    inspect_notebook,
    list_project_files,
    read_project_file,
)


PROMPT_PATH = Path(__file__).with_name("prompt.md")


def build_agent() -> Agent:
    instructions = PROMPT_PATH.read_text(encoding="utf-8")
    model = os.getenv("AI4MARS_AGENT_MODEL", "gpt-5.6")
    return Agent(
        name="AI4Mars Senior ML Researcher",
        instructions=instructions,
        model=model,
        tools=[
            list_project_files,
            read_project_file,
            inspect_notebook,
            compute_segmentation_metrics,
            estimate_tensor_memory,
            WebSearchTool(),
        ],
    )


research_agent = build_agent()
