# AI4Mars Senior ML Researcher

This is a persistent OpenAI Agents SDK advisor grounded in the AI4Mars repository. It is intentionally one focused agent with read-only project tools.

## Capabilities

- Inspect repository files and notebook source/text outputs.
- Critique research direction, baselines, experimental controls, and claims.
- Diagnose training and evaluation behavior with ranked hypotheses.
- Explain mathematical and statistical concepts using project-specific examples.
- Compute metrics from a confusion matrix and estimate raw tensor memory.
- Search current literature when claims require external evidence.
- Resume prior conversations through a local SQLite session.

The agent cannot modify project files, execute training, or treat preliminary findings as established evidence.

## Run

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m research_agent
```

For a one-turn question:

```powershell
python -m research_agent --ask "Audit our current evidence for the big_rock failure mode."
```

Use a named conversation when you want a separate thread:

```powershell
python -m research_agent --session paper-planning
```

The default model is `gpt-5.6`. Override it with `AI4MARS_AGENT_MODEL` if needed. The API key is loaded from the repository's ignored `.env.local` file. Local conversation state is stored under the ignored `.research_agent/` directory.

ChatGPT subscriptions and OpenAI API billing are separate. If the API reports `insufficient_quota`, add API billing or adjust the project limit in the OpenAI Platform settings. For lower-cost experiments after billing is active, you can set `AI4MARS_AGENT_MODEL=gpt-5.4-mini`.

## Maintaining project context

Update `context/project_state.md` after a result becomes trustworthy or a research decision changes. Clearly label preliminary results and unresolved assumptions. The agent is instructed to read this file before consequential project advice.
