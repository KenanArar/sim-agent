"""
Configuration for the simulation agent.
Paths, timeouts, and environment settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── LTspice Configuration ──────────────────────────────────────────
# Mac default install path. Update if yours is different.
LTSPICE_BINARY = os.getenv(
    "LTSPICE_BINARY",
    "/Applications/LTspice.app/Contents/MacOS/LTspice"
)

# Maximum time (seconds) to wait for a simulation to finish
SIMULATION_TIMEOUT = int(os.getenv("SIMULATION_TIMEOUT", "120"))

# ── Project Paths ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
CIRCUITS_DIR = PROJECT_ROOT / "circuits"

# ── LLM Configuration ──────────────────────────────────────────────
# Using OpenAI-compatible API for LangGraph tool calling.
# Swap model string to use Gemini, Anthropic, etc.
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
