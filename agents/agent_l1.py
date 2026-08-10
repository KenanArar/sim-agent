"""
Level 1 — Simulation Subagent.

Runs LTspice simulations and parses results. This is a self-contained
LangGraph subgraph invoked by the root agent.

Tools:
  - list_circuits
  - run_ltspice_simulation
  - read_netlist
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from state import AgentState
from config import CIRCUITS_DIR, LLM_MODEL, OPENAI_API_KEY
from tools.ltspice_runner import run_simulation
from parsers.log_parser import parse_log
from parsers.raw_parser import parse_raw


# ═══════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════

@tool
def list_circuits() -> str:
    """List all available circuit netlist files (.cir) in the circuits directory."""
    circuits = list(CIRCUITS_DIR.glob("*.cir"))
    if not circuits:
        return "No circuit files found in the circuits/ directory."
    lines = [f"Available circuits ({len(circuits)}):"]
    for c in sorted(circuits):
        lines.append(f"  - {c.name}")
    return "\n".join(lines)


@tool
def run_ltspice_simulation(netlist_filename: str) -> str:
    """
    Run an LTspice simulation on a netlist file.

    Args:
        netlist_filename: Name of the .cir file (e.g. 'rc_lowpass.cir').
                          Must exist in the circuits/ directory.

    Returns:
        Simulation status and summary of results.
    """
    netlist_path = CIRCUITS_DIR / netlist_filename

    result = run_simulation(netlist_path)

    if not result.success:
        return f"Simulation FAILED: {result.error_message}"

    parts = [
        f"Simulation of {netlist_filename} completed in {result.elapsed_seconds:.2f}s.",
    ]

    log_data = None

    if result.log_path and result.log_path.exists():
        log_data = parse_log(result.log_path)
        if log_data.has_errors:
            parts.append("\n⚠ ERRORS found in log:")
            for e in log_data.errors:
                parts.append(f"  - {e}")
        if log_data.warnings:
            parts.append(f"\nWarnings: {len(log_data.warnings)}")
            for w in log_data.warnings:
                parts.append(f"  - {w}")
        if log_data.operating_point:
            parts.append("\nOperating Point:")
            for name, val in log_data.operating_point.items():
                parts.append(f"  {name} = {val:.6g}")
        if log_data.analysis_types:
            parts.append(f"\nAnalysis types: {', '.join(log_data.analysis_types)}")

    # The ltspice package cannot parse DC .op raw files (raises ValueError).
    # That's fine — the log already has all operating point values in that case.
    if result.raw_path and result.raw_path.exists():
        try:
            waveform = parse_raw(result.raw_path)
            parts.append(f"\nWaveform data ({waveform.analysis_type}):")
            parts.append(f"  Variables: {', '.join(waveform.variables)}")
            parts.append(f"  Data points: {waveform.point_count}")
            if waveform.measurements:
                parts.append("\nKey measurements:")
                for name, val in waveform.measurements.items():
                    parts.append(f"  {name}: {val}")
        except ValueError:
            pass  # DC .op raw file — log has all the data we need
        except Exception as e:
            parts.append(f"\nCould not parse .raw file: {e}")

    return "\n".join(parts)


@tool
def read_netlist(netlist_filename: str) -> str:
    """
    Read and return the contents of a netlist file.

    Args:
        netlist_filename: Name of the .cir file to read.

    Returns:
        The full text of the netlist file.
    """
    netlist_path = CIRCUITS_DIR / netlist_filename
    if not netlist_path.exists():
        return f"File not found: {netlist_filename}"
    return netlist_path.read_text()


# ═══════════════════════════════════════════════════════════════════
# GRAPH
# ═══════════════════════════════════════════════════════════════════

TOOLS = [list_circuits, run_ltspice_simulation, read_netlist]

SYSTEM_PROMPT = """You are a circuit simulation assistant. You run LTspice simulations
and interpret the results.

Your capabilities:
1. List available circuit files
2. Run LTspice simulations and report results
3. Read netlist files to explain circuit topology

When running a simulation:
- Use run_ltspice_simulation with the exact filename
- Interpret the results in plain language
- For RC filters: explain cutoff frequency in terms of R and C values
- For voltage dividers: explain how the resistor ratio determines output voltage

Keep responses concise. You're talking to an engineering student who understands
circuit fundamentals.
"""


def _agent_node(state: AgentState) -> dict:
    llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY).bind_tools(TOOLS)
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm.invoke(messages)
    return {"messages": [response]}


def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_simulation_graph():
    """Build and compile the Level 1 simulation subgraph."""
    tool_node = ToolNode(TOOLS)
    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()
