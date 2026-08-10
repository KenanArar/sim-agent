"""
LangGraph agent for the LTspice simulation runner.

Graph structure (Level 1):

    User Input → agent (LLM) → should_continue?
                    ↑               ↓ (tool call)
                    └── tool_node ←─┘
                                    ↓ (no tool call)
                                  END

The agent can call tools to:
  1. List available circuit files
  2. Run a simulation
  3. Parse simulation results
"""

import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from state import AgentState
from config import CIRCUITS_DIR, LLM_MODEL, OPENAI_API_KEY
from tools.ltspice_runner import run_simulation
from parsers.log_parser import parse_log
from parsers.raw_parser import parse_raw


# ═══════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS
# These are the capabilities the LLM can invoke.
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

    # Run the simulation
    result = run_simulation(netlist_path)

    if not result.success:
        return f"Simulation FAILED: {result.error_message}"

    # Build response from parsed outputs
    parts = [
        f"Simulation of {netlist_filename} completed in {result.elapsed_seconds:.2f}s.",
    ]

    log_data = None

    # Parse log file
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

    # Parse raw file for waveform data.
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
# GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════

TOOLS = [list_circuits, run_ltspice_simulation, read_netlist]

# Initialize the LLM with tools bound
llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=OPENAI_API_KEY,
).bind_tools(TOOLS)

SYSTEM_PROMPT = """You are a circuit simulation assistant. You help users run and analyze
LTspice circuit simulations.

Your capabilities:
1. List available circuit files in the project
2. Run LTspice simulations and report results
3. Read netlist files to explain circuit topology

When a user asks you to run a simulation:
- Use the run_ltspice_simulation tool with the filename
- Interpret the results in plain language
- Explain what the measurements mean (e.g. what a -3dB cutoff frequency tells us)

When explaining results, relate them to the circuit:
- For RC filters: explain cutoff frequency in terms of R and C values
- For voltage dividers: explain how the ratio determines output voltage
- For any circuit: connect the numbers back to the physics

Keep responses concise but informative. You're talking to an engineering student
who understands circuit fundamentals.
"""


def agent_node(state: AgentState) -> dict:
    """LLM agent node — reads messages, decides whether to call tools or respond."""
    from langchain_core.messages import SystemMessage

    messages = state["messages"]
    # Prepend system prompt if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = llm.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    Routing function: check if the last message has tool calls.
    If yes → route to tool_node. If no → END.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def build_graph() -> StateGraph:
    """
    Build and compile the Level 1 simulation agent graph.

    Graph:
        agent → (tool calls?) → tool_node → agent → ... → END
    """
    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Set entry point
    graph.set_entry_point("agent")

    # Add conditional edge from agent
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        END: END,
    })

    # Tools always route back to agent
    graph.add_edge("tools", "agent")

    return graph.compile()
