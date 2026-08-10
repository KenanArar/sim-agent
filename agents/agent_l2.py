"""
Level 2 — Circuit Generator Subagent.

Builds new circuits from natural language specs, saves them as netlists,
then verifies them by running a Level 1 simulation.

The LLM extracts structured parameters from the user's request and calls
generate_netlist with them. All math is done in generators/netlist_builder.py.

Tools:
  - generate_netlist
  - run_ltspice_simulation  (reused from agent_l1)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from state import AgentState
from config import LLM_MODEL, OPENAI_API_KEY
from agents.agent_l1 import run_ltspice_simulation
from generators import netlist_builder


# ═══════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════

@tool
def generate_netlist(circuit_type: str, params: str, filename: str) -> str:
    """
    Generate a SPICE netlist for the specified circuit type and save it.

    Args:
        circuit_type: One of: voltage_divider, rc_lowpass, rc_highpass,
                      rl_lowpass, rl_highpass, rlc_bandpass, rlc_bandstop
        params: JSON string of numeric parameters for the circuit type.
                voltage_divider:  {"vin": <V>, "vout": <V>}
                rc_lowpass:       {"f_c": <Hz>}
                rc_highpass:      {"f_c": <Hz>}
                rl_lowpass:       {"f_c": <Hz>}
                rl_highpass:      {"f_c": <Hz>}
                rlc_bandpass:     {"f_0": <Hz>, "Q": <float>}
                rlc_bandstop:     {"f_0": <Hz>, "Q": <float>}
        filename: Name for the saved .cir file (e.g. 'my_filter.cir').

    Returns:
        The filename that was saved, or an error message.
    """
    builders = {
        "voltage_divider": netlist_builder.voltage_divider,
        "rc_lowpass":      netlist_builder.rc_lowpass,
        "rc_highpass":     netlist_builder.rc_highpass,
        "rl_lowpass":      netlist_builder.rl_lowpass,
        "rl_highpass":     netlist_builder.rl_highpass,
        "rlc_bandpass":    netlist_builder.rlc_bandpass,
        "rlc_bandstop":    netlist_builder.rlc_bandstop,
    }

    if circuit_type not in builders:
        return (
            f"Unknown circuit type '{circuit_type}'. "
            f"Supported: {', '.join(builders)}"
        )

    try:
        p = json.loads(params)
    except json.JSONDecodeError as e:
        return f"Invalid params JSON: {e}"

    if not filename.endswith(".cir"):
        filename += ".cir"

    try:
        cir_text = builders[circuit_type](**p)
    except (TypeError, ValueError) as e:
        return f"Could not build netlist: {e}"

    saved_path = netlist_builder.save_netlist(cir_text, filename)
    return f"Netlist saved to {saved_path.name}\n\nNetlist:\n{cir_text}"


# ═══════════════════════════════════════════════════════════════════
# GRAPH
# ═══════════════════════════════════════════════════════════════════

TOOLS = [generate_netlist, run_ltspice_simulation]

SYSTEM_PROMPT = """You are a circuit design assistant. You generate LTspice netlists
from user specifications, then verify them by running a simulation.

Supported circuit types and required parameters:
  voltage_divider  — vin (V), vout (V)
  rc_lowpass       — f_c (Hz)
  rc_highpass      — f_c (Hz)
  rl_lowpass       — f_c (Hz)
  rl_highpass      — f_c (Hz)
  rlc_bandpass     — f_0 (Hz), Q (dimensionless)
  rlc_bandstop     — f_0 (Hz), Q (dimensionless)

Workflow for every request:
1. Identify the circuit type and extract numeric parameters from the user's spec.
2. Choose a descriptive filename (e.g. 'rc_lowpass_5khz.cir').
3. Call generate_netlist with circuit_type, params as JSON, and filename.
4. Call run_ltspice_simulation on the saved filename to verify the circuit.
5. Report: what was built, component values used, and whether the simulation
   confirms the spec (compare designed vs measured cutoff/frequency/voltage).

For Q factor: if the user doesn't specify, default to Q=1 for bandpass/bandstop.
Always confirm units: Hz for frequencies, V for voltages.
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


def build_generator_graph():
    """Build and compile the Level 2 circuit generator subgraph."""
    tool_node = ToolNode(TOOLS)
    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()
