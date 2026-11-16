"""
Root Orchestrator Agent.

Routes user requests to the appropriate specialized subagent:
  - simulate_circuit  →  Level 1 (run/interpret existing simulations)
  - generate_circuit  →  Level 2 (design new circuits from specs)
  - analyze_circuit   →  Level 3 (pick + run analysis for a bare netlist)
  - verify_rtl        →  Level 4 (SystemVerilog testbenches, cocotb regression triage)

The root agent never runs simulations or builds netlists itself.
It only decides which subagent should handle the request.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from state import AgentState
from config import LLM_MODEL, OPENAI_API_KEY
from agents.agent_l1 import build_simulation_graph
from agents.agent_l2 import build_generator_graph
from agents.agent_l3 import build_l3_graph
from agents.agent_l4 import build_l4_graph


# ═══════════════════════════════════════════════════════════════════
# TOOLS (each wraps a subagent)
# ═══════════════════════════════════════════════════════════════════

@tool
def simulate_circuit(request: str) -> str:
    """
    Run or interpret a circuit simulation.
    Use this when the user wants to simulate an existing netlist, list
    available circuits, or read a netlist file.

    Args:
        request: The user's full simulation request in natural language.
    """
    graph = build_simulation_graph()
    result = graph.invoke({"messages": [HumanMessage(content=request)]})
    return result["messages"][-1].content


@tool
def generate_circuit(request: str) -> str:
    """
    Design and build a new circuit from a natural language specification,
    then verify it by simulation.
    Use this when the user describes a circuit they want created
    (e.g. 'build a 5kHz low-pass filter' or 'make a voltage divider').

    Args:
        request: The user's full circuit specification in natural language.
    """
    graph = build_generator_graph()
    result = graph.invoke({"messages": [HumanMessage(content=request)]})
    return result["messages"][-1].content


@tool
def analyze_circuit(request: str) -> str:
    """
    Determine and run the appropriate analysis for a netlist that has no
    analysis directive. Use when the user has an existing circuit file but
    doesn't know what simulation to run, or wants the agent to figure it out.

    Examples: "analyze mystery.cir", "what does unknown_amp.cir do",
              "figure out what simulation to run for my_filter.cir"

    Args:
        request: The user's full analysis request in natural language.
    """
    graph = build_l3_graph()
    result = graph.invoke({"messages": [HumanMessage(content=request)]})
    return result["messages"][-1].content


@tool
def verify_rtl(request: str) -> str:
    """
    Verify SystemVerilog RTL (digital logic, not SPICE netlists): generate a
    testbench for a module, or run its cocotb regression and cluster the
    failures by root cause.

    Examples: "generate a testbench for counter.sv",
              "run the regression on alu.sv and tell me why it fails"

    Args:
        request: The user's full RTL verification request in natural language.
    """
    graph = build_l4_graph()
    result = graph.invoke({"messages": [HumanMessage(content=request)]})
    return result["messages"][-1].content



# ═══════════════════════════════════════════════════════════════════
# GRAPH
# ═══════════════════════════════════════════════════════════════════

TOOLS = [simulate_circuit, generate_circuit, analyze_circuit, verify_rtl]

SYSTEM_PROMPT = """You are the main interface for an LTspice circuit simulation assistant.
You route user requests to the right specialist:

- simulate_circuit: Use when the user wants to run, analyze, or read an existing circuit.
  Examples: "run the low-pass filter", "what's in rc_lowpass.cir", "list circuits"

- generate_circuit: Use when the user wants to design something new from a specification.
  Examples: "build a 5kHz RC low-pass", "make a voltage divider that gives 3.3V from 5V",
            "design an RLC bandpass centered at 10kHz with Q=2"

- analyze_circuit: Use when the user has a netlist with no analysis directive,
  or wants the agent to determine what simulation to run.
  Examples: "analyze mystery.cir", "what does unknown.cir do",
            "figure out and run the right simulation for amp.cir"

- verify_rtl: Use for SystemVerilog / RTL / digital design work (.sv files), as
  opposed to analog SPICE netlists (.cir files).
  Examples: "generate a testbench for counter.sv", "run the alu regression",
            "why are the alu tests failing"

Pass the user's request verbatim to the appropriate tool — do not paraphrase or
truncate it. The specialist agents will handle the details.
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


def build_graph():
    """Build and compile the root orchestrator graph."""
    tool_node = ToolNode(TOOLS)
    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()
