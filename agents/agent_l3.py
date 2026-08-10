"""
Level 3 — Circuit Analyzer Subagent.

Given a netlist with no analysis directive, this agent:
  1. Reads the circuit topology (read_netlist_topology)
  2. Determines the appropriate analysis type (dc/ac/tran)
  3. Injects the directive, runs the simulation, explains results
     (prepare_and_run_analysis)

Tools:
  - read_netlist_topology
  - prepare_and_run_analysis
"""

import sys
from pathlib import Path

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
from parsers.netlist_parser import parse_topology, strip_analysis_directives


# ═══════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════

@tool
def read_netlist_topology(netlist_filename: str) -> str:
    """
    Read a netlist file and summarize its topology and suggested analysis.
    Always call this first before running any analysis.

    Args:
        netlist_filename: Name of the .cir file (e.g. 'mystery.cir').
                          Must exist in the circuits/ directory.

    Returns:
        Human-readable summary of components found and suggested analysis type.
    """
    netlist_path = CIRCUITS_DIR / netlist_filename
    if not netlist_path.exists():
        return f"File not found: {netlist_filename}. Use list_circuits to see available files."

    topo = parse_topology(netlist_path)

    if topo["error"]:
        return f"Could not parse netlist: {topo['error']}"

    lines = [f"Topology of {netlist_filename}:"]

    if not topo["components"]:
        lines.append("  No components found (file may be empty or comment-only).")
    else:
        for ctype, count in topo["components"].items():
            lines.append(f"  {ctype}: {count}")

    lines.append("\nProperties:")
    lines.append(f"  Reactive components (C/L): {topo['has_reactive']}")
    lines.append(f"  Time-varying sources (SINE/PULSE/PWL): {topo['has_time_varying_sources']}")

    if topo["existing_directives"]:
        lines.append("\nExisting analysis directives (will be stripped and replaced):")
        for d in topo["existing_directives"]:
            lines.append(f"  {d}")
    else:
        lines.append("\nNo analysis directive found — this is expected.")

    lines.append(f"\nSuggested analysis: {topo['suggested_analysis'].upper()}")
    lines.append(f"Suggested directive: {topo['suggested_directive']}")

    if topo["skipped_lines"]:
        lines.append(f"\nNote: {len(topo['skipped_lines'])} malformed line(s) were skipped.")

    return "\n".join(lines)


@tool
def prepare_and_run_analysis(
    netlist_filename: str,
    analysis_type: str,
    tran_duration_ms: float = 10.0,
) -> str:
    """
    Inject an analysis directive into a netlist, run the simulation,
    and return parsed results.

    Args:
        netlist_filename:  Original .cir file in the circuits/ directory.
        analysis_type:     One of "dc", "ac", or "tran".
        tran_duration_ms:  Duration in milliseconds for transient analysis
                           (default 10). Ignored for dc and ac.

    Returns:
        Simulation results with analysis context.
    """
    netlist_path = CIRCUITS_DIR / netlist_filename
    if not netlist_path.exists():
        return f"File not found: {netlist_filename}."

    analysis_type = analysis_type.lower().strip()
    if analysis_type not in ("dc", "ac", "tran"):
        return f"Unknown analysis_type '{analysis_type}'. Must be 'dc', 'ac', or 'tran'."

    # Read and parse topology (also gives us the decoded raw text)
    topo = parse_topology(netlist_path)
    if topo["error"]:
        return f"Could not read netlist: {topo['error']}"

    # Compute output filename — avoid double _analyzed suffix
    stem = Path(netlist_filename).stem
    if stem.endswith("_analyzed"):
        stem = stem[: -len("_analyzed")]
    analyzed_filename = f"{stem}_analyzed.cir"
    analyzed_path = CIRCUITS_DIR / analyzed_filename

    # Strip existing analysis directives and .end
    body = strip_analysis_directives(topo["raw_text"])

    # SPICE always ignores the first line as the title. If the original netlist
    # has no title comment (starts directly with a component), prepend one so
    # that the voltage source / first component isn't silently discarded.
    first_nonempty = next((l for l in body.splitlines() if l.strip()), "")
    if not first_nonempty.strip().startswith("*"):
        body = f"* {analyzed_filename}\n" + body

    # Build the directive block to inject
    if analysis_type == "tran":
        directive_block = f".tran {tran_duration_ms}m"
    elif analysis_type == "ac":
        directive_block = ".ac dec 100 1 1Meg"
    else:  # dc
        directive_block = ".op\n.backanno"

    # Compose the modified netlist
    modified_text = body + f"\n\n{directive_block}\n.end\n"

    # Write as UTF-8 (LTspice accepts UTF-8 input)
    analyzed_path.write_text(modified_text, encoding="utf-8")

    # Run the simulation
    result = run_simulation(analyzed_path)

    if not result.success:
        return f"Simulation FAILED: {result.error_message}"

    parts = [
        f"Prepared '{analyzed_filename}' with: {directive_block}",
        f"Simulation completed in {result.elapsed_seconds:.2f}s.",
    ]

    if result.log_path and result.log_path.exists():
        log_data = parse_log(result.log_path)
        if log_data.has_errors:
            parts.append("\nERRORS found in log:")
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
            parts.append(f"\nAnalysis types detected: {', '.join(log_data.analysis_types)}")

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


# ═══════════════════════════════════════════════════════════════════
# GRAPH
# ═══════════════════════════════════════════════════════════════════

TOOLS = [read_netlist_topology, prepare_and_run_analysis]

SYSTEM_PROMPT = """You are a circuit analysis assistant. You receive netlists that
have no analysis directive and determine the correct analysis to run.

Workflow for every request:
1. Call read_netlist_topology with the netlist filename to understand the circuit.
2. Reason about what analysis is appropriate based on the topology:
   - Only resistors + DC sources → "dc" (DC operating point)
   - Capacitors or inductors present → "ac" (frequency sweep)
   - Time-varying sources (SINE, PULSE, PWL) → "tran" (transient)
3. Call prepare_and_run_analysis with the filename, chosen analysis_type,
   and (if tran) an appropriate tran_duration_ms.
4. Explain the results in context of what kind of circuit this is:
   - DC: what are the node voltages, name the circuit type (divider, amplifier, etc.)
   - AC: what is the -3dB frequency, is this a low-pass/high-pass/bandpass filter
   - Tran: waveform behavior, period, amplitude, signal shape

Always state the circuit type in plain English. Be concise but educational.
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


def build_l3_graph():
    """Build and compile the Level 3 circuit analyzer subgraph."""
    tool_node = ToolNode(TOOLS)
    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()
