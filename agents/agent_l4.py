"""
Level 4 — RTL Verification Subagent.

Extends the agent from SPICE netlists to SystemVerilog RTL:

  1. Testbench generation: parse a module's ports, emit a self-checking
     SystemVerilog testbench (clock/reset, random stimulus, X/Z checks),
     then compile and run it with Icarus Verilog to confirm it works.
  2. Regression triage: run the module's cocotb tests over many random seeds,
     cluster the failures by normalized signature, then read the RTL and
     propose a root cause for each cluster.

Tools:
  - list_rtl
  - read_rtl
  - generate_sv_testbench
  - run_regression
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
from config import LLM_MODEL, OPENAI_API_KEY, RTL_DIR, RTL_TB_DIR, RTL_TESTS_DIR
from generators.tb_generator import parse_module, generate_testbench, find_clock, find_reset
from tools.rtl_runner import run_sv_testbench, run_cocotb_regression
from parsers.failure_clusterer import cluster_failures, format_clusters


def _rtl_path(filename: str) -> Path:
    # Path(...).name keeps lookups inside rtl/
    return RTL_DIR / Path(filename).name


# ═══════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════

@tool
def list_rtl() -> str:
    """List the SystemVerilog files in rtl/ and the cocotb test modules in rtl_tests/."""
    rtl = sorted(p.name for p in RTL_DIR.glob("*.sv"))
    tests = sorted(p.stem for p in RTL_TESTS_DIR.glob("test_*.py"))
    return (f"RTL files: {', '.join(rtl) or 'none'}\n"
            f"cocotb test modules: {', '.join(tests) or 'none'}")


@tool
def read_rtl(filename: str) -> str:
    """
    Return an RTL file's source with line numbers. Use this to point at the
    specific line responsible for a failure cluster.

    Args:
        filename: e.g. 'alu.sv' (must be in rtl/).
    """
    path = _rtl_path(filename)
    if not path.exists():
        return f"File not found: {filename}. Use list_rtl."
    return "\n".join(f"{i:3d}| {line}" for i, line in enumerate(path.read_text().splitlines(), 1))


@tool
def generate_sv_testbench(rtl_filename: str, n_cycles: int = 200) -> str:
    """
    Generate a self-checking SystemVerilog testbench for an RTL module, then
    compile and run it with Icarus Verilog.

    Args:
        rtl_filename: e.g. 'counter.sv' (must be in rtl/).
        n_cycles:     number of random-stimulus cycles (default 200).
    """
    path = _rtl_path(rtl_filename)
    if not path.exists():
        return f"File not found: {rtl_filename}. Use list_rtl."
    try:
        info = parse_module(path.read_text())
    except ValueError as e:
        return f"Could not parse module header: {e}"

    RTL_TB_DIR.mkdir(parents=True, exist_ok=True)
    tb_path = RTL_TB_DIR / f"tb_{info.name}.sv"
    tb_path.write_text(generate_testbench(info, n_cycles=max(1, min(n_cycles, 100_000))))

    clk, rst = find_clock(info), find_reset(info)
    lines = [
        f"Parsed module '{info.name}':",
        f"  parameters: {', '.join(f'{n}={v}' for n, v in info.params) or 'none'}",
        *(f"  {p.direction:6} {p.width:>12} {p.name}" for p in info.ports),
        f"  clock: {clk.name if clk else 'none (combinational stimulus)'}",
        f"  reset: {rst.name if rst else 'none'}",
        f"Wrote testbench: {tb_path.relative_to(RTL_DIR.parent)}",
    ]

    run = run_sv_testbench(tb_path, [path], f"tb_{info.name}")
    if run.stage == "compile" and not run.ok:
        lines.append(f"COMPILE FAILED:\n{run.output}")
    else:
        lines.append(f"Simulation: {'PASS' if run.ok else 'FAIL'}")
        if run.vcd:
            lines.append(f"Waveform: {run.vcd.relative_to(RTL_DIR.parent)}")
        lines.append(f"Simulator output:\n{run.output[-3000:]}")
    return "\n".join(lines)


@tool
def run_regression(rtl_filename: str, n_seeds: int = 10, test_module: str = "") -> str:
    """
    Run the cocotb tests for an RTL module across multiple random seeds and
    return the failures clustered by likely root cause.

    Args:
        rtl_filename: e.g. 'alu.sv' (must be in rtl/).
        n_seeds:      how many seeds to run (default 10, max 50).
        test_module:  cocotb module name; defaults to 'test_<module>'.
    """
    path = _rtl_path(rtl_filename)
    if not path.exists():
        return f"File not found: {rtl_filename}. Use list_rtl."
    try:
        top = parse_module(path.read_text()).name
    except ValueError as e:
        return f"Could not parse module header: {e}"

    test_module = test_module or f"test_{top}"
    if not (RTL_TESTS_DIR / f"{test_module}.py").exists():
        return f"No cocotb test module rtl_tests/{test_module}.py for '{top}'."

    seeds = list(range(1, max(1, min(n_seeds, 50)) + 1))
    res = run_cocotb_regression([path], top, test_module, seeds)
    if res.error:
        return f"Regression could not run: {res.error}"

    total = res.passed + len(res.failures)
    header = (f"Regression of '{top}' with {test_module}: {len(seeds)} seeds, "
              f"{total} test runs, {res.passed} passed, {len(res.failures)} failed.")
    return header + "\n\n" + format_clusters(cluster_failures(res.failures))


# ═══════════════════════════════════════════════════════════════════
# GRAPH
# ═══════════════════════════════════════════════════════════════════

TOOLS = [list_rtl, read_rtl, generate_sv_testbench, run_regression]

SYSTEM_PROMPT = """You are an RTL verification assistant for SystemVerilog designs.

Testbench requests ("generate a testbench for counter.sv"):
  Call generate_sv_testbench, then summarize the detected ports, clock and
  reset, whether the testbench passed, and where the testbench and VCD are.

Regression or debugging requests ("run the regression on alu.sv", "why is alu failing"):
  1. Call run_regression.
  2. If there are failure clusters, call read_rtl on the same file.
  3. For EACH cluster, give a root-cause hypothesis:
     - what the failures have in common (look at the example got vs exp values
       for a pattern, e.g. always off by one, or only when the sign bit differs)
     - the specific RTL line number you believe is responsible, and why
     - a suggested fix
  Clusters come from a message-signature heuristic, so call your conclusion a
  hypothesis, and say so if two clusters look like they share a cause.
  Do not edit any files.

Use list_rtl if the user doesn't name a file. Be concise.
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


def build_l4_graph():
    """Build and compile the Level 4 RTL verification subgraph."""
    tool_node = ToolNode(TOOLS)
    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()
