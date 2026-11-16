"""
SystemVerilog testbench generator.

Parses an RTL module's header (name, parameters, ANSI-style port list) and emits
a self-checking SystemVerilog testbench:

  - instantiates the DUT, copying its parameter defaults as localparams
  - generates a clock and a reset sequence if it finds clk / reset inputs
    (active-low when the reset name ends in 'n', e.g. rst_n, resetn)
  - drives every other input with $urandom for N cycles
  - self-check: after reset, no output may ever be X or Z
  - dumps a VCD and prints a TB_RESULT: PASS/FAIL line that the runner greps

Like netlist_builder.py, this is deterministic. No LLM writes the SystemVerilog;
the agent only decides when to call this and interprets the result.

Limitations (deliberate, to keep it small): ANSI port lists only, no interfaces
or packed structs, and the only generic check is X/Z on outputs. Functional
checks against a reference model are what the cocotb tests in rtl_tests/ are for.
"""

import re
from dataclasses import dataclass, field


@dataclass
class Port:
    direction: str      # input | output | inout
    name: str
    width: str = ""     # e.g. "[7:0]" or "[WIDTH-1:0]", "" for 1-bit


@dataclass
class ModuleInfo:
    name: str
    params: list[tuple[str, str]] = field(default_factory=list)   # (name, default)
    ports: list[Port] = field(default_factory=list)

    @property
    def inputs(self) -> list[Port]:
        return [p for p in self.ports if p.direction == "input"]

    @property
    def outputs(self) -> list[Port]:
        return [p for p in self.ports if p.direction == "output"]


# ═══════════════════════════════════════════════════════════════════
# PARSING
# ═══════════════════════════════════════════════════════════════════

_PORT_RE = re.compile(
    r"^(?:(input|output|inout)\s+)?"
    r"(?:(?:wire|logic|reg|bit)\s+)?"
    r"(?:(?:signed|unsigned)\s+)?"
    r"(\[[^\]]+\])?\s*"
    r"(\w+)$"
)
_PARAM_RE = re.compile(r"\bparameter\s+(?:\w+\s+)?(\w+)\s*=\s*([^,]+)")
_CLK_RE = re.compile(r"(i_)?(clk|clock|aclk)(_i)?")
_RST_RE = re.compile(r"(i_)?(a?rst|reset|a?rst_?n|a?reset_?n)(_i)?")


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _balanced(text: str, i: int) -> tuple[str, int]:
    """text[i] is '('. Return (contents, index just past the matching ')')."""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1
    raise ValueError("unbalanced parentheses in module header")


def _skip_ws(text: str, i: int) -> int:
    while i < len(text) and text[i].isspace():
        i += 1
    return i


def parse_module(sv_text: str) -> ModuleInfo:
    """Extract name, parameters and ports from the first module in sv_text."""
    text = _strip_comments(sv_text)
    m = re.search(r"\bmodule\s+(\w+)", text)
    if not m:
        raise ValueError("no module declaration found")
    info = ModuleInfo(name=m.group(1))
    i = _skip_ws(text, m.end())

    # Optional parameter block: #( parameter int WIDTH = 8, ... )
    if i < len(text) and text[i] == "#":
        i = _skip_ws(text, i + 1)
        body, i = _balanced(text, i)
        info.params = [(pm.group(1), pm.group(2).strip()) for pm in _PARAM_RE.finditer(body)]
        i = _skip_ws(text, i)

    if i >= len(text) or text[i] != "(":
        raise ValueError(f"module {info.name}: expected a port list")
    body, _ = _balanced(text, i)

    direction = None
    for item in body.split(","):
        item = " ".join(item.split())
        if not item:
            continue
        pm = _PORT_RE.match(item)
        if not pm:
            raise ValueError(f"could not parse port declaration: '{item}'")
        direction = pm.group(1) or direction   # "input logic a, b" -> b inherits
        if direction is None:
            raise ValueError("non-ANSI port lists are not supported")
        info.ports.append(Port(direction, pm.group(3), pm.group(2) or ""))
    return info


def find_clock(info: ModuleInfo) -> Port | None:
    return next((p for p in info.inputs if not p.width and _CLK_RE.fullmatch(p.name)), None)


def find_reset(info: ModuleInfo) -> Port | None:
    return next((p for p in info.inputs if not p.width and _RST_RE.fullmatch(p.name)), None)


def reset_active_low(port: Port) -> bool:
    return bool(re.search(r"n(_i)?$", port.name))


# ═══════════════════════════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_testbench(info: ModuleInfo, n_cycles: int = 200) -> str:
    """Return the text of tb_<module>.sv for the parsed module."""
    clk, rst = find_clock(info), find_reset(info)
    data_inputs = [p for p in info.inputs if p not in (clk, rst)]
    tb = f"tb_{info.name}"

    def rand(p: Port) -> str:
        return f"{p.name} = {'{$urandom, $urandom}' if p.width else '$urandom_range(0, 1)'};"

    L = [
        f"// Auto-generated by sim-agent generators/tb_generator.py for module '{info.name}'.",
        "`timescale 1ns/1ps",
        f"module {tb};",
    ]
    L += [f"  localparam {n} = {v};" for n, v in info.params]
    L += [f"  localparam int N_CYCLES = {n_cycles};", ""]
    L += [f"  logic {p.width + ' ' if p.width else ''}{p.name};" for p in info.ports if p.direction != "inout"]
    L += ["  int errors = 0;", ""]

    param_map = ", ".join(f".{n}({n})" for n, _ in info.params)
    conns = ", ".join(f".{p.name}({p.name})" for p in info.ports if p.direction != "inout")
    L.append(f"  {info.name} {'#(' + param_map + ') ' if param_map else ''}dut ({conns});")
    L.append("")

    if clk:
        L += [f"  initial {clk.name} = 1'b0;", f"  always #5 {clk.name} = ~{clk.name};", ""]

    # Generic self-check: outputs must be fully resolved after reset.
    L.append("  task automatic check_outputs(input int cycle);")
    for p in info.outputs:
        L += [
            f"    if ($isunknown({p.name})) begin",
            "      errors++;",
            f'      $display("ERROR cycle %0d: output {p.name} is X/Z (%b)", cycle, {p.name});',
            "    end",
        ]
    L += ["  endtask", ""]

    L += ["  initial begin", f'    $dumpfile("{tb}.vcd");', f"    $dumpvars(0, {tb});"]
    L += [f"    {p.name} = '0;" for p in data_inputs]

    if rst:
        on, off = ("1'b0", "1'b1") if reset_active_low(rst) else ("1'b1", "1'b0")
        L.append(f"    {rst.name} = {on};")
        if clk:
            L += [f"    repeat (2) @(posedge {clk.name});", f"    @(negedge {clk.name}) {rst.name} = {off};"]
        else:
            L += ["    #10;", f"    {rst.name} = {off};"]

    L.append("    for (int i = 0; i < N_CYCLES; i++) begin")
    if clk:
        L.append(f"      @(negedge {clk.name});")
        L += [f"      {rand(p)}" for p in data_inputs]
        L += [f"      @(posedge {clk.name}); #1;"]
    else:
        L += [f"      {rand(p)}" for p in data_inputs]
        L.append("      #5;")
    L += ["      check_outputs(i);", "    end", ""]

    L += [
        '    if (errors == 0) $display("TB_RESULT: PASS (%0d cycles, no X/Z on outputs)", N_CYCLES);',
        '    else             $display("TB_RESULT: FAIL (%0d errors)", errors);',
        "    $finish;",
        "  end",
        "endmodule",
        "",
    ]
    return "\n".join(L)
