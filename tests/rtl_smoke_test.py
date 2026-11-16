"""
Smoke test for Level 4 (RTL verification). Needs Icarus Verilog + cocotb, but
no API key: it exercises the generator, runner and clusterer directly.

Usage:
    python tests/rtl_smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RTL_BUILD_DIR, RTL_DIR, RTL_TB_DIR
from generators.tb_generator import parse_module, generate_testbench, find_clock, find_reset
from parsers.failure_clusterer import normalize, cluster_failures, format_clusters
from tools.rtl_runner import run_sv_testbench, run_cocotb_regression


def test_parse_ports():
    info = parse_module((RTL_DIR / "counter.sv").read_text())
    assert info.name == "counter"
    assert info.params == [("WIDTH", "8")]
    assert [(p.direction, p.name, p.width) for p in info.ports] == [
        ("input", "clk", ""), ("input", "rst_n", ""), ("input", "en", ""),
        ("output", "count", "[WIDTH-1:0]")]
    assert find_clock(info).name == "clk" and find_reset(info).name == "rst_n"
    # shared declarations: "input logic [3:0] a, b" -> b inherits direction
    info = parse_module("module m(input logic [3:0] a, b, output logic y); endmodule")
    assert [(p.direction, p.name) for p in info.ports] == [("input", "a"), ("input", "b"), ("output", "y")]
    print("  ✓ Port parsing OK")


def test_normalize():
    a = normalize("SUB mismatch: a=0x05 b=0x03 got=0x01 exp=0x02\nassert 1 == 2")
    b = normalize("SUB mismatch: a=0xef b=0xaf got=0x3f exp=0x40")
    assert a == b == "SUB mismatch: a=<HEX> b=<HEX> got=<HEX> exp=<HEX>", a
    assert normalize("count=17 at 250.5ns, r2") == "count=<N> at <N>ns, r2"
    print("  ✓ Signature normalization OK")


def test_generated_testbenches_pass():
    RTL_TB_DIR.mkdir(parents=True, exist_ok=True)
    for f in ("alu.sv", "counter.sv"):
        info = parse_module((RTL_DIR / f).read_text())
        tb = RTL_TB_DIR / f"tb_{info.name}.sv"
        tb.write_text(generate_testbench(info, n_cycles=50))
        r = run_sv_testbench(tb, [RTL_DIR / f], f"tb_{info.name}")
        assert r.ok, f"{f}: {r.output}"
        print(f"  ✓ Generated tb_{info.name}.sv compiles and passes")


def test_generated_testbench_catches_x():
    """An undriven output must make the generated testbench FAIL."""
    work = RTL_BUILD_DIR / "smoke"
    work.mkdir(parents=True, exist_ok=True)
    dut = work / "undriven.sv"
    dut.write_text("module undriven(input logic a, output logic y); endmodule\n")
    tb = work / "tb_undriven.sv"
    tb.write_text(generate_testbench(parse_module(dut.read_text()), n_cycles=5))
    r = run_sv_testbench(tb, [dut], "tb_undriven")
    assert r.stage == "run" and not r.ok and "is X/Z" in r.output, r.output
    print("  ✓ Generated testbench flags X on an undriven output")


def test_regression_clusters_buggy_alu():
    res = run_cocotb_regression([RTL_DIR / "alu.sv"], "alu", "test_alu", [1, 2, 3])
    assert res.error is None, res.error
    clusters = cluster_failures(res.failures)
    sigs = sorted(c.signature.split()[0] for c in clusters)
    assert sigs == ["SLT", "SUB"], format_clusters(clusters)
    print(f"  ✓ {len(res.failures)} failures -> {len(clusters)} clusters ({', '.join(sigs)})")


def test_regression_passes_on_fixed_alu():
    """Apply the two documented fixes to a copy: every test should pass,
    which shows each cluster corresponded to exactly one bug."""
    src = (RTL_DIR / "alu.sv").read_text()
    fixed = (src.replace("y = a + ~b;", "y = a + ~b + 8'd1;")
                .replace("y = {7'b0, a < b};", "y = {7'b0, $signed(a) < $signed(b)};"))
    assert fixed != src
    work = RTL_BUILD_DIR / "smoke"
    work.mkdir(parents=True, exist_ok=True)
    (work / "alu_fixed.sv").write_text(fixed)
    res = run_cocotb_regression([work / "alu_fixed.sv"], "alu", "test_alu", [1, 2, 3])
    assert res.error is None and not res.failures, format_clusters(cluster_failures(res.failures))
    print(f"  ✓ Fixed ALU: {res.passed} passed, 0 failed")


if __name__ == "__main__":
    tests = [test_parse_ports, test_normalize, test_generated_testbenches_pass,
             test_generated_testbench_catches_x, test_regression_clusters_buggy_alu,
             test_regression_passes_on_fixed_alu]
    for t in tests:
        print(f"\n{t.__name__}")
        t()
    print(f"\nAll {len(tests)} RTL smoke tests passed.")
