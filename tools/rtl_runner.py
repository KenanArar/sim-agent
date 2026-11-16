"""
RTL simulation runners (Icarus Verilog).

  run_sv_testbench       compile + run a plain SystemVerilog testbench with iverilog/vvp
  run_cocotb_regression  run a cocotb test module once per random seed, collect
                         results.xml from each run, and return every failure
"""

import shutil
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from config import RTL_BUILD_DIR, RTL_TESTS_DIR, SIMULATION_TIMEOUT
from parsers.failure_clusterer import Failure, parse_results_xml


def _require_iverilog() -> str | None:
    if shutil.which("iverilog") and shutil.which("vvp"):
        return None
    return "Icarus Verilog not found on PATH (macOS: brew install icarus-verilog)."


# ═══════════════════════════════════════════════════════════════════
# Plain SystemVerilog testbench
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TbRunResult:
    stage: str              # "compile" or "run": how far it got
    ok: bool                # compiled, ran, and printed TB_RESULT: PASS
    output: str
    vcd: Path | None = None


def run_sv_testbench(tb_path: Path, rtl_sources: list[Path], top: str) -> TbRunResult:
    err = _require_iverilog()
    if err:
        return TbRunResult("compile", False, err)

    build = RTL_BUILD_DIR / top
    build.mkdir(parents=True, exist_ok=True)
    vvp_file = build / f"{top}.vvp"

    # Testbench first so its `timescale applies to the RTL files that follow.
    compile_cmd = ["iverilog", "-g2012", "-s", top, "-o", str(vvp_file),
                   str(tb_path), *map(str, rtl_sources)]
    c = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=SIMULATION_TIMEOUT)
    if c.returncode != 0:
        return TbRunResult("compile", False, (c.stdout + c.stderr).strip())

    r = subprocess.run(["vvp", "-n", str(vvp_file)], cwd=build,
                       capture_output=True, text=True, timeout=SIMULATION_TIMEOUT)
    out = (c.stderr + r.stdout + r.stderr).strip()
    vcd = build / f"{top}.vcd"
    return TbRunResult("run", "TB_RESULT: PASS" in out, out, vcd if vcd.exists() else None)


# ═══════════════════════════════════════════════════════════════════
# cocotb multi-seed regression
# ═══════════════════════════════════════════════════════════════════

# LangGraph's ToolNode runs parallel tool calls in threads. Two regressions
# sharing a build dir corrupt each other's results.xml, so run one at a time.
_REGRESSION_LOCK = threading.Lock()


@dataclass
class RegressionResult:
    toplevel: str
    seeds: list[int]
    passed: int = 0
    failures: list[Failure] = field(default_factory=list)
    error: str | None = None     # infrastructure problem (build failed, etc.)


def run_cocotb_regression(rtl_sources: list[Path], toplevel: str, test_module: str,
                          seeds: list[int]) -> RegressionResult:
    """Build once, then run `test_module` once per seed. cocotb seeds Python's
    `random` from the seed, so each run gets different constrained-random vectors."""
    with _REGRESSION_LOCK:
        return _run_cocotb_regression(rtl_sources, toplevel, test_module, seeds)


def _run_cocotb_regression(rtl_sources, toplevel, test_module, seeds) -> RegressionResult:
    result = RegressionResult(toplevel, seeds)
    err = _require_iverilog()
    if err:
        result.error = err
        return result

    from cocotb_tools.runner import get_runner

    # The cocotb runner builds the simulator's PYTHONPATH from *this* process's
    # sys.path, so the test directory has to be on it (as an absolute path).
    if str(RTL_TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(RTL_TESTS_DIR))

    build = RTL_BUILD_DIR / f"cocotb_{toplevel}"
    runner = get_runner("icarus")
    try:
        runner.build(sources=rtl_sources, hdl_toplevel=toplevel, build_dir=build,
                     always=True, timescale=("1ns", "1ps"), log_file=build / "build.log")
    except Exception as e:  # noqa: BLE001 - report any build failure to the agent
        log = build / "build.log"
        result.error = f"RTL build failed: {e}\n{log.read_text()[-2000:] if log.exists() else ''}"
        return result

    for seed in seeds:
        xml = build / f"results_seed{seed}.xml"
        xml.unlink(missing_ok=True)
        runner.test(test_module=test_module, hdl_toplevel=toplevel, seed=seed,
                    test_dir=build, results_xml=str(xml), log_file=build / f"seed{seed}.log")
        if not xml.exists():
            # Simulator died before cocotb wrote results; that's a failure too.
            result.failures.append(Failure("<whole run>", seed, "SimulatorCrash",
                                           f"no results.xml (see seed{seed}.log)", "unknown"))
            continue
        try:
            passed, failures = parse_results_xml(xml, seed)
        except ET.ParseError as e:
            result.failures.append(Failure("<whole run>", seed, "BadResultsXml",
                                           f"unreadable results.xml: {e}", "unknown"))
            continue
        result.passed += len(passed)
        result.failures += failures
    return result
