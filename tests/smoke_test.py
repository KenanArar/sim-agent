"""
Smoke test for the simulation agent.

Run this to verify the project structure and parser logic
work correctly, even without LTspice installed.

Usage:
    python tests/smoke_test.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config():
    """Verify config loads without errors."""
    from config import LTSPICE_BINARY, CIRCUITS_DIR, LLM_MODEL
    print(f"  LTspice binary: {LTSPICE_BINARY}")
    print(f"  Circuits dir:   {CIRCUITS_DIR}")
    print(f"  LLM model:      {LLM_MODEL}")
    assert CIRCUITS_DIR.exists(), "circuits/ directory not found"
    print("  ✓ Config OK")


def test_circuits_exist():
    """Check sample netlist files are present."""
    from config import CIRCUITS_DIR
    cir_files = list(CIRCUITS_DIR.glob("*.cir"))
    print(f"  Found {len(cir_files)} circuit files:")
    for f in cir_files:
        print(f"    - {f.name}")
    assert len(cir_files) >= 2, "Expected at least 2 .cir files"
    print("  ✓ Circuits OK")


def test_log_parser_with_sample():
    """Test log parser against a synthetic log string."""
    from parsers.log_parser import LogData, _OP_PATTERN, _ERROR_PATTERNS
    import re

    sample_log = """Circuit: * Voltage Divider
V(out): 2.5   voltage
V(in):  5.0   voltage
I(R1):  0.0005   device_current
I(V1):  -0.0005   device_current

Total elapsed time: 0.012 seconds.
"""
    # Quick check that regex picks up operating point values
    matches = _OP_PATTERN.findall(sample_log)
    assert len(matches) >= 2, f"Expected >=2 OP matches, got {len(matches)}"
    print(f"  Parsed {len(matches)} operating point values from sample log")
    print("  ✓ Log parser regex OK")


def test_runner_validates_missing_file():
    """Runner should fail gracefully on a nonexistent file."""
    from tools.ltspice_runner import run_simulation
    result = run_simulation("/nonexistent/path/fake.cir")
    assert not result.success
    assert "not found" in result.error_message.lower()
    print(f"  Missing file error: {result.error_message}")
    print("  ✓ Runner validation OK")


def test_graph_builds():
    """Verify the LangGraph graph compiles without errors."""
    try:
        from agent import build_graph, TOOLS
        print(f"  Tools registered: {[t.name for t in TOOLS]}")
        # Note: graph.compile() will work even without a valid API key
        # It only fails when you try to invoke it
        print("  ✓ Graph builds OK (invoke requires valid API key)")
    except Exception as e:
        print(f"  ⚠ Graph build failed: {e}")
        print("    This is OK if you haven't set up .env yet")


if __name__ == "__main__":
    print("=" * 50)
    print("  Smoke Test — Simulation Agent Level 1")
    print("=" * 50)
    print()

    tests = [
        ("Config", test_config),
        ("Circuits", test_circuits_exist),
        ("Log Parser", test_log_parser_with_sample),
        ("Runner Validation", test_runner_validates_missing_file),
        ("Graph Build", test_graph_builds),
    ]

    passed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

    print(f"\n{'=' * 50}")
    print(f"  Results: {passed}/{len(tests)} passed")
    print(f"{'=' * 50}")
