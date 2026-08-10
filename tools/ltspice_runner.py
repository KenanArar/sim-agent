"""
LTspice CLI wrapper for macOS.

Runs LTspice in batch mode via subprocess and returns paths
to the generated output files (.log, .raw).
"""

import subprocess
import time
from pathlib import Path
from dataclasses import dataclass

from config import LTSPICE_BINARY, SIMULATION_TIMEOUT


@dataclass
class SimulationResult:
    """Result of an LTspice simulation run."""
    success: bool
    netlist_path: Path
    log_path: Path | None
    raw_path: Path | None
    elapsed_seconds: float
    error_message: str | None = None


def verify_ltspice_install() -> bool:
    """Check that LTspice binary exists at the configured path."""
    return Path(LTSPICE_BINARY).exists()


def run_simulation(netlist_path: str | Path) -> SimulationResult:
    """
    Run an LTspice simulation in batch mode.

    Args:
        netlist_path: Path to a .cir or .asc netlist file.

    Returns:
        SimulationResult with paths to output files and status.
    """
    netlist_path = Path(netlist_path).resolve()

    # ── Validate inputs ─────────────────────────────────────────
    if not netlist_path.exists():
        return SimulationResult(
            success=False,
            netlist_path=netlist_path,
            log_path=None,
            raw_path=None,
            elapsed_seconds=0,
            error_message=f"Netlist not found: {netlist_path}",
        )

    if not verify_ltspice_install():
        return SimulationResult(
            success=False,
            netlist_path=netlist_path,
            log_path=None,
            raw_path=None,
            elapsed_seconds=0,
            error_message=f"LTspice not found at: {LTSPICE_BINARY}",
        )

    # ── Determine expected output paths ─────────────────────────
    stem = netlist_path.stem
    parent = netlist_path.parent
    log_path = parent / f"{stem}.log"
    raw_path = parent / f"{stem}.raw"

    # Remove stale output files so we don't read old results
    for f in [log_path, raw_path]:
        if f.exists():
            f.unlink()

    # ── Run LTspice ─────────────────────────────────────────────
    cmd = [LTSPICE_BINARY, "-b", str(netlist_path)]

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=SIMULATION_TIMEOUT,
            text=True,
        )
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return SimulationResult(
            success=False,
            netlist_path=netlist_path,
            log_path=None,
            raw_path=None,
            elapsed_seconds=elapsed,
            error_message=f"Simulation timed out after {SIMULATION_TIMEOUT}s",
        )
    except Exception as e:
        elapsed = time.time() - start
        return SimulationResult(
            success=False,
            netlist_path=netlist_path,
            log_path=None,
            raw_path=None,
            elapsed_seconds=elapsed,
            error_message=f"Failed to run LTspice: {e}",
        )

    # ── Check for output files ──────────────────────────────────
    # LTspice on Mac doesn't always return nonzero on errors,
    # so we check for output files as the source of truth.
    log_exists = log_path.exists()
    raw_exists = raw_path.exists()

    if not log_exists:
        return SimulationResult(
            success=False,
            netlist_path=netlist_path,
            log_path=None,
            raw_path=None,
            elapsed_seconds=elapsed,
            error_message="No .log file generated — simulation may have failed silently.",
        )

    return SimulationResult(
        success=True,
        netlist_path=netlist_path,
        log_path=log_path if log_exists else None,
        raw_path=raw_path if raw_exists else None,
        elapsed_seconds=elapsed,
    )
