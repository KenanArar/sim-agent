"""
Parser for LTspice .log files.

Extracts simulation metadata, errors, warnings, and operating
point results from the text-based log output.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class LogData:
    """Structured data extracted from an LTspice .log file."""
    raw_text: str
    analysis_types: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    operating_point: dict[str, float] = field(default_factory=dict)
    total_elapsed: str | None = None
    has_errors: bool = False

    def summary(self) -> str:
        """Human-readable summary of the log."""
        parts = []

        if self.analysis_types:
            parts.append(f"Analysis: {', '.join(self.analysis_types)}")

        if self.has_errors:
            parts.append(f"ERRORS ({len(self.errors)}):")
            for e in self.errors:
                parts.append(f"  - {e}")

        if self.warnings:
            parts.append(f"Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                parts.append(f"  - {w}")

        if self.operating_point:
            parts.append("Operating Point:")
            for name, val in self.operating_point.items():
                parts.append(f"  {name} = {val:.6g}")

        if self.total_elapsed:
            parts.append(f"Simulation time: {self.total_elapsed}")

        return "\n".join(parts) if parts else "Log parsed — no notable content."


# ── Regex patterns for LTspice log parsing ──────────────────────────

# Matches lines like: "V(out)                         2.5   voltage"
# (LTspice Mac omits the colon; values are whitespace-separated from the name)
_OP_PATTERN = re.compile(
    r"^\s*([VI]\([^)]+\))\s+([-\d.eE+]+)\s+\w+",
    re.MULTILINE,
)

# Matches analysis directives echoed in the log
_ANALYSIS_PATTERN = re.compile(
    r"^\s*\.(ac|dc|tran|op|noise|tf|sens)\b",
    re.MULTILINE | re.IGNORECASE,
)

# Matches the total elapsed time line
_ELAPSED_PATTERN = re.compile(
    r"Total elapsed time:\s*(.+)",
    re.IGNORECASE,
)

# Error patterns — LTspice uses several formats
_ERROR_PATTERNS = [
    re.compile(r"^Error[: ].*", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^.*convergence.*fail.*", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^.*singular matrix.*", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^.*floating node.*", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^.*unknown subcircuit.*", re.MULTILINE | re.IGNORECASE),
]

_WARNING_PATTERN = re.compile(r"^Warning[: ].*", re.MULTILINE | re.IGNORECASE)


def parse_log(log_path: str | Path) -> LogData:
    """
    Parse an LTspice .log file into structured data.

    Args:
        log_path: Path to the .log file.

    Returns:
        LogData with extracted fields.
    """
    log_path = Path(log_path)
    raw_bytes = log_path.read_bytes()
    # LTspice on macOS writes log files in UTF-16 LE (with or without BOM)
    if raw_bytes.startswith(b'\xff\xfe') or raw_bytes.startswith(b'\xfe\xff'):
        text = raw_bytes.decode('utf-16')
    elif len(raw_bytes) >= 2 and raw_bytes[1] == 0:
        text = raw_bytes.decode('utf-16-le', errors='replace')
    else:
        text = raw_bytes.decode('utf-8', errors='replace')

    data = LogData(raw_text=text)

    # ── Analysis types ──────────────────────────────────────────
    for m in _ANALYSIS_PATTERN.finditer(text):
        atype = m.group(1).upper()
        if atype == "OP":
            atype = "DC Operating Point"
        elif atype == "AC":
            atype = "AC Analysis"
        elif atype == "TRAN":
            atype = "Transient"
        elif atype == "DC":
            atype = "DC Sweep"
        if atype not in data.analysis_types:
            data.analysis_types.append(atype)

    # LTspice Mac doesn't always echo the .op directive at line start —
    # detect it from the result header instead
    if re.search(r"Operating.*Point.*Solution", text, re.IGNORECASE):
        if "DC Operating Point" not in data.analysis_types:
            data.analysis_types.append("DC Operating Point")

    # ── Operating point values ──────────────────────────────────
    for m in _OP_PATTERN.finditer(text):
        name = m.group(1)
        value = float(m.group(2))
        data.operating_point[name] = value

    # ── Errors ──────────────────────────────────────────────────
    for pattern in _ERROR_PATTERNS:
        for m in pattern.finditer(text):
            err = m.group(0).strip()
            if err not in data.errors:
                data.errors.append(err)

    data.has_errors = len(data.errors) > 0

    # ── Warnings ────────────────────────────────────────────────
    for m in _WARNING_PATTERN.finditer(text):
        warn = m.group(0).strip()
        if warn not in data.warnings:
            data.warnings.append(warn)

    # ── Elapsed time ────────────────────────────────────────────
    m = _ELAPSED_PATTERN.search(text)
    if m:
        data.total_elapsed = m.group(1).strip()

    return data
