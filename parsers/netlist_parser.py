"""
Parser for SPICE netlist files.

Extracts topology information (component types, reactive elements,
time-varying sources) and determines the appropriate LTspice analysis
type to run when no directive is present in the file.
"""

import re
from pathlib import Path


# ── Regex patterns ────────────────────────────────────────────────────

# Analysis directives to detect and strip
_DIRECTIVE_PATTERN = re.compile(
    r"^\s*\.(op|ac|dc|tran)\b.*",
    re.IGNORECASE,
)

# Time-varying waveform keywords in source lines
_TIME_VARYING_PATTERN = re.compile(
    r"\b(SINE|PULSE|PWL|EXP|SFFM)\s*\(",
    re.IGNORECASE,
)

# .end line
_END_PATTERN = re.compile(r"^\s*\.end\s*$", re.IGNORECASE)

# Strip analysis directives + .end from a block of text
_STRIP_DIRECTIVES_RE = re.compile(
    r"^\s*\.(op|ac|dc|tran)\b.*\n?",
    re.IGNORECASE | re.MULTILINE,
)
_STRIP_END_RE = re.compile(r"^\s*\.end\s*\n?", re.IGNORECASE | re.MULTILINE)


# ── Encoding helper (mirrors log_parser.py) ───────────────────────────

def _read_netlist_text(path: Path) -> str:
    """Read a netlist file, detecting UTF-16 LE (written by LTspice GUI)."""
    raw_bytes = path.read_bytes()
    if raw_bytes.startswith(b'\xff\xfe') or raw_bytes.startswith(b'\xfe\xff'):
        return raw_bytes.decode('utf-16')
    elif len(raw_bytes) >= 2 and raw_bytes[1] == 0:
        return raw_bytes.decode('utf-16-le', errors='replace')
    else:
        return raw_bytes.decode('utf-8', errors='replace')


# ── Public helpers ────────────────────────────────────────────────────

def strip_analysis_directives(text: str) -> str:
    """
    Remove all analysis directive lines (.op, .ac, .dc, .tran) and
    the .end line from a netlist text block.

    Returns the cleaned body text (without trailing whitespace).
    """
    body = _STRIP_DIRECTIVES_RE.sub("", text)
    body = _STRIP_END_RE.sub("", body)
    return body.rstrip()


def generate_dc_measurements(component_list: list[str]) -> str:
    """
    Generate .meas op directives for each resistor in the component list,
    plus .backanno. Appended after .op to get explicit voltage-drop readings
    across each resistor in the simulation output.

    For resistor 'R1 A B value', generates '.meas op VR1 V(A,B)'.
    Signs follow the component's node order; negative values simply indicate
    the reference direction is opposite to actual current flow.

    Args:
        component_list: Raw component lines from parse_topology().

    Returns:
        Multi-line string of .meas directives + .backanno, or empty string
        if no resistors are found.
    """
    lines = []
    for line in component_list:
        tokens = line.split()
        if len(tokens) < 3:
            continue
        ref = tokens[0]
        if ref[0].upper() != "R":
            continue
        node_a = tokens[1]
        node_b = tokens[2]
        lines.append(f".meas op V{ref} V({node_a},{node_b})")

    if lines:
        lines.append(".backanno")

    return "\n".join(lines)


# ── Main function ─────────────────────────────────────────────────────

def parse_topology(netlist_path: str | Path) -> dict:
    """
    Parse a SPICE netlist and return a topology summary.

    Returns a dict with keys:
        components              dict[str, int]  — component type → count
                                e.g. {"resistors": 2, "capacitors": 1}
        has_reactive            bool  — True if any C or L present
        has_capacitors          bool
        has_inductors           bool
        has_time_varying_sources bool
        existing_directives     list[str] — .op/.ac/.dc/.tran lines found
        skipped_lines           list[str] — malformed lines ignored
        suggested_analysis      str  — "dc", "ac", or "tran"
        suggested_directive     str  — ready-to-insert SPICE line
        component_list          list[str] — raw component lines (for LLM)
        raw_text                str  — full decoded file text
        error                   str | None
    """
    netlist_path = Path(netlist_path)

    result = {
        "components": {},
        "has_reactive": False,
        "has_capacitors": False,
        "has_inductors": False,
        "has_time_varying_sources": False,
        "existing_directives": [],
        "skipped_lines": [],
        "suggested_analysis": "dc",
        "suggested_directive": ".op",
        "component_list": [],
        "raw_text": "",
        "error": None,
    }

    # ── Read file ──────────────────────────────────────────────────
    try:
        text = _read_netlist_text(netlist_path)
    except Exception as e:
        result["error"] = str(e)
        return result

    result["raw_text"] = text

    # ── Parse lines ────────────────────────────────────────────────
    title_skipped = False

    for line in text.splitlines():
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # First non-empty line is the title in SPICE format
        if not title_skipped:
            title_skipped = True
            continue

        # Comment lines
        if stripped.startswith("*"):
            continue

        # Directive lines
        if stripped.startswith("."):
            if _END_PATTERN.match(stripped):
                continue  # .end — skip
            if _DIRECTIVE_PATTERN.match(stripped):
                result["existing_directives"].append(stripped)
            # All other directives (.model, .subckt, .param, etc.) are ignored
            continue

        # Component lines — first character (case-insensitive) is the type prefix
        tokens = stripped.split()
        if len(tokens) < 3:
            result["skipped_lines"].append(stripped)
            continue

        prefix = stripped[0].upper()
        result["component_list"].append(stripped)

        if prefix == "R":
            result["components"]["resistors"] = result["components"].get("resistors", 0) + 1

        elif prefix == "C":
            result["components"]["capacitors"] = result["components"].get("capacitors", 0) + 1
            result["has_capacitors"] = True
            result["has_reactive"] = True

        elif prefix == "L":
            result["components"]["inductors"] = result["components"].get("inductors", 0) + 1
            result["has_inductors"] = True
            result["has_reactive"] = True

        elif prefix == "V":
            result["components"]["voltage_sources"] = result["components"].get("voltage_sources", 0) + 1
            if _TIME_VARYING_PATTERN.search(stripped):
                result["has_time_varying_sources"] = True

        elif prefix == "I":
            result["components"]["current_sources"] = result["components"].get("current_sources", 0) + 1
            if _TIME_VARYING_PATTERN.search(stripped):
                result["has_time_varying_sources"] = True

        elif prefix == "X":
            result["components"]["subcircuits"] = result["components"].get("subcircuits", 0) + 1

        elif prefix in ("D",):
            result["components"]["diodes"] = result["components"].get("diodes", 0) + 1

        elif prefix in ("Q",):
            result["components"]["bjts"] = result["components"].get("bjts", 0) + 1

        elif prefix in ("M",):
            result["components"]["mosfets"] = result["components"].get("mosfets", 0) + 1

        else:
            # Unknown prefix — count generically
            key = f"other_{prefix}"
            result["components"][key] = result["components"].get(key, 0) + 1

    # ── Analysis selection (deterministic, priority order) ─────────
    if result["has_time_varying_sources"]:
        result["suggested_analysis"] = "tran"
        result["suggested_directive"] = ".tran 10m"
    elif result["has_reactive"]:
        result["suggested_analysis"] = "ac"
        result["suggested_directive"] = ".ac dec 100 1 1Meg"
    else:
        result["suggested_analysis"] = "dc"
        result["suggested_directive"] = ".op"

    return result
