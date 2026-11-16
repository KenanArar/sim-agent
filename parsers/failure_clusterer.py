"""
Cluster cocotb regression failures by likely root cause.

The idea: failures with the same cause tend to produce the same message, and
differ only in the specific values involved (operands, seeds, sim times). So
each failure message is normalized into a *signature* by replacing those values
with placeholders:

    "SUB mismatch: a=0x05 b=0x03 got=0x01 exp=0x02"
    "SUB mismatch: a=0xef b=0xaf got=0x3f exp=0x40"
        -> "SUB mismatch: a=<HEX> b=<HEX> got=<HEX> exp=<HEX>"

Failures are grouped by (exception type, signature, failing source line). This
collapses something like "40 failures across 4 tests and 10 seeds" into "2
distinct problems". The agent then reads the RTL and proposes which line causes
each cluster.

This is a heuristic. Two different bugs can share a signature, and one bug can
show up as two signatures. It's a triage tool, not a proof.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Failure:
    test: str
    seed: int
    exc_type: str
    message: str        # first line of the failure message, raw
    location: str       # "test_alu.py:52" — deepest frame in user test code


@dataclass
class Cluster:
    signature: str
    exc_type: str
    location: str
    failures: list[Failure] = field(default_factory=list)

    @property
    def tests(self) -> list[str]:
        return sorted({f.test for f in self.failures})

    @property
    def seeds(self) -> list[int]:
        return sorted({f.seed for f in self.failures})


# Order matters: hex before decimal, so "0x1f" doesn't become "<N>x1f".
_NORMALIZERS = [
    (re.compile(r"\b0x[0-9a-fA-F_]+\b"), "<HEX>"),
    (re.compile(r"\b0b[01_]+\b"), "<BIN>"),
    (re.compile(r"\b\d+'[bdhoBDHO][0-9a-fA-FxzXZ_]+"), "<SV_LIT>"),
    (re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?"), "<N>"),   # also "250.5ns" -> "<N>ns"
]
_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+)')


def normalize(message: str) -> str:
    """Turn a concrete failure message into a value-independent signature."""
    sig = message.strip().splitlines()[0] if message.strip() else "<empty message>"
    for pattern, placeholder in _NORMALIZERS:
        sig = pattern.sub(placeholder, sig)
    return " ".join(sig.split())


def _user_location(traceback_text: str) -> str:
    """Deepest traceback frame outside site-packages, i.e. where the user's
    test actually failed (an assert, or the call into cocotb that raised)."""
    frames = [(f, ln) for f, ln in _FRAME_RE.findall(traceback_text or "")
              if "site-packages" not in f and "<frozen" not in f]
    if not frames:
        return "unknown"
    f, ln = frames[-1]
    return f"{Path(f).name}:{ln}"


def parse_results_xml(xml_path: Path, seed: int) -> tuple[list[str], list[Failure]]:
    """Read one cocotb results.xml (JUnit format). Returns (passed_tests, failures)."""
    passed, failures = [], []
    for tc in ET.parse(xml_path).getroot().iter("testcase"):
        name = tc.get("name", "?")
        fail = tc.find("failure")
        if fail is None:
            fail = tc.find("error")
        if fail is None:
            if tc.find("skipped") is None:
                passed.append(name)
            continue
        msg_lines = (fail.get("message") or "").strip().splitlines()
        failures.append(Failure(
            test=name,
            seed=seed,
            exc_type=fail.get("type", "Failure"),
            message=msg_lines[0] if msg_lines else "",
            location=_user_location(fail.text or ""),
        ))
    return passed, failures


def cluster_failures(failures: list[Failure]) -> list[Cluster]:
    """Group failures by (exception type, normalized message, location), biggest first."""
    clusters: dict[tuple[str, str, str], Cluster] = {}
    for f in failures:
        key = (f.exc_type, normalize(f.message), f.location)
        if key not in clusters:
            clusters[key] = Cluster(signature=key[1], exc_type=key[0], location=key[2])
        clusters[key].failures.append(f)
    return sorted(clusters.values(), key=lambda c: len(c.failures), reverse=True)


def format_clusters(clusters: list[Cluster], n_examples: int = 4) -> str:
    """Human/LLM-readable summary of the clusters."""
    if not clusters:
        return "No failures."
    total = sum(len(c.failures) for c in clusters)
    lines = [f"{total} failures collapsed into {len(clusters)} cluster(s):"]
    for i, c in enumerate(clusters, 1):
        lines += [
            "",
            f"Cluster {i}: {len(c.failures)} failures  [{c.exc_type} at {c.location}]",
            f"  signature: {c.signature}",
            f"  tests:     {', '.join(c.tests)}",
            f"  seeds:     {', '.join(map(str, c.seeds))}",
            "  examples:",
        ]
        seen = set()
        for f in c.failures:
            if f.message not in seen:
                seen.add(f.message)
                lines.append(f"    [{f.test}, seed {f.seed}] {f.message}")
            if len(seen) >= n_examples:
                break
    return "\n".join(lines)
