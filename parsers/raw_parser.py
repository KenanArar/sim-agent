"""
Parser for LTspice .raw binary files.

Extracts waveform data (voltages, currents vs frequency/time)
and computes key metrics like -3dB cutoff frequencies.

Uses the `ltspice` pip package for binary parsing.
"""

import numpy as np
import ltspice
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class WaveformData:
    """Structured data extracted from an LTspice .raw file."""
    variables: list[str] = field(default_factory=list)
    case_count: int = 0
    point_count: int = 0
    analysis_type: str | None = None  # "AC", "Transient", "DC", etc.
    measurements: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary of the waveform data."""
        parts = []
        if self.analysis_type:
            parts.append(f"Analysis type: {self.analysis_type}")
        parts.append(f"Variables: {', '.join(self.variables)}")
        parts.append(f"Data points: {self.point_count}")
        if self.measurements:
            parts.append("Measurements:")
            for name, val in self.measurements.items():
                parts.append(f"  {name}: {val}")
        return "\n".join(parts)


def _detect_analysis_type(raw) -> str:
    """Infer the analysis type from the independent variable."""
    # The first variable is the sweep variable
    var0 = raw.variables[0].lower() if raw.variables else ""
    if "frequency" in var0:
        return "AC"
    elif "time" in var0:
        return "Transient"
    else:
        return "DC"


def _compute_ac_metrics(raw) -> dict[str, str]:
    """
    For AC analysis, compute DC gain, -3dB cutoff, and rolloff for every
    voltage node in the simulation. No hardcoded node names.
    """
    metrics = {}

    freq = raw.get_frequency()
    if freq is None or len(freq) == 0:
        return metrics
    freq = np.abs(freq)

    voltage_vars = [v for v in raw.variables if v.upper().startswith("V(")]
    for var in voltage_vars:
        data = raw.get_data(var)
        if data is None or len(data) == 0:
            continue

        magnitude_db = 20 * np.log10(np.abs(data) + 1e-15)
        dc_gain = magnitude_db[0]
        metrics[f"{var} DC Gain"] = f"{dc_gain:.2f} dB"

        target = dc_gain - 3.0
        crossings = np.where(np.diff(np.sign(magnitude_db - target)))[0]
        if len(crossings) > 0:
            idx = crossings[0]
            if idx + 1 < len(freq):
                f1, f2 = freq[idx], freq[idx + 1]
                m1, m2 = magnitude_db[idx], magnitude_db[idx + 1]
                f_3db = f1 + (target - m1) * (f2 - f1) / (m2 - m1) if m2 != m1 else f1
                metrics[f"{var} -3dB Cutoff"] = _format_freq(f_3db)

        metrics[f"{var} Gain at {_format_freq(freq[-1])}"] = f"{magnitude_db[-1]:.2f} dB"

    return metrics


def _compute_dc_metrics(raw) -> dict[str, str]:
    """For DC operating point, return all node voltages and branch currents."""
    metrics = {}
    for var in raw.variables:
        data = raw.get_data(var)
        if data is not None and len(data) > 0:
            val = np.real(data[-1])  # Take last (or only) value
            if var.startswith("V("):
                metrics[var] = f"{val:.4g} V"
            elif var.startswith("I("):
                metrics[var] = f"{val:.4g} A"
    return metrics


def _compute_transient_metrics(raw) -> dict[str, str]:
    """For transient analysis, return min/max/final values of each variable."""
    metrics = {}
    time = raw.get_time()
    if time is not None and len(time) > 0:
        metrics["Duration"] = f"{time[-1]:.4g} s"
    for var in raw.variables:
        data = raw.get_data(var)
        if data is not None and len(data) > 0:
            real_data = np.real(data)
            metrics[f"{var} (final)"] = f"{real_data[-1]:.4g}"
            metrics[f"{var} (min)"] = f"{np.min(real_data):.4g}"
            metrics[f"{var} (max)"] = f"{np.max(real_data):.4g}"
    return metrics


def _format_freq(f: float) -> str:
    """Format a frequency value with appropriate unit prefix."""
    if f >= 1e9:
        return f"{f/1e9:.2f} GHz"
    elif f >= 1e6:
        return f"{f/1e6:.2f} MHz"
    elif f >= 1e3:
        return f"{f/1e3:.2f} kHz"
    else:
        return f"{f:.2f} Hz"


def parse_raw(raw_path: str | Path) -> WaveformData:
    """
    Parse an LTspice .raw file into structured waveform data.

    Args:
        raw_path: Path to the .raw file.

    Returns:
        WaveformData with variables, metrics, and analysis info.
    """
    raw_path = Path(raw_path)
    raw = ltspice.Ltspice(str(raw_path))
    raw.parse()

    data = WaveformData()
    data.variables = list(raw.variables)
    data.case_count = raw.case_count
    first_data = raw.get_data(raw.variables[0]) if raw.variables else None
    data.point_count = len(first_data) if first_data is not None else 0
    data.analysis_type = _detect_analysis_type(raw)

    # Compute analysis-specific metrics
    if data.analysis_type == "AC":
        data.measurements = _compute_ac_metrics(raw)
    elif data.analysis_type == "Transient":
        data.measurements = _compute_transient_metrics(raw)
    else:
        data.measurements = _compute_dc_metrics(raw)

    return data
