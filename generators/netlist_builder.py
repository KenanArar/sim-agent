"""
Deterministic SPICE netlist builders for supported circuit types.

Each function takes numeric parameters (already extracted by the LLM),
computes component values using standard formulas, and returns a valid
LTspice .cir netlist string.

No LLM involvement here — pure math and string templates.
"""

import math
from pathlib import Path

from config import CIRCUITS_DIR


# ── Value formatter ──────────────────────────────────────────────────

def _fmt_r(v: float) -> str:
    """Format a resistance value with SPICE suffix."""
    if v >= 1e6:
        return f"{v/1e6:.4g}Meg"
    elif v >= 1e3:
        return f"{v/1e3:.4g}k"
    return f"{v:.4g}"


def _fmt_c(v: float) -> str:
    """Format a capacitance value with SPICE suffix."""
    if v >= 1e-6:
        return f"{v/1e-6:.4g}u"
    elif v >= 1e-9:
        return f"{v/1e-9:.4g}n"
    elif v >= 1e-12:
        return f"{v/1e-12:.4g}p"
    return f"{v:.4g}"


def _fmt_l(v: float) -> str:
    """Format an inductance value with SPICE suffix."""
    if v >= 1:
        return f"{v:.4g}"
    elif v >= 1e-3:
        return f"{v/1e-3:.4g}m"
    elif v >= 1e-6:
        return f"{v/1e-6:.4g}u"
    return f"{v:.4g}"


def _fmt_freq(f: float) -> str:
    """Human-readable frequency string for comments."""
    if f >= 1e6:
        return f"{f/1e6:.4g}MHz"
    elif f >= 1e3:
        return f"{f/1e3:.4g}kHz"
    return f"{f:.4g}Hz"


def _spice_freq(f: float) -> str:
    """Format a frequency as a plain SPICE value (no Hz unit, uses k/Meg suffix)."""
    if f >= 1e6:
        return f"{f/1e6:.4g}Meg"
    elif f >= 1e3:
        return f"{f/1e3:.4g}k"
    return f"{f:.4g}"


def _ac_range(f_center: float) -> tuple[str, str]:
    """Return AC sweep start/stop (as SPICE strings) bracketing the frequency of interest."""
    f_start = max(1.0, f_center / 1000)
    f_stop = min(1e8, f_center * 1000)
    return _spice_freq(f_start), _spice_freq(f_stop)


# ── Circuit builders ─────────────────────────────────────────────────

def voltage_divider(vin: float, vout: float) -> str:
    """
    Resistive voltage divider.

    Chooses R_total = 20kΩ and splits to achieve the desired ratio.
    Vout = Vin * R2 / (R1 + R2)
    """
    if vout >= vin or vout <= 0:
        raise ValueError(f"vout ({vout}V) must be between 0 and vin ({vin}V)")
    ratio = vout / vin
    r_total = 20e3
    r2 = ratio * r_total
    r1 = r_total - r2
    return (
        f"* Voltage Divider\n"
        f"* Vout = {vout}V from Vin = {vin}V  (ratio {ratio:.4g})\n"
        f"* R1={_fmt_r(r1)}, R2={_fmt_r(r2)}\n"
        f"\n"
        f"V1 in 0 DC {vin}\n"
        f"R1 in out {_fmt_r(r1)}\n"
        f"R2 out 0 {_fmt_r(r2)}\n"
        f"\n"
        f".op\n"
        f".end\n"
    )


def rc_lowpass(f_c: float) -> str:
    """
    First-order RC low-pass filter.

    f_c = 1 / (2π·R·C)
    Fixes C = 100nF, computes R.
    """
    c = 100e-9
    r = 1 / (2 * math.pi * f_c * c)
    f_start, f_stop = _ac_range(f_c)
    return (
        f"* RC Low-Pass Filter\n"
        f"* f_c = {_fmt_freq(f_c)}  (R={_fmt_r(r)}, C={_fmt_c(c)})\n"
        f"\n"
        f"V1 in 0 AC 1\n"
        f"R1 in out {_fmt_r(r)}\n"
        f"C1 out 0 {_fmt_c(c)}\n"
        f"\n"
        f".ac dec 100 {f_start} {f_stop}\n"
        f".end\n"
    )


def rc_highpass(f_c: float) -> str:
    """
    First-order RC high-pass filter.

    f_c = 1 / (2π·R·C)
    Fixes C = 100nF, computes R.
    """
    c = 100e-9
    r = 1 / (2 * math.pi * f_c * c)
    f_start, f_stop = _ac_range(f_c)
    return (
        f"* RC High-Pass Filter\n"
        f"* f_c = {_fmt_freq(f_c)}  (C={_fmt_c(c)}, R={_fmt_r(r)})\n"
        f"\n"
        f"V1 in 0 AC 1\n"
        f"C1 in out {_fmt_c(c)}\n"
        f"R1 out 0 {_fmt_r(r)}\n"
        f"\n"
        f".ac dec 100 {f_start} {f_stop}\n"
        f".end\n"
    )


def rl_lowpass(f_c: float) -> str:
    """
    First-order RL low-pass filter.

    f_c = R / (2π·L)  →  output across R (inductor in series).
    Fixes L = 10mH, computes R.
    """
    l = 10e-3
    r = 2 * math.pi * f_c * l
    f_start, f_stop = _ac_range(f_c)
    return (
        f"* RL Low-Pass Filter\n"
        f"* f_c = {_fmt_freq(f_c)}  (L={_fmt_l(l)}, R={_fmt_r(r)})\n"
        f"\n"
        f"V1 in 0 AC 1\n"
        f"L1 in n1 {_fmt_l(l)}\n"
        f"R1 n1 0 {_fmt_r(r)}\n"
        f"\n"
        f".ac dec 100 {f_start} {f_stop}\n"
        f".end\n"
    )


def rl_highpass(f_c: float) -> str:
    """
    First-order RL high-pass filter.

    f_c = R / (2π·L)  →  output across L (resistor in series).
    Fixes L = 10mH, computes R.
    """
    l = 10e-3
    r = 2 * math.pi * f_c * l
    f_start, f_stop = _ac_range(f_c)
    return (
        f"* RL High-Pass Filter\n"
        f"* f_c = {_fmt_freq(f_c)}  (R={_fmt_r(r)}, L={_fmt_l(l)})\n"
        f"\n"
        f"V1 in 0 AC 1\n"
        f"R1 in n1 {_fmt_r(r)}\n"
        f"L1 n1 0 {_fmt_l(l)}\n"
        f"\n"
        f".ac dec 100 {f_start} {f_stop}\n"
        f".end\n"
    )


def rlc_bandpass(f_0: float, Q: float) -> str:
    """
    Series RLC band-pass filter. Output is across R.

    f_0 = 1 / (2π·√(LC))
    Q   = (1/R)·√(L/C)  →  R = √(L/C) / Q

    Fixes L = 10mH, computes C and R.
    """
    l = 10e-3
    c = 1 / ((2 * math.pi * f_0) ** 2 * l)
    r = math.sqrt(l / c) / Q
    f_start, f_stop = _ac_range(f_0)
    return (
        f"* RLC Band-Pass Filter\n"
        f"* f_0 = {_fmt_freq(f_0)}, Q = {Q:.4g}\n"
        f"* L={_fmt_l(l)}, C={_fmt_c(c)}, R={_fmt_r(r)}\n"
        f"\n"
        f"V1 in 0 AC 1\n"
        f"L1 in n1 {_fmt_l(l)}\n"
        f"C1 n1 n2 {_fmt_c(c)}\n"
        f"R1 n2 0 {_fmt_r(r)}\n"
        f"\n"
        f".ac dec 100 {f_start} {f_stop}\n"
        f".end\n"
    )


def rlc_bandstop(f_0: float, Q: float) -> str:
    """
    RLC band-stop (notch) filter. Series LC shunt to ground.

    A series R feeds the output node. A series L-C tank is connected
    from the output node to ground. At resonance the tank is a short
    to ground → output is nulled (notch).

    f_0 = 1 / (2π·√(LC))
    Q   = R / (ω_0·L)  →  R = Q·ω_0·L

    Fixes L = 10mH, computes C and R.
    """
    l = 10e-3
    c = 1 / ((2 * math.pi * f_0) ** 2 * l)
    r = Q * 2 * math.pi * f_0 * l
    f_start, f_stop = _ac_range(f_0)
    return (
        f"* RLC Band-Stop (Notch) Filter\n"
        f"* f_0 = {_fmt_freq(f_0)}, Q = {Q:.4g}\n"
        f"* R={_fmt_r(r)}, L={_fmt_l(l)}, C={_fmt_c(c)}\n"
        f"\n"
        f"V1 in 0 AC 1\n"
        f"R1 in out {_fmt_r(r)}\n"
        f"L1 out n1 {_fmt_l(l)}\n"
        f"C1 n1 0 {_fmt_c(c)}\n"
        f"\n"
        f".ac dec 100 {f_start} {f_stop}\n"
        f".end\n"
    )


# ── Save helper ──────────────────────────────────────────────────────

def save_netlist(cir_text: str, filename: str) -> Path:
    """
    Write a netlist string to circuits/<filename>.

    Args:
        cir_text:  Full .cir file content.
        filename:  Filename including .cir extension.

    Returns:
        Resolved path to the saved file.
    """
    path = CIRCUITS_DIR / filename
    path.write_text(cir_text)
    return path.resolve()
