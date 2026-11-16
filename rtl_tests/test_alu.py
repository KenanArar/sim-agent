"""
cocotb tests for rtl/alu.sv, checked against an independent Python reference model.

Don't run this file directly. tools/rtl_runner.run_cocotb_regression runs it
under Icarus Verilog once per random seed and collects the results.
"""

import random

import cocotb
from cocotb.triggers import Timer

OPS = {"ADD": 0, "SUB": 1, "AND": 2, "OR": 3, "XOR": 4, "SLT": 5, "SLL": 6, "SRL": 7}


def _signed8(x: int) -> int:
    return x - 256 if x & 0x80 else x


def alu_model(op: str, a: int, b: int) -> int:
    """Reference model written from the spec, not from the RTL."""
    if op == "ADD":
        return (a + b) & 0xFF
    if op == "SUB":
        return (a - b) & 0xFF
    if op == "AND":
        return a & b
    if op == "OR":
        return a | b
    if op == "XOR":
        return a ^ b
    if op == "SLT":
        return int(_signed8(a) < _signed8(b))
    if op == "SLL":
        return (a << (b & 7)) & 0xFF
    if op == "SRL":
        return a >> (b & 7)
    raise ValueError(op)


async def check(dut, op: str, a: int, b: int) -> None:
    """Drive one vector and compare against the model. Every test goes through
    here, so the same bug produces the same message from different tests."""
    dut.a.value = a
    dut.b.value = b
    dut.op.value = OPS[op]
    await Timer(1, unit="ns")

    got = dut.y.value.to_unsigned()
    exp = alu_model(op, a, b)
    assert got == exp, f"{op} mismatch: a=0x{a:02x} b=0x{b:02x} got=0x{got:02x} exp=0x{exp:02x}"

    zero = int(dut.zero.value)
    assert zero == int(exp == 0), f"zero flag wrong for {op}: y=0x{got:02x} zero={zero}"


@cocotb.test()
async def test_directed_logic_ops(dut):
    """Hand-picked vectors for ADD and the bitwise ops."""
    for op, a, b in [("ADD", 1, 2), ("ADD", 0xFF, 1), ("AND", 0xF0, 0x3C),
                     ("OR", 0xF0, 0x0F), ("XOR", 0xAA, 0xFF)]:
        await check(dut, op, a, b)


@cocotb.test()
async def test_sub_borrow_corners(dut):
    """SUB at the borrow boundaries."""
    for a, b in [(5, 3), (0, 1), (0x80, 0x01), (0xFF, 0xFF)]:
        await check(dut, "SUB", a, b)


@cocotb.test()
async def test_signed_compare_corners(dut):
    """SLT across the sign boundary."""
    for a, b in [(0x01, 0x02), (0x80, 0x01), (0xFF, 0x00), (0x7F, 0x80)]:
        await check(dut, "SLT", a, b)


@cocotb.test()
async def test_shift_corners(dut):
    """Shift amounts 0 and 7."""
    for op in ("SLL", "SRL"):
        for a, b in [(0x81, 0), (0x81, 7), (0x01, 3)]:
            await check(dut, op, a, b)


@cocotb.parametrize(op=list(OPS))
async def test_random_op(dut, op):
    """100 constrained-random vectors per opcode. cocotb seeds `random` from
    the regression seed, so each seed sees different vectors."""
    for _ in range(100):
        await check(dut, op, random.randrange(256), random.randrange(256))
