# sim-agent — An LLM Agent for LTspice

A LangGraph agent system that drives real SPICE simulations. You describe what you
want in plain English; the agent picks the right specialist, writes or reads the
netlist, shells out to LTspice, parses the binary results, and explains them.

It does three things:

| | Capability | Example prompt |
|---|---|---|
| **L1** | Run & interpret an existing netlist | *"Run rc_lowpass.cir"* |
| **L2** | Design a new circuit from a spec, then verify it by simulation | *"Build a 5 kHz RC low-pass"* |
| **L3** | Analyze an unknown netlist that has no analysis directive | *"What does mystery_ladder.cir do?"* |

## Architecture

A root orchestrator exposes each specialist subagent as a tool. It never touches
LTspice itself — it only decides who should handle the request.

```
                      ┌──────────────────┐
   user (natural  ──► │   root_agent     │  routes, never simulates
   language)          └────────┬─────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
  ┌───────────┐         ┌────────────┐         ┌────────────┐
  │ agent_l1  │         │  agent_l2  │         │  agent_l3  │
  │ simulate  │         │  generate  │         │  analyze   │
  └─────┬─────┘         └──────┬─────┘         └──────┬─────┘
        │                      │                      │
        │              netlist_builder         netlist_parser
        │              (pure math →            (topology →
        │               SPICE text)             analysis choice)
        │                      │                      │
        └──────────────────────┴──────────────────────┘
                               ▼
                    ltspice_runner (subprocess -b)
                               ▼
                  log_parser  +  raw_parser
```

Every agent is the same compiled `StateGraph`: an LLM node bound to tools, a
conditional edge back into a `ToolNode`, looping until the model stops calling tools.

**Design note:** no LLM ever computes a component value. The model's job is to
extract structured parameters (`f_c = 5000`) and choose a topology; all arithmetic
lives in `generators/netlist_builder.py` as ordinary Python. Same idea in L3 — the
model reasons about *which* analysis fits, but the topology facts it reasons from
come from a deterministic parser.

## How each level works

### Level 1 — Simulation runner
Tools: `list_circuits`, `run_ltspice_simulation`, `read_netlist`.
Invokes `LTspice -b file.cir`, then parses both outputs: the `.log` for errors,
warnings and operating-point values, the binary `.raw` for waveform data and
derived metrics (−3 dB cutoff, DC gain, min/max/final values).

### Level 2 — Spec → circuit
Tools: `generate_netlist`, plus L1's simulation tool for verification.
Supported topologies: voltage divider, RC low/high-pass, RL low/high-pass,
RLC bandpass/bandstop. The agent picks the topology, the builder solves for
component values using standard formulas (`f_c = 1/2πRC`, `Q = (1/R)√(L/C)`, …),
formats them with SPICE suffixes, and picks an AC sweep range that brackets the
frequency of interest. Then it simulates its own output and reports designed vs.
measured.

### Level 3 — Unknown circuit analyzer
Tools: `read_netlist_topology`, `prepare_and_run_analysis`.
This is the interesting one, because the reasoning has to happen *before* the
simulation. Given a bare netlist with no `.op`/`.ac`/`.tran`, the parser reports
what components exist and whether any are reactive or time-varying. The agent uses
that to choose an analysis, the tool injects the directive into a copy
(`*_analyzed.cir`), runs it, and the agent explains the result in terms of what
kind of circuit it turned out to be.

## Setup

**1. Install LTspice** — [download](https://www.analog.com/en/resources/design-tools-and-calculators/ltspice-simulator.html).
The default macOS path (`/Applications/LTspice.app`) is assumed; override with
`LTSPICE_BINARY` if yours differs.

Verify the CLI works:
```bash
/Applications/LTspice.app/Contents/MacOS/LTspice -b circuits/rc_lowpass.cir
```
That should produce `circuits/rc_lowpass.log` and `circuits/rc_lowpass.raw`.

**2. Install dependencies**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Configure**
```bash
cp .env.example .env
# add your OPENAI_API_KEY
```

**4. Run**
```bash
python main.py
```

There's also a smoke test that exercises the structure and parsers without needing
an API key: `python tests/smoke_test.py`.

## Example session

```
You: build me an RC high-pass with a 500Hz cutoff

Agent: Designed a first-order RC high-pass. Picking C = 100nF and solving
       f_c = 1/(2πRC) gives R = 3.183k.
       Saved to highpass_500hz.cir and simulated it:
       - measured -3dB point: 501 Hz (0.2% off spec)
       - gain at 10 Hz: -33.8 dB, at 10 kHz: -0.04 dB
       Matches the spec.

You: what does mystery_ladder.cir do?

Agent: Topology: one 5V DC source, five resistors, no reactive components
       and no time-varying sources — so a DC operating point is the right
       analysis. Injected .op and ran it.
       This is a loaded resistive ladder: R1 feeds a series chain
       (R2→R3→R4) to ground, with R5 shunting the R1/R2 junction...
```

## Project layout

```
sim-agent/
├── main.py                     # interactive CLI
├── config.py                   # paths, timeouts, model selection
├── state.py                    # AgentState (message history + reducer)
├── agents/
│   ├── root_agent.py           # orchestrator — routes to the three below
│   ├── agent_l1.py             # simulation runner
│   ├── agent_l2.py             # spec → circuit generator
│   └── agent_l3.py             # unknown-circuit analyzer
├── generators/
│   └── netlist_builder.py      # deterministic math → SPICE netlist text
├── parsers/
│   ├── log_parser.py           # .log (UTF-16 LE) → errors, warnings, .op values
│   ├── raw_parser.py           # binary .raw → waveforms + derived metrics
│   └── netlist_parser.py       # topology extraction, directive stripping
├── tools/
│   └── ltspice_runner.py       # subprocess wrapper around the LTspice CLI
├── circuits/                   # sample netlists (generated ones land here too)
└── tests/smoke_test.py
```

## Things that turned out to be non-obvious

A few LTspice-on-macOS details that cost real time:

- **`.log` files are UTF-16 LE**, not UTF-8. Reading them naively gives you a wall
  of null bytes. `log_parser.py` sniffs and decodes accordingly.
- **Operating points in the log have no delimiter after the value** —
  `V(out): 2.5   voltage` — so the parser can't just split on whitespace pairs.
- **The `ltspice` package cannot parse DC `.op` raw files** and raises `ValueError`.
  That's fine and expected: for a `.op` run the log already carries every node
  voltage, so the exception is caught and the log data is used instead.
- **SPICE silently discards the first line as the title.** A netlist starting
  directly with a component loses that component. L3 prepends a title comment
  before injecting directives.
- Use `raw.get_frequency()` / `raw.get_time()`; `get_data("frequency")` doesn't work.

## Roadmap

Levels 1–3 are complete. Sketched but not built:

- **L4 — Parameter sweeps:** run N simulations over a component range, present the
  trade-off table.
- **L5 — Topology comparison:** generate competing designs for one spec, simulate
  both, recommend one.

## Stack

Python 3.11+ · LangGraph · LangChain · OpenAI (`gpt-4o-mini` by default, swappable)
· LTspice · `ltspice` + NumPy for binary parsing
