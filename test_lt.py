import subprocess
import ltspice
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

LTSPICE_BIN = "/Applications/LTspice.app/Contents/MacOS/LTspice"
CIR_FILE = "lowpass.cir"
RAW_FILE = "/Users/kenanarar/Documents/LTspice/lowpass.raw"
LOG_FILE = "/Users/kenanarar/Documents/LTspice/lowpass.log"

# --------------------------------------------------
# 1. Run LTspice in batch mode
# --------------------------------------------------
result = subprocess.run(
    [LTSPICE_BIN, "-b", CIR_FILE],
    capture_output=True,
    text=True
)

# --------------------------------------------------
# 2. Check log for errors
# --------------------------------------------------
with open(LOG_FILE, "r", errors="ignore") as f:
    log_text = f.read()

if "Fatal Error" in log_text or "Error" in log_text:
    print("❌ LTspice simulation failed")
    print(log_text)
    sys.exit(1)

print("✅ LTspice simulation completed successfully")

# --------------------------------------------------
# 3. Parse the .raw file
# --------------------------------------------------
l = ltspice.Ltspice(RAW_FILE)
l.parse()

# --------------------------------------------------
# 4. Extract AC analysis data
# --------------------------------------------------
freq = l.get_frequency()          # Frequency axis
vout = l.get_data("V(out)")       # Complex voltage

# Magnitude in dB
mag_db = 20 * np.log10(np.abs(vout))

# --------------------------------------------------
# 5. Compute useful metrics (agent-friendly)
# --------------------------------------------------
dc_gain = mag_db[0]

# Find -3 dB cutoff
idx = np.where(mag_db <= dc_gain - 3)[0][0]
cutoff_freq = freq[idx]

print(f"DC gain: {dc_gain:.2f} dB")
print(f"Cutoff frequency: {cutoff_freq:.1f} Hz")

# --------------------------------------------------
# 6. Return structured result (for agent use)
# --------------------------------------------------
result = {
    "dc_gain_db": float(dc_gain),
    "cutoff_hz": float(cutoff_freq)
}

print("\nResult object:")
print(result)

# Frequency axis (AC analysis)
freq = l.get_frequency()

# Output voltage at node "out"
Vout = l.get_data("V(out)")

# Convert to magnitude (dB)
Vout_mag_db = 20 * np.log10(np.abs(Vout))
plt.figure()
plt.semilogx(freq, Vout_mag_db)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("RC Low-Pass Filter Frequency Response")
plt.grid(True, which="both")
plt.show()
