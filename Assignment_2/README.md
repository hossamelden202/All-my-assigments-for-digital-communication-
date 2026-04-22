# BER Simulation — ELC325B Assignment 2

## Requirements

Python 3.8+ with the following packages:

```
numpy
matplotlib
scipy
```

Install them with:

```bash
pip install numpy matplotlib scipy
```

---

## How to Run

```bash
python3 ber_simulation.py
```

---

## What Happens

- Progress prints to the terminal for each E/N₀ value (−10 dB to 20 dB), showing the number of bits simulated and error counts for all three cases.
- Two plot windows open when the simulation finishes.
- Two image files are saved in the same folder:
  - `filter_outputs.png` receive filter output waveforms at E/N₀ = 10 dB
  - `ber_vs_eno.png`  BER vs E/N₀ (theory + simulation) for all three cases


