# bnn-goboard

A tiny **binary neural network** that recognizes handwritten digits **entirely
in FPGA hardware** — no CPU, no software inference. It runs on the
[Nandland Go Board](https://nandland.com/the-go-board/) (Lattice iCE40-HX1K) and
is built with a fully open toolchain (apio: yosys + nextpnr + icestorm).

This is a modern port of MIT Han Lab's
[`bnn-icestick`](https://github.com/mit-han-lab/bnn-icestick). The original
trained in TensorFlow 1.x and described the hardware in Magma (a Python HDL)
targeting the Lattice iCEstick. This port keeps the same network and weights but
rewrites the design as **plain, readable Verilog**, swaps the build flow to
**apio**, retargets the **Go Board**, and (planned) moves training to
**PyTorch**.

> **Result:** 9 of 10 test digits classify correctly; digit **4** is read as
> **9** — identical to the original. Both a Python model and a Verilog testbench
> verify this on every commit.

## What it does

You give it a 16×16 image of a handwritten digit; the FPGA decides which digit
0–9 it is and shows the answer on the LEDs and the 7-segment display. The
"network" is a single fully-connected layer where every weight and pixel is one
bit, so a neuron is just **XNOR + popcount**. See
[`docs/pipeline.md`](docs/pipeline.md) for the full explanation.

## Project structure

```
bnn-goboard/
├── README.md
├── LICENSE                      MIT (original authors + this port)
├── requirements.txt
├── .gitignore
│
├── model/
│   └── BNN.pkl                  pre-trained binary weights (from the original)
│
├── tools/
│   ├── golden_reference.py      Stage 1: pure-Python model of the hardware
│   └── gen_bnn_data.py          turns BNN.pkl into weights.mem + images.mem
│
├── training/
│   └── train_bnn.py             Stage opt: PyTorch training (planned stub)
│
├── fpga/                        the apio project (Stages 2-3)
│   ├── apio.ini                 board = go-board
│   ├── top.v                    wrapper: bakes in an image, drives LEDs + 7-seg
│   ├── bnn_classifier.v         the classifier (XNOR/popcount/argmax FSM)
│   ├── seg7.v                   digit -> 7-segment decoder
│   ├── bnn_tb.v                 testbench: checks all 10 images
│   ├── weights.mem             generated weight ROM contents
│   ├── images.mem              generated, pre-transposed images
│   └── goboard.pcf             Stage 3: Go Board pin map (placeholder)
│
├── docs/
│   ├── pipeline.md              how the BNN and the 5-stage pipeline work
│   └── images/bmv.png
│
└── .github/workflows/sim.yml    CI: runs the testbench on every push
```

## Quickstart

### 1. Prerequisites

```bash
pip install -r requirements.txt        # numpy
sudo apt install iverilog              # simulation
pip install apio && apio install -a    # FPGA toolchain (yosys/nextpnr/icestorm)
```

### 2. Generate the memory files from the weights

```bash
python3 tools/gen_bnn_data.py model/BNN.pkl fpga/
```

### 3. Check the software model (your "known correct answer")

```bash
python3 tools/golden_reference.py model/BNN.pkl
```

### 4. Simulate the hardware and verify it matches

```bash
cd fpga
iverilog -o bnn_sim bnn_classifier.v bnn_tb.v && vvp bnn_sim
# expect: PASS: all 10 predictions match golden_reference.py
```

### 5. Build and flash (after Stage 3 fills in goboard.pcf)

```bash
cd fpga
apio build      # synth + place & route -> bitstream
apio upload     # flash the Go Board
```

## Roadmap

- [x] **Stage 1 — Software golden reference.** Decode `BNN.pkl`, reproduce the
      classification in Python (9/10, digit 4 → 9).
- [x] **Stage 2 — Verilog datapath + simulation.** Synthesizable classifier,
      testbench passes against the golden reference, fits ~435/1280 LUTs.
- [ ] **Stage 3 — Go Board bring-up.** Pin constraints, 7-segment polarity,
      `apio build` + `apio upload`, see a real digit light up.
- [ ] **Optional — PyTorch training.** Replace the TensorFlow 1.x training so
      weights can be retrained/improved and re-exported to `BNN.pkl`.

## Hardware notes

The Go Board and the original iCEstick use the **same FPGA chip** (iCE40-HX1K),
so the logic fits unchanged. Differences this port handles: the Go Board runs at
25 MHz (vs 12), has 4 LEDs plus a dual 7-segment display (vs 5 LEDs), uses the
VQ100 package, and has no PLL (unused here anyway).

## Credits & license

Based on [`mit-han-lab/bnn-icestick`](https://github.com/mit-han-lab/bnn-icestick)
by Yujun Lin, Kaidi Cao, and Song Han. Original weights, the `bmv.png` figure,
and the network design are theirs. Released under the MIT License — see
[`LICENSE`](LICENSE).
