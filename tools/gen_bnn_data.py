#!/usr/bin/env python3
"""
gen_bnn_data.py - turn BNN.pkl into memory files the Verilog reads.

Outputs two hex files (one 16-bit word per line):

  weights.mem : 160 words = 10 classes x 16 cycles, address = class*16 + cycle
  images.mem  : 160 words = 10 images  x 16 chunks, PRE-TRANSPOSED so the
                Verilog can index image chunks straight by cycle.

The pre-transpose is the whole trick: in the original the image lived across 16
LUTs and the hardware grabbed one bit per LUT per cycle. We do that gather here
in Python instead, so chunk c already holds the 16 bits the EXE stage needs.
"""
import sys, os, pickle, warnings, numpy as np
warnings.filterwarnings("ignore")

path   = sys.argv[1] if len(sys.argv) > 1 else "model/BNN.pkl"
outdir = sys.argv[2] if len(sys.argv) > 2 else "."
d = pickle.load(open(path, "rb"))
Wi, Xi = d["weights_int16"], d["imgs_int16"]
MASK = 0xFFFF

def image_chunk(m, c):
    val = 0
    for i in range(16):
        val |= (((int(Xi[m][i]) >> c) & 1) << i)
    return val & MASK

with open(os.path.join(outdir, "weights.mem"), "w") as f:
    for k in range(10):
        for c in range(16):
            f.write(f"{int(Wi[k][c]) & MASK:04x}\n")

with open(os.path.join(outdir, "images.mem"), "w") as f:
    for m in range(10):
        for c in range(16):
            f.write(f"{image_chunk(m, c):04x}\n")

# expected predictions, for the testbench to check against
def predict(m):
    best, idx = -1, 0
    for k in range(10):
        acc = 0
        for c in range(16):
            xnor = (~(int(Wi[k][c]) ^ image_chunk(m, c))) & MASK
            acc += bin(xnor).count("1")
        if acc > best:
            best, idx = acc, k
    return idx

preds = [predict(m) for m in range(10)]
print("wrote weights.mem (160 words) and images.mem (160 words)")
print("expected predictions:", preds)
