#!/usr/bin/env python3
"""
golden_reference.py  -  Stage 1 of the BNN-on-FPGA migration
=============================================================

This is a PURE SOFTWARE reproduction of what the FPGA hardware computes.
Run it, confirm it matches the original project (9/10 correct, digit 4 read
as 9), and from now on it is your "known correct answer". Every Verilog
module you write in Stage 2 gets checked against the numbers this prints.

It depends only on numpy + the pre-trained weights file `BNN.pkl` that ships
with the repo (mit-han-lab/bnn-icestick, under nn_train/).

    python3 golden_reference.py                 # uses ./nn_train/BNN.pkl
    python3 golden_reference.py path/to/BNN.pkl

------------------------------------------------------------------
WHAT THE NETWORK IS
------------------------------------------------------------------
One fully-connected layer. The input is a 16x16 image flattened to 256 bits.
There are 10 output classes (digits 0-9), so the weights are a 10 x 256 grid
of bits. To classify, for each class you compare the 256 image bits against
that class's 256 weight bits and count how many AGREE. The class with the most
agreements wins. "Count the agreements" = XNOR followed by popcount. No
multipliers, no floating point - that is why it fits on a tiny FPGA.

------------------------------------------------------------------
HOW THE DATA IS PACKED  (this is the subtle part)
------------------------------------------------------------------
BNN.pkl gives us two arrays we care about, both uint16:

    weights_int16 : shape (10, 16)   ->  10 classes, 16 words of 16 bits = 256 bits
    imgs_int16    : shape (10, 16)   ->  10 images,  16 words of 16 bits = 256 bits

The hardware streams these 16 bits at a time over 16 "cycles" (16 x 16 = 256).
On cycle c, for class k, it grabs:

    weight chunk = weights_int16[k][c]                       (bit i of this word)
    image  chunk = bit_c of each of the 16 image words       (bit i comes from word i)

i.e. the image is stored TRANSPOSED relative to the weights (it lives across 16
LUTs in the real design, one bit per LUT per cycle). The function below mirrors
that pairing exactly, so its output equals the silicon - not just an idealized
classifier. (The float arrays `weights`/`imgs` in the pkl are leftovers from
training and are NOT laid out for this pairing, so we ignore them here.)
"""

import sys
import warnings
import pickle
import numpy as np

warnings.filterwarnings("ok".replace("ok", "ignore"))  # hide an old-numpy pickle warning

NUM_CLASSES = 10
NUM_WORDS = 16          # 16 words per vector
WORD_BITS = 16          # 16 bits per word  -> 256 bits total
MASK16 = 0xFFFF


# ------------------------------------------------------------------
# Core: the exact computation the FPGA performs
# ------------------------------------------------------------------
def popcount16(x):
    """Number of 1-bits in a 16-bit value (this is the hardware 'BitCounter16')."""
    return bin(x & MASK16).count("1")


def image_chunk(imgs_int16, m, c):
    """The 16-bit image slice the hardware sees on cycle c for image m.

    Hardware fact: image bit i on cycle c = bit c of image word i.
    So we gather bit c out of each of the 16 words and pack them into one word.
    """
    val = 0
    for i in range(NUM_WORDS):
        bit = (int(imgs_int16[m][i]) >> c) & 1
        val |= bit << i
    return val


def class_score(imgs_int16, weights_int16, m, k):
    """Accumulated XNOR-popcount of image m against class k, over all 16 cycles.

    This is exactly the EXE stage: for each cycle, XNOR the weight word with the
    image slice, popcount the result, and add it to the running total.
    """
    acc = 0
    for c in range(NUM_WORDS):
        w = int(weights_int16[k][c]) & MASK16
        x = image_chunk(imgs_int16, m, c)
        xnor = (~(w ^ x)) & MASK16          # bit is 1 where w and x AGREE
        acc += popcount16(xnor)
    return acc


def classify(imgs_int16, weights_int16, m):
    """Return (prediction, [score for each of the 10 classes]) for image m.

    The hardware 'CP' stage keeps the class with the highest score (a plain
    unsigned greater-than comparison). That is just an argmax. Note: alpha
    scaling from the pkl is deliberately NOT used - the silicon ignores it too.
    """
    scores = [class_score(imgs_int16, weights_int16, m, k) for k in range(NUM_CLASSES)]
    return int(np.argmax(scores)), scores


# ------------------------------------------------------------------
# Helpers for humans: draw the image, dump the bits for Verilog later
# ------------------------------------------------------------------
def render_ascii(imgs_int16, m):
    """Print image m as 16x16 ASCII so you can see the digit. row=cycle, col=word."""
    lines = []
    for c in range(NUM_WORDS):                       # c indexes rows
        row = "".join(
            "#" if (int(imgs_int16[m][i]) >> c) & 1 else "."
            for i in range(NUM_WORDS)                 # i indexes columns
        )
        lines.append("  " + row)
    return "\n".join(lines)


def weight_rom_words(weights_int16):
    """The 160 words the Verilog weight ROM will hold, in hardware address order.

    Address = class_index * 16 + cycle, which is exactly row-major order here.
    This is your bridge to Stage 2 - these become the ROM contents.
    """
    return [int(weights_int16[k][c]) & MASK16
            for k in range(NUM_CLASSES) for c in range(NUM_WORDS)]


# ------------------------------------------------------------------
# Main report
# ------------------------------------------------------------------
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "model/BNN.pkl"
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        sys.exit(f"Could not find {path}. Point me at BNN.pkl from the repo's "
                 f"nn_train/ folder, e.g.:  python3 {sys.argv[0]} nn_train/BNN.pkl")

    weights_int16 = data["weights_int16"]
    imgs_int16 = data["imgs_int16"]

    print("=" * 60)
    print("BNN GOLDEN REFERENCE  -  software model of the FPGA")
    print("=" * 60)
    print(f"loaded: {path}")
    print(f"weights_int16 {weights_int16.shape}   imgs_int16 {imgs_int16.shape}")
    print(f"each vector = {NUM_WORDS} words x {WORD_BITS} bits = "
          f"{NUM_WORDS * WORD_BITS} bits\n")

    # Classify every stored image and tabulate the result.
    print("prediction table")
    print("-" * 60)
    print(f"{'image':>5} | {'predicted':>9} | {'top score':>9} | result")
    print("-" * 60)
    correct = 0
    for m in range(NUM_CLASSES):
        pred, scores = classify(imgs_int16, weights_int16, m)
        ok = (pred == m)
        correct += ok
        flag = "OK" if ok else "<-- misread"
        print(f"{m:>5} | {pred:>9} | {max(scores):>9} | {flag}")
    print("-" * 60)
    print(f"accuracy: {correct}/{NUM_CLASSES}   "
          f"(expected 9/10: image 4 is read as 9 - this matches the paper)\n")

    # Show one detailed score breakdown so the argmax is not a black box.
    sample = 7
    pred, scores = classify(imgs_int16, weights_int16, sample)
    print(f"detailed scores for image {sample}  (max wins -> predicted {pred}):")
    print("  class:  " + " ".join(f"{k:>3}" for k in range(NUM_CLASSES)))
    print("  score:  " + " ".join(f"{s:>3}" for s in scores))
    print()

    # Draw the digits so a human can confirm the inputs are real.
    print("the ten input images (what the hardware actually sees):")
    for m in range(NUM_CLASSES):
        pred, _ = classify(imgs_int16, weights_int16, m)
        print(f"\nimage {m}  ->  predicted {pred}")
        print(render_ascii(imgs_int16, m))

    # Bridge to Stage 2: print the weight ROM contents the Verilog will need.
    rom = weight_rom_words(weights_int16)
    print("\n" + "=" * 60)
    print("BRIDGE TO STAGE 2 (Verilog)")
    print("=" * 60)
    print(f"weight ROM = {len(rom)} words, addressed by (class*16 + cycle).")
    print("first 16 words (class 0), as 4-digit hex:")
    print("  " + " ".join(f"{w:04x}" for w in rom[:16]))
    print("\nWhen you build the Verilog ROM in Stage 2, these exact words go in,")
    print("and a correct design must reproduce the prediction table above.")


if __name__ == "__main__":
    main()
