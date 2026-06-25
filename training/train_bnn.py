#!/usr/bin/env python3
"""
train_bnn.py  -  (PLANNED) PyTorch re-implementation of the BNN training.

Status: stub. The repo ships a pre-trained model/BNN.pkl, so the hardware
works without this. This file exists to replace the original TensorFlow 1.x
training (which used removed APIs like tf.py_func / tf.contrib).

Plan:
  1. Load MNIST, downscale to 16x16, binarize inputs to +/-1.
  2. One binary fully-connected layer: weights binarized with sign().
  3. Train with the straight-through estimator (STE): forward uses sign(),
     backward passes gradients through unchanged.
  4. Export the learned weights in the same packed int16 format as BNN.pkl
     so tools/gen_bnn_data.py and the Verilog ROM keep working unchanged.

Until implemented, use the bundled model/BNN.pkl.
"""
raise SystemExit("train_bnn.py is a planned stub - use the bundled model/BNN.pkl for now.")
