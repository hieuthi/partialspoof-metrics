#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2024 Hieu-Thi Luong (contact@hieuthi.com)
# MIT License

"""Calculate Accuracy Precision Recall and F1 from EER results"""

import sys
import os.path
import argparse
import numpy as np
import time
import math
import logging
import warnings

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Calculate Accuracy Precision Recall and F1 from the EER result")
  parser.add_argument('--loadpath', type=str, default=None, help="Path to the EER result directory.")
  parser.add_argument('--threshold', type=float, default=0.5, help="The threshold to calculate accuracy")
  parser.add_argument('--eer_threshold', action="store_true", help="Using EER threshold for calculation")

  args       = parser.parse_args()
  threshold  = args.threshold

  minval, maxval = -2.0, 2.0
  resolution = 8000
  with open(f"{args.loadpath}/result.txt", 'r') as f:
    for line in f:
      vargs = line.strip().split("=")
      if vargs[0] == "minval":
        minval = float(vargs[1])
      elif vargs[0] == "maxval":
        maxval = float(vargs[1])
      elif vargs[0] == "resolution":
        resolution = int(vargs[1])
      elif vargs[0] == "Threshold":
        if args.eer_threshold:
          threshold = float(vargs[1])

    index   = int((threshold-minval)/(maxval-minval)*resolution)
    counter = np.load(f"{args.loadpath}/counter.npy")
    assert counter.shape[1] == resolution + 1, "ERROR: resolution does not equal with counter array length"
    data = np.cumsum(counter, axis=1)
    tn, fn = data[0,index], data[1,index]
    tp, fp = data[1,-1] - data[1,index], data[0,-1] - data[0, index]
    accuracy  = (tn+tp)/(tn+tp+fn+fp)
    precision = tp / (tp+fp)
    recall    = tp / (tp+fn)
    f1        = (2*precision*recall) / (precision+recall)

    print(f"Threshold={threshold}")
    print(f"Index={index}")
    print(f"Accuracy={accuracy*100:.04f}")
    print(f"Precision={precision*100:.04f}")
    print(f"Recall={recall*100:.04f}")
    print(f"F1={f1*100:.04f}")

