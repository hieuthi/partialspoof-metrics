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
  parser.add_argument('--recall', type=float, default=0.0, help="Calculate at a specific recall value")

  args       = parser.parse_args()
  threshold  = args.threshold

  minval, maxval = -2.0, 2.0
  resolution = 8000
  negative_class = False
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
      elif vargs[0] == "negative_class":
          if vargs[1] == "True":
              negative_class = True

    counter = np.load(f"{args.loadpath}/counter.npy")
    assert counter.shape[1] == resolution + 1, "ERROR: resolution does not equal with counter array length"
    data = np.cumsum(counter, axis=1)
    tn, fn = data[0,:], data[1,:]
    tp, fp = data[1,-1] - data[1,:], data[0,-1] - data[0, :]
    total  = data[0,-1]+data[1,-1]
    accuracy  = (tn+tp)/total
    # Prevent divide by 0 error
    tpfp, tpfn = tp+fp, tp+fn
    tpfp[tpfp==0] = 1
    tpfn[tpfn==0] = 1
    precision = np.divide(tp, tpfp)
    recall    = np.divide(tp, tpfn)

    index   = int((threshold-minval)/(maxval-minval)*resolution)
    if args.recall > 0:
        for i in range(len(recall)):
            if recall[i] > args.recall:
                index = i
                threshold = index * 1.0 / resolution * (maxval-minval) + minval

    f1 = 2*precision[index]*recall[index] / (precision[index]+recall[index])

    print(f"Threshold={threshold:.04f}")
    print(f"Index={index}")
    print(f"Accuracy={accuracy[index]*100:.02f}")
    print(f"Precision={precision[index]*100:.02f}")
    print(f"Recall={recall[index]*100:.02f}")
    print(f"F1={f1*100:.02f}")

