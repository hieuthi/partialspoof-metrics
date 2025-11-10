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
  parser.add_argument('--loadpath', type=str, default=None, required=True, help="Path to the EER result directory.")
  parser.add_argument('--threshold', type=float, default=0.5, help="The threshold to calculate accuracy")
  parser.add_argument('--eer_threshold', action="store_true", help="Using EER threshold for calculation")
  parser.add_argument('--recall', type=float, default=0.0, help="Calculate at a specific recall value")
  parser.add_argument('--precision', type=float, default=0.0, help="Calculate at a specific precision value")


  args       = parser.parse_args()
  threshold  = args.threshold

  minval, maxval = -2.0, 2.0
  resolution = 8000
  with open(f"{args.loadpath}/result.txt", 'r') as f:
    for line in f:
      vargs = line.strip().split("=")
      if vargs[0].lower() == "minval":
        minval = float(vargs[1])
      elif vargs[0].lower() == "maxval":
        maxval = float(vargs[1])
      elif vargs[0].lower() == "resolution":
        resolution = int(vargs[1])
      elif vargs[0].lower() == "threshold":
        if args.eer_threshold:
          threshold = float(vargs[1])


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
    elif args.precision > 0:
        for i in range(len(precision)-1, 0, -1):
            if precision[i] > args.precision:
                index = i
                threshold = index * 1.0 / resolution * (maxval-minval) + minval

    if precision[index] == 0 and recall[index] == 0:
      f1 = 0
    else:
      f1 = 2*precision[index]*recall[index] / (precision[index]+recall[index])

    print(f"threshold={threshold:.04f} index={index} accuracy={accuracy[index]*100:.02f}% precision={precision[index]*100:.02f}% recall={recall[index]*100:.02f}% f1={f1*100:.02f}%")


