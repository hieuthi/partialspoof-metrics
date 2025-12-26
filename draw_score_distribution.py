#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2025 Hieu-Thi Luong (contact@hieuthi.com)
# MIT License

"""Draw score distribution figure"""

import sys
import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import logging
import warnings

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Draw score distribution figure")
  parser.add_argument('--loadpath', type=str, default=None, required=True, help="Path to the EER result directory.")
  parser.add_argument('--savepath', type=str, default=None, help="Path to the save figure")
  parser.add_argument('--threshold', type=float, action='append', help="The threshold to calculate accuracy")
  parser.add_argument('--xmin', type=float, default=None, help="Minimum x axis value")
  parser.add_argument('--xmax', type=float, default=None, help="Maximum x axis value")
  parser.add_argument('--ymax', type=float, default=None, help="Maximum y axis value")

  args       = parser.parse_args()

  resolution = None
  eer_threshold = None
  minscore, maxscore = -1.0, 1.0
  minval, maxval = -2.0, 2.0

  with open(f"{args.loadpath}/result.txt", 'r') as f:
    for line in f:
      vargs = line.strip().split("=")
      if vargs[0].lower() == "resolution":
        resolution = int(vargs[1])
      elif vargs[0].lower() == "threshold":
        eer_threshold = float(vargs[1])
      elif vargs[0].lower() == "minval":
        minval = float(vargs[1])
      elif vargs[0].lower() == "maxval":
        maxval = float(vargs[1])
      elif vargs[0].lower() == "minscore":
        minscore = float(vargs[1])
      elif vargs[0].lower() == "maxscore":
        maxscore = float(vargs[1])
  counter = np.load(f"{args.loadpath}/counter.npy")
  countersum = np.sum(counter, axis=1, keepdims=True)
  probability = 100.0* counter / countersum
  assert counter.shape[1] == resolution + 1, "ERROR: resolution does not equal with counter array length"

  xmin = args.xmin if args.xmin else minscore
  xmax = args.xmax if args.xmax else maxscore

  x = np.array(range(resolution+1)) / resolution * (maxval-minval) + minval
  minidx = int((xmin-minval) / (maxval-minval) * resolution) - 1
  maxidx = int((xmax-minval) / (maxval-minval) * resolution) + 1

  plt.figure(figsize=(8, 4), dpi=80)
  plt.subplots_adjust(left=0.08, right=0.92, top=0.98, bottom=0.12)
  plt.tick_params(axis='both', which='major', labelsize=10)

  plt.bar(x[minidx:maxidx],probability[0][minidx:maxidx], width=(xmax-xmin)/(maxidx-minidx+1))
  plt.bar(x[minidx:maxidx],probability[1][minidx:maxidx], width=(xmax-xmin)/(maxidx-minidx+1))

  plt.xlim(xmin, xmax)
  if args.ymax:
      plt.ylim(0, args.ymax)

  xmin, xmax, ymin, ymax = plt.axis()
  xunit = (xmax-xmin) / 100
  yunit = (ymax-ymin) / 100

  if args.threshold:
      for thres in args.threshold:
        plt.axvline(x=thres, color='black', linestyle='-', linewidth=0.5)
        plt.text(thres+xunit, yunit*75, f"{thres:.2f}", fontsize=10, color='black')
  plt.axvline(x=eer_threshold, color='red', linestyle='-', linewidth=1)
  plt.text(eer_threshold+xunit, yunit*50, f"{eer_threshold:.2f}", fontsize=10, color='red')

  plt.ylabel("Density (%)", fontsize=12)
  plt.xlabel("Score", fontsize=12)

  plt.savefig(args.savepath)


