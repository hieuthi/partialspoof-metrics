#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2025 Hieu-Thi Luong (contact@hieuthi.com)
# MIT License

"""Combine multiple EER result directories"""

import sys
import os.path
import argparse
import numpy as np
import time
import math
import logging
import warnings

from metrics.eer import _calculate_det_curve, _calculate_eer

logger = logging.getLogger(__name__)
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#warnings.filterwarnings("ignore")

def load_result_info(infile):
  res = {}
  with open(infile, "r") as f:
      for line in f:
          args = line.strip().split("=")
          res[args[0]] = args[1]
  return res

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Combine multiple EER results directory into one")
  parser.add_argument('savepath', type=str, help="Path to the directory to save the results")
  parser.add_argument('loadpaths', type=str, nargs='+', help="Paths to all the inputs results")
  args = parser.parse_args()

  start = time.time()

  negative_class = None
  resolution = None
  minscore, maxscore = None, None
  minval, maxval = None, None

  counters = []
  labpaths, scopaths = [], []
  utterances = 0
  scoreindex, unit, unit_cal = 0, 0, 0


  for loadpath in args.loadpaths:
    resinfo = load_result_info(f"{loadpath}/result.txt")
    counter = np.load(f"{loadpath}/counter.npy")
    counters.append(counter)

    labpaths.append(resinfo["labpath"])
    scopaths.append(resinfo["scopath"])
    utterances = utterances + int(resinfo["utterances"])

    if resolution is None:
      resolution = int(resinfo["resolution"])
      negative_class = True if resinfo["negative_class"] == "True" else False
      minscore, maxscore = float(resinfo["minscore"]), float(resinfo["maxscore"])
      minval, maxval = float(resinfo["minval"]), float(resinfo["maxval"])
      scoreindex, unit, unit_cal = int(resinfo["scoreindex"]), float(resinfo["unit_input"]), float(resinfo["unit_cal"])
      logger.info(f"INFO: Combining results with resolution={resolution}, minval={minval}, maxval={maxval}, negative_class={negative_class}, UNIT_INPUT={unit} and UNIT_CAL={unit_cal} (0 means utterance-based)")
    else:
      cur_resolution = int(resinfo["resolution"])
      cur_negative_class = True if resinfo["negative_class"] == "True" else False
      cur_minscore, cur_maxscore = float(resinfo["minscore"]), float(resinfo["maxscore"])
      cur_minval, cur_maxval = float(resinfo["minval"]), float(resinfo["maxval"])
      assert resolution == cur_resolution, f"ERROR: the input {loadpath} has different resolution ({cur_resolution}) than previous inputs ({resolution})"
      assert negative_class == cur_negative_class, f"ERROR: the input {loadpath} has different negative_class ({cur_negative_class}) than previous inputs ({negative_class})"
      assert minval == cur_minval, f"ERROR: the input {loadpath} has different minval ({cur_minval}) than previous inputs ({minval})"
      assert maxval == cur_maxval, f"ERROR: the input {loadpath} has different maxval ({cur_maxval}) than previous inputs ({maxval})"

      minscore = cur_minscore if cur_minscore < minscore else minscore
      maxscore = cur_maxscore if cur_maxscore > maxscore else maxscore

    logger.info(f"Loading EER result from {loadpath}")
  logger.info(f"INFO: Finish loading {len(args.loadpaths)} EER results")

  counter = np.sum(counters, axis=0)
  fpr, fnr = _calculate_det_curve(counter)
  eer, threshold, margin = _calculate_eer(fpr,fnr)
  threshold = threshold * (maxval-minval) + minval

  print(f"eer={eer*100:.2f}% margin={margin*100:.2f}% threshold={threshold:.4f} negative={negative_class} n_loadpaths={len(args.loadpaths)}")

  end     = time.time()
  elapsed = (end-start)/60
  logger.info(f"INFO: Combine EER results took {elapsed:.2f} minutes")

  if args.savepath is not None:
    os.makedirs(args.savepath, exist_ok=True)
    np.save(f"{args.savepath}/fpr.npy", fpr)
    np.save(f"{args.savepath}/fnr.npy", fnr)
    np.save(f"{args.savepath}/counter.npy", counter)
    countersum = np.sum(counter, axis=1)
    with open(f"{args.savepath}/result.txt", "w") as f:
      f.write(f"eer={eer}\n")
      f.write(f"threshold={threshold}\n")
      f.write(f"margin={margin}\n")
      f.write(f"unit_input={unit}\n")
      f.write(f"unit_cal={unit_cal}\n")
      f.write(f"minscore={minscore}\n")
      f.write(f"maxscore={maxscore}\n")
      f.write(f"minval={minval}\n")
      f.write(f"maxval={maxval}\n")
      f.write(f"negative_class={negative_class}\n")
      f.write(f"resolution={resolution}\n")
      f.write(f"scoreindex={scoreindex}\n")
      f.write(f"labpath={labpaths}\n")
      f.write(f"scopath={scopaths}\n")
      f.write(f"savepath={args.savepath}\n")
      f.write(f"utterances={utterances}\n")
      f.write(f"class_0={countersum[0]}\n")
      f.write(f"class_1={countersum[1]}\n")
      f.write(f"class_total={sum(countersum)}\n")
    logger.info(f"INFO: Saved computed data to {args.savepath}")
