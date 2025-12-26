#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2024 Hieu-Thi Luong (contact@hieuthi.com)
# MIT License

"""Calculate Utterance-base EER using LlamaPartialSpoof label and a score file"""

import sys
import os.path
import argparse
import numpy as np
import time
import math
import logging
import warnings

from metrics.eer import compute_eer
from utils.label import load_partialspoof_labels

logger = logging.getLogger(__name__)
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#warnings.filterwarnings("ignore")


def load_scores(filepath, scoreindex=1, negative_class=False):
  scos, minscore, maxscore = {}, math.inf, -math.inf
  with open(filepath, 'r') as f:
    for line in f:
      args  = line.strip().split()
      name, score = args[0], float(args[scoreindex])
      score = score if not negative_class else 1 - score
      if name in scos:
        scos[name].append(score)
      else:
        scos[name] = [score]
      minscore = score if score < minscore else minscore
      maxscore = score if score > maxscore else maxscore

  return scos, minscore, maxscore

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Calculate Utterance-based EER for Llama Partial Spoof")
  parser.add_argument('--labpath', type=str, required=True, help="Path to Llama Partial Spoof label file.")
  parser.add_argument('--scopath', type=str, required=True, help="Path to score file.")
  parser.add_argument('--savepath', type=str, default=None, help="Path to directory tp save computed data.")
  parser.add_argument('--resolution', type=int, default=10000, help="Threshold resolution.")
  parser.add_argument('--scoreindex', type=int, default=1, help="Index of the score column.")
  parser.add_argument('--unit', type=float, default=0.0, help="Segment duration if unit>0.0 else utterance-based.")
  parser.add_argument('--zoom', type=int, default=1, help="Zoom in or out to get finer or coaster scores")
  parser.add_argument('--negative_class', action="store_true", help="Using score of the negative class")
  parser.add_argument('--minval', type=float, default=-2.0, help="Score lower bound.")
  parser.add_argument('--maxval', type=float, default=2.0, help="Score higher bound.")
  parser.add_argument('--sensitivity', type=float, default=0.0, help="Sensitivity to extract real/fake label.")


  args = parser.parse_args()

  start = time.time()

  resolution = args.resolution
  tag = "Utterance-based" if args.unit == 0 else f"{args.unit}s Segment-based"
  logger.info(f"INFO: Calculate {tag} EER using {resolution}-bucket threshold NAGATIVE_CLASS={args.negative_class}")

  if args.zoom == 0:
      unit_cal = 0.0
  elif args.zoom > 0:
      unit_cal = args.unit / args.zoom
  else:
      unit_cal = args.unit * (-args.zoom)

  labs = load_partialspoof_labels(args.labpath, unit=unit_cal, sensitivity=args.sensitivity)
  logger.info(f"INFO: Loaded {len(labs)} labels from {args.labpath} with UNIT_INPUT={args.unit} and UNIT_CAL={unit_cal} (0 means utterance-based)")
  scos, minscore, maxscore = load_scores(args.scopath, scoreindex=args.scoreindex, negative_class=args.negative_class)
  logger.info(f"INFO: Loaded {len(scos)} scores from {args.scopath} INDEX={args.scoreindex}")
  assert minscore > args.minval and maxscore < args.maxval, f"ERROR: score ({minscore},{maxscore}) is outside calculating boundary ({args.minval},{args.maxval})"

  if args.zoom > 1:
      for name in scos:
          scos[name] = np.repeat(np.array(scos[name]),args.zoom).tolist()
  elif args.zoom < -1:
      for name in scos:
          item, itemlen = scos[name], len(scos[name])
          if itemlen % (-args.zoom) > 0:
              item = np.pad(item,(0,-args.zoom-(itemlen%(-args.zoom))), 'edge')
          item = np.reshape(item, (-1, -args.zoom))
          item = np.max(item, axis=1)
          scos[name] = item
  elif args.zoom==0:
      for name in scos:
          item, itemlen = scos[name], len(scos[name])
          item = np.max(item)
          scos[name] = [ item ]


  end     = time.time()
  elapsed = (end-start)/60
  logger.info(f"INFO: Loading data took {elapsed:.2f} minutes")
  start   = end

  eer, threshold, margin, fpr, fnr, counter = compute_eer(labs, scos, resolution=resolution, minval=args.minval, maxval=args.maxval)

  print(f"eer={eer*100:.2f}% margin={margin*100:.2f}% threshold={threshold:.4f} negative={args.negative_class}")
  sys.stdout.flush()

  end     = time.time()
  elapsed = (end-start)/60
  logger.info(f"INFO: Calculate {tag} EER took {elapsed:.2f} minutes")

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
      f.write(f"unit_input={args.unit}\n")
      f.write(f"unit_cal={unit_cal}\n")
      f.write(f"minscore={minscore}\n")
      f.write(f"maxscore={maxscore}\n")
      f.write(f"minval={args.minval}\n")
      f.write(f"maxval={args.maxval}\n")
      f.write(f"negative_class={args.negative_class}\n")
      f.write(f"resolution={resolution}\n")
      f.write(f"scoreindex={args.scoreindex}\n")
      f.write(f"labpath={args.labpath}\n")
      f.write(f"scopath={args.scopath}\n")
      f.write(f"savepath={args.savepath}\n")
      f.write(f"utterances={len(labs)}\n")
      f.write(f"class_0={countersum[0]}\n")
      f.write(f"class_1={countersum[1]}\n")
      f.write(f"class_total={sum(countersum)}\n")
    logger.info(f"INFO: Saved computed data to {args.savepath}")
