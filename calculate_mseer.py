#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2025 Hieu-Thi Luong (contact@hieuthi.com)
# MIT License

"""Calculate Millisecond EER using LlamaPartialSpoof label and a score file"""

import sys
import os.path
import argparse
import numpy as np
import time
import math
import logging
import warnings

from metrics.mseer import compute_mseer
from utils.label import load_partialspoof_timestamp


logger = logging.getLogger(__name__)
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#warnings.filterwarnings("ignore")


def load_scores(filepath, scoreindex=1, unit=0.02, negative_class=False):
  scos, minscore, maxscore = {}, math.inf, -math.inf
  with open(filepath, 'r') as f:
    for idx, line in enumerate(f):
      args  = line.strip().split()
      name, i, score = args[0], int(args[1]), float(args[scoreindex])
      score = score if not negative_class else 1 - score
      item = [i*unit, (i+1)*unit, score]
      if name in scos:
        scos[name].append(item)
      else:
        scos[name] = [item]
      minscore = score if score < minscore else minscore
      maxscore = score if score > maxscore else maxscore
  return scos, minscore, maxscore



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Calculate Range EER')
  parser.add_argument('--labpath', type=str, required=True, help="Path to Llama Partial Spoof label file.")
  parser.add_argument('--scopath', type=str, required=True, help="Path to score file.")
  parser.add_argument('--savepath', type=str, default=None, help="Path to directory tp save computed data.")
  parser.add_argument('--resolution', type=int, default=100000, help="Threshold resolution")
  parser.add_argument('--unit', type=float, default=0.0, help="Segment duration if unit>0.0 else utterance-based.")
  parser.add_argument('--negative_class', action="store_true", help="Using score of the negative class")
  parser.add_argument('--scoreindex', type=int, default=1, help="Index of the score column.")
  parser.add_argument('--minval', type=float, default=-2.0, help="Score lower bound.")
  parser.add_argument('--maxval', type=float, default=2.0, help="Score higher bound.")


  args = parser.parse_args()

  start = time.time()


  labs = load_partialspoof_timestamp(args.labpath)
  scos, minscore, maxscore = load_scores(args.scopath, scoreindex=args.scoreindex, unit=args.unit, negative_class=args.negative_class)

  assert minscore > args.minval and maxscore < args.maxval, f"ERROR: score ({minscore},{maxscore}) is outside calculating boundary ({args.minval},{args.maxval})"

  eer, threshold, margin, fpr, fnr, counter = compute_mseer(labs, scos, resolution=args.resolution, minval=args.minval, maxval=args.maxval)
  print(f"mseer={eer*100:.2f}% margin={margin*100:.2f}% threshold={threshold:.4f} minscore={minscore:.3f} maxscore={maxscore:.3f} negative={args.negative_class}")

  totaldur = np.sum(counter) / 3600

  end     = time.time()
  elapsed = (end-start)/60
  logger.info(f"INFO: Calculate 1-ms EER took {elapsed:.2f} minutes")


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
      f.write(f"unit_cal=millisecond\n")
      f.write(f"minscore={minscore}\n")
      f.write(f"maxscore={maxscore}\n")
      f.write(f"minval={args.minval}\n")
      f.write(f"maxval={args.maxval}\n")
      f.write(f"nagative_class={args.negative_class}\n")
      f.write(f"resolution={args.resolution}\n")
      f.write(f"scoreindex={args.scoreindex}\n")
      f.write(f"labpath={args.labpath}\n")
      f.write(f"scopath={args.scopath}\n")
      f.write(f"savepath={args.savepath}\n")
      f.write(f"utterances={len(labs)}\n")
      f.write(f"class_0={countersum[0]:.04f}\n")
      f.write(f"class_1={countersum[1]:.04f}\n")
      f.write(f"class_total={sum(countersum):.04f}\n")
    logger.info(f"INFO: Saved computed data to {args.savepath}")
