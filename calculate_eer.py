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

logger = logging.getLogger(__name__)
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#warnings.filterwarnings("ignore")


def intersect(a, b):
  return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def items_to_segs(items, tag_filter=None):
  segs = []
  for item in items:
    args = item.split('-')
    assert args[2] in ["spoof","bonafide"], f"ERROR: Unknown label type {args[2]}"
    if args[2] == tag_filter or tag_filter is None:
      segs.append([float(args[0]), float(args[1]), args[2]])
  return segs

def segs_to_lab(fakesegs, dur, unit=0.02, sensitivity=0.0):
  lab, n = [], int(dur / unit)
  idx, start, end = 0, 0.0, 0.0
  for i in range(n):
      area, start, end = 0.0, end, end + unit
      for idx in range(idx, len(fakesegs)):
          if idx >= len(fakesegs) or end < fakesegs[idx][0]:
              break
          else:
              area = area + intersect([start,end], fakesegs[idx][:2])
              if end < fakesegs[idx][1]:
                  break
      lab.append(1 if area/unit > sensitivity else 0)
  return lab

def load_partialspoof_labels(filepath, unit=0.0, sensitivity=0.0):
  labs = {}
  with open(filepath, 'r') as f:
    for line in f:
      args      = line.strip().split()
      name, dur = args[0], float(args[1])
      if unit == 0.0: # utterance-based
        assert args[2] in ["spoof","bonafide"], f"ERROR: Unknown utterance-based label type {args[2]}"
        labs[name] = [1] if args[2]=="spoof" else [0]
      else: # segment-based
        fakesegs   = items_to_segs(args[3:], tag_filter="spoof")
        labs[name] = segs_to_lab(fakesegs, dur, unit=unit, sensitivity=sensitivity)
  return labs

def load_normalized_scores(filepath, scoreindex=1, negative_class=False):
  scos, minscore, maxscore = {}, math.inf, -math.inf
  with open(filepath, 'r') as f:
    for line in f:
      args  = line.strip().split()
      name, score = args[0], float(args[scoreindex])
      if name in scos:
        scos[name].append(score)
      else:
        scos[name] = [score]
      minscore = score if score < minscore else minscore
      maxscore = score if score > maxscore else maxscore
  for name in scos:
    scos[name] = score if not negative_class else 1 - score
  return scos, minscore, maxscore

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
  parser.add_argument('--labpath', type=str, help="Path to Llama Partial Spoof label file.")
  parser.add_argument('--scopath', type=str, help="Path to score file.")
  parser.add_argument('--savepath', type=str, default=None, help="Path to directory tp save computed data.")
  parser.add_argument('--resolution', type=int, default=8000, help="Threshold resolution.")
  parser.add_argument('--scoreindex', type=int, default=1, help="Index of the score column.")
  parser.add_argument('--unit', type=float, default=0.0, help="Segment duration if unit>0.0 else utterance-based.")
  parser.add_argument('--zoom', type=int, default=1, help="Zoom in or out to get finer or coaster scores")
  parser.add_argument('--negative_class', action="store_true", help="Using score of the negative class")
  parser.add_argument('--minval', type=float, default=-2.0, help="Score lower bound.")
  parser.add_argument('--maxval', type=float, default=2.0, help="Score higher bound.")

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

  labs = load_partialspoof_labels(args.labpath, unit=unit_cal)
  logger.info(f"INFO: Loaded {len(labs)} labels from {args.labpath} with UNIT_INPUT={args.unit} and UNIT_CAL={unit_cal} (0 means utterance-based)")
  scos, minscore, maxscore = load_scores(args.scopath, scoreindex=args.scoreindex, negative_class=args.negative_class)
  logger.info(f"INFO: Loaded {len(scos)} scores from {args.scopath} INDEX={args.scoreindex}")
  if args.zoom > 1:
      for name in scos:
          scos[name] = np.repeat(np.array(scos[name]),args.zoom).tolist()
  elif args.zoom < -1:
      for name in scos:
          item, itemlen = scos[name], len(scos[name])
          if itemlen % (-args.zoom) > 0:
              item = np.pad(item,(0,-args.zoom-(itemlen%(-args.zoom))), 'edge')
          item = np.reshape(item, (-1, -args.zoom))
          item = np.min(item, axis=1) if args.negative_class else np.max(item, axis=1)
          scos[name] = item
  elif args.zoom==0:
      for name in scos:
          item, itemlen = scos[name], len(scos[name])
          item = np.min(item) if args.negative_class else np.max(item)
          scos[name] = [ item ]




  end     = time.time()
  elapsed = (end-start)/60
  logger.info(f"INFO: Loading data took {elapsed:.2f} minutes")
  start   = end

  eer, threshold, margin, fpr, fnr, counter = compute_eer(labs, scos, resolution=resolution, minval=args.minval, maxval=args.maxval)

  threshold = 1 - threshold if args.negative_class else threshold
  print(f"{eer*100:.2f} {margin*100:.2f} {threshold:.4f}\n")
  sys.stdout.flush()

  end     = time.time()
  elapsed = (end-start)/60
  logger.info(f"INFO: Calculate {tag} EER took {elapsed:.2f} minutes")

  if args.savepath is not None:
    np.save(f"{args.savepath}/fpr.npy", fpr)
    np.save(f"{args.savepath}/fnr.npy", fnr)
    np.save(f"{args.savepath}/counter.npy", counter)
    with open(f"{args.savepath}/result.txt", "w") as f:
      f.write(f"EER={eer}\n")
      f.write(f"Threshold={threshold}\n")
      f.write(f"Margin={margin}\n")
      f.write(f"unit_input={args.unit}\n")
      f.write(f"unit_cal={unit_cal}\n")
      f.write(f"minscore={minscore}\n")
      f.write(f"maxscore={maxscore}\n")
      f.write(f"minval={args.minval}\n")
      f.write(f"maxval={args.maxval}\n")
      f.write(f"nagative_class={args.negative_class}\n")
      f.write(f"resolution={resolution}\n")
      f.write(f"scoreindex={args.scoreindex}\n")
      f.write(f"labpath={args.labpath}\n")
      f.write(f"scopath={args.scopath}\n")
    logger.info(f"INFO: Saved computed data to {args.savepath}")
