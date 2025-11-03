#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2025 Hieu-Thi Luong (contact@hieuthi.com)
# MIT License

"""Utility functions to calculate Millisecond Equal Error Rate"""

import numpy as np
import warnings

from .eer import _calculate_det_curve, _calculate_eer

def _pad_score_array(sco, lab):
  dur = lab[-1][1]
  for i in range(len(sco)-1,-1,-1):
    if sco[i][1] <= dur:
      sco = sco[:i+1]
      break
  sco[-1][1] = dur
  return sco

def _count_one_sample(counter, ref, hyp, resolution=8000, minval=-2.0, maxval=2.0):
  dur = ref[-1][1]

  label = ref[0][-1]                  # starting label
  rscur, recur, ridx = 0.0, 0.0, 0    # ref cursor
  hscur, hecur, hidx = 0.0, 0.0, 0    # hyp cursor

  totaldur = 0                        # for validating

  while rscur < dur:
    if rscur >= recur: # update reference if start touch end
      if ridx < len(ref): # there is annotation left
        ritem = ref[ridx]
        recur, ridx, label = ritem[1], ridx+1, ritem[-1]
      else:
        break

    if hscur >= hecur: # update hypothesis if start touch end
      if hidx < len(hyp):
        hitem = hyp[hidx]
        hecur, hidx, score = hitem[1], hidx+1, hitem[2]
        score = (score-minval)/(maxval-minval)
      else:
        break

    # Create a predicted segment and update start cursors
    segdur = hecur-hscur if hecur<=recur else recur-rscur
    bidx   = int(score*resolution)
    counter[label,bidx] += segdur
    rscur = rscur+segdur
    hscur = rscur

    totaldur += segdur
    # # DEBUG
    # print(f"split at {rscur} : {rscur} {recur} : {hscur} {hecur}")

  return counter

def _count_samples(labs, scos, resolution=8000, counter=None, minval=-2.0, maxval=2.0):
  if counter is None:
    counter = np.zeros((2,resolution+1))
  assert resolution+1 == counter.shape[1], "ERROR: the length of the preloaded counter and the resolution is not equal"
  names = labs.keys()
  for name in names:
    lab, sco = labs[name], scos[name]
    warnings.warn(f"WARNING: {name} is {lab[-1][1]}s long but the score is {sco[-1][1]}s so the score will be padded")
    sco = _pad_score_array(sco, lab)
    counter = _count_one_sample(counter, lab, sco, resolution=resolution, minval=minval, maxval=maxval)
  return counter

def compute_mseer(labs, scos, resolution=8000, counter=None, minval=-2.0, maxval=2.0):
  counter  = _count_samples(labs, scos, resolution=resolution, counter=counter, minval=minval, maxval=maxval)
  fpr, fnr = _calculate_det_curve(counter)
  eer, threshold, margin = _calculate_eer(fpr,fnr)
  threshold = threshold * (maxval-minval) + minval
  return eer, threshold, margin, fpr, fnr, counter

