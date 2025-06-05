#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2024 Hieu-Thi Luong (contact@hieuthi.com)
# MIT License

"""Utility functions to calculate Equal Error Rate"""

import numpy as np
import warnings

def _pad_score_array(score, length):
    if score.shape[0] < length:
        return np.pad(score,(0,length-score.shape[0]),mode='edge')
    else:
        return score[:length]

def _calculate_det_curve(counter):
  data = np.cumsum(counter, axis=1)
  data = np.divide(data, data[:,-2:-1])
  fpr = 1 - data[0,:]
  fnr = data[1,:]
  return fpr, fnr

def _calculate_eer(fpr, fnr):
  margin = np.abs(fpr - fnr)
  idxmin = np.argmin(margin)
  eer       = (fpr[idxmin]+fnr[idxmin])/2
  threshold = idxmin / (margin.shape[0]-1)
  return eer, threshold, margin[idxmin]

def _count_samples(labs, scos, resolution=8000, counter=None, minval=-2.0, maxval=2.0):
  if counter is None:
    counter = np.zeros((2,resolution+1))
  assert resolution+1 == counter.shape[1], "ERROR: the length of the preloaded counter and the resolution is not equal"
  names = labs.keys()
  for name in names:
    lab, sco = labs[name], scos[name]
    lab      = np.array(lab) if not isinstance(lab, np.ndarray) else lab
    sco      = np.array(sco) if not isinstance(sco, np.ndarray) else sco
    if lab.shape[0] != sco.shape[0]:
      warnings.warn(f"WARNING: {name} has {lab.shape[0]} labels but has {sco.shape[0]} scores and will be padded")
      sco = _pad_score_array(sco, lab.shape[0])
    sco = (sco-minval)/(maxval-minval)
    for labtype in [0,1]:
      idxs           = (sco[lab==labtype]*resolution).astype(np.int64)
      unique, counts = np.unique(idxs, return_counts=True)
      counter[labtype,unique] = counter[labtype,unique] + counts
  return counter

def compute_eer(labs, scos, resolution=8000, counter=None, minval=-2.0, maxval=2.0):
  """Compute EER using an evenly spacing threshold

  Parameters:
  ----------
  labs: dictionary[int] or dictionary[np.narray(int)]
    Labels [0,1] for the testing utterances
  scos: dictionary[float] or dictionary[np.narray(float)]
    Scores (float) of the posstive (1) class
  resolution: int, optional
    Number of threshold bucket
  counter: np.narray with shape (2,resolution), optional
    Using a preloaded data, use for large evaluation set
  """
  counter  = _count_samples(labs, scos, resolution=resolution, counter=counter, minval=minval, maxval=maxval)
  fpr, fnr = _calculate_det_curve(counter)
  eer, threshold, margin = _calculate_eer(fpr,fnr)
  threshold = threshold * (maxval-minval) + minval
  return eer, threshold, margin, fpr, fnr, counter