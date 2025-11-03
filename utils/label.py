#!/usr/bin/env python
# encoding: utf-8

# Copyright (c) 2025 Hieu-Thi Luong (contact@hieuthi.com)
# MIT License

import numpy as np


def intersect(a, b):
  return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def items_to_segs(items, tag_filter=None, labdict=None):
  segs = []
  for item in items:
    args = item.split('-')
    assert args[2] in ["spoof","bonafide"], f"ERROR: Unknown label type {args[2]}"
    if args[2] == tag_filter or tag_filter is None:
      start, end = float(args[0]), float(args[1])
      label = labdict[args[2]] if labdict else args[2]
      segs.append([ start, end, label])
  return segs

def segs_to_lab(fakesegs, dur, unit=0.02, sensitivity=0.0):
  lab, n = [], int(dur / unit + 0.5)
  idx, start, end = 0, 0.0, 0.0
  for i in range(n):
      area, start, end = 0.0, end, end + unit
      for idx in range(idx, len(fakesegs)):
          if end < fakesegs[idx][0]:
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

def load_partialspoof_timestamp(filepath):
  labs = {}
  with open(filepath, 'r') as f:
    for line in f:
      args      = line.strip().split()
      name, dur = args[0], float(args[1])
      labs[name] = items_to_segs(args[3:], labdict={"bonafide":0, "spoof": 1})
  return labs