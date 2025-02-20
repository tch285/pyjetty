#!/usr/bin/env python3

import os
import ROOT

base = "/rstorage/mhwang/sherpa/lund_60"

total = 0
for rootdir, _, filenames in os.walk(base):
  for filename in filenames:
    file_path = os.path.join(rootdir, filename)
    with ROOT.TFile(file_path, "R") as f:
      nentries = f.Get("PWGHF_TreeCreator/tree_event_char").GetEntriesFast()
      total += nentries

print(total)