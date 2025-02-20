#!/usr/bin/env python3

import sys

import pyhepmc_ng

# filename = "/rstorage/ploskon/eec_sherpa_inclusive_5TeV/sherpa2x/jetpt20/sherpa_LHC_jets_20.hepmc"
# filename = "/rstorage/mhwang/sherpa/sherpa_ahadic.txt"
filenames = ["/rstorage/mhwang/sherpa/sherpa_ahadic.txt", "/rstorage/mhwang/sherpa/sherpa_lund.txt"]

def count_events(hepmc_filename):
  hepmc_filename = hepmc_filename.strip()
  print("input file:", hepmc_filename)

  hepmc = 3

  if hepmc == 3:
    # input_hepmc = pyhepmc.io.ReaderAscii(hepmc_filename)
    input_hepmc = pyhepmc_ng.ReaderAscii(hepmc_filename)
  if hepmc == 2:
    # input_hepmc = pyhepmc.io.ReaderAsciiHepMC2(hepmc_filename)
    input_hepmc = pyhepmc_ng.ReaderAsciiHepMC2(hepmc_filename)

  if input_hepmc.failed():
    print (f"[error] unable to read from {hepmc_filename}")
    sys.exit(1)

  evid = 0
  for event in input_hepmc:
    if evid % 10000 == 0:
      print(evid)
    evid += 1

  # event_hepmc = pyhepmc.GenEvent()
  # event_hepmc = pyhepmc_ng.GenEvent()
  # evid = 0
  # while not input_hepmc.failed():
  #   ev = input_hepmc.read_event(event_hepmc)
  #   # print("input_hepmc", evid, input_hepmc.failed())
  #   if input_hepmc.failed():
  #     break
  #   evid+=1

  print(f"Total number of events in hepmc file {hepmc_filename} is {evid}")

count_events("/rstorage/ploskon/eec_sherpa_inclusive_5TeV/sherpa2x/jetpt40_lund/sherpa_LHC_jets_40.0.hepmc")
# for filename in filenames:
#   with open(filename, 'r') as file:
#     # Loop through each line in the file
#     for ifile, hepmc_filename in enumerate(file):
#       count_events(hepmc_filename)