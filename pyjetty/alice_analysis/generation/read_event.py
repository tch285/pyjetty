#!/usr/bin/env python3

import sys

# import pyhepmc
import pyhepmc_ng

# Open the file in read mode
# filename = "/rstorage/ploskon/eec_sherpa_inclusive_5TeV/sherpa2x/jetpt20/sherpa_LHC_jets_20.hepmc"
# filename = "/rstorage/mhwang/sherpa/sherpa_ahadic.txt"
filenames = ["/rstorage/mhwang/sherpa/sherpa_ahadic.txt", "/rstorage/mhwang/sherpa/sherpa_lund.txt"]

for filename in filenames:
  with open(filename, 'r') as file:
    # Loop through each line in the file
    for ifile, input_file in enumerate(file):
      input_file = input_file.strip()
      print("input file: ", ifile, input_file) # strip() removes any leading/trailing whitespace

      hepmc = 3

      if hepmc == 3:
        # input_hepmc = pyhepmc.io.ReaderAscii(input_file)
        input_hepmc = pyhepmc_ng.ReaderAscii(input_file)
      if hepmc == 2:
        # input_hepmc = pyhepmc.io.ReaderAsciiHepMC2(input_file)
        input_hepmc = pyhepmc_ng.ReaderAsciiHepMC2(input_file)

      if input_hepmc.failed():
        print ("[error] unable to read from {}".format(input_file))
        sys.exit(1)

      # event_hepmc = pyhepmc.GenEvent()
      event_hepmc = pyhepmc_ng.GenEvent()
      ev = input_hepmc.read_event(event_hepmc)
      # # for event in input_hepmc:
      print('reading one event')
      print(len(event_hepmc.particles))
      print("done!")