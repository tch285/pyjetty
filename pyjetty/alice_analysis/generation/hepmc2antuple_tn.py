#!/usr/bin/env python3

import os
import argparse
import sys
from time import perf_counter
import pyhepmc_ng

import hepmc2antuple_base
# import faulthandler
# faulthandler.enable()

################################################################
class HepMC2antuple(hepmc2antuple_base.HepMC2antupleBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, event_split = 10_000, **kwargs):
    super(HepMC2antuple, self).__init__(**kwargs)
    self.event_split = event_split
    self.init()
    print(self)

  #---------------------------------------------------------------
  def setup_split(self, isplit):
    self.outdir = f"{self.output}/{isplit}"
    os.makedirs(self.outdir, exist_ok = True)
    self.init_trees()
    print(f"Setting up split number {isplit}.")
    self.start_time = perf_counter()

  def setup_output(self):
    self.outdir = self.output
    os.makedirs(self.outdir, exist_ok = True)
    self.init_trees()
    print("Setting up output directory.")
    self.start_time = perf_counter()

  #---------------------------------------------------------------
  def main(self):
    if self.hepmc == 3:
      input_hepmc = pyhepmc_ng.ReaderAscii(self.input)
    if self.hepmc == 2:
      input_hepmc = pyhepmc_ng.ReaderAsciiHepMC2(self.input)

    if input_hepmc.failed():
      print ("[error] unable to read from {}".format(self.input))
      sys.exit(1)

    if self.event_split != 0:
      isplit = 1
      self.setup_split(isplit)

      for event in input_hepmc:
        if self.ev_id % self.event_split == 0 and self.ev_id > 0:
          self.save_trees()
          print(f"Split {isplit} completed in {perf_counter() - self.start_time:.2f} s.")
          isplit += 1
          self.setup_split(isplit)
        self.fill_event(event)
        self.increment_event()
      self.save_trees()
      print(f"Final split {isplit} completed in {perf_counter() - self.start_time:.2f} s.")
    else:
      self.setup_output()
      for event in input_hepmc:
        self.fill_event(event)
        self.increment_event()
      self.save_trees()

    print("Conversion done!")

  #---------------------------------------------------------------
  def fill_event(self, event_hepmc):

    self.t_e.Fill(self.run_number, self.ev_id, 0, 0)

    for part in event_hepmc.particles:

      if self.accept_particle(part, part.status, part.end_vertex, part.pid, self.pdg, self.gen):
        self.particles_accepted.add(self.pdg.GetParticle(part.pid).GetName())
        # charge = PDGID(part.pid).charge
        charge = self.pdg.GetParticle(part.pid).Charge() / 3 # PDG charge is normalized to quark charge not elementary charge
        self.t_p.Fill(self.run_number, self.ev_id, part.momentum.pt(), part.momentum.eta(), part.momentum.phi(), charge)
        # self.t_p.Fill(self.run_number, self.ev_id, part.momentum.pt(), part.momentum.eta(), part.momentum.phi(), part.pid)

      elif self.include_parton and self.accept_particle(part, part.status, part.end_vertex, part.pid, self.pdg, self.gen, parton=True):

        self.partons_accepted.add(self.pdg.GetParticle(part.pid).GetName())
        self.t_pp.Fill(self.run_number, self.ev_id, part.momentum.pt(), part.momentum.eta(), part.momentum.phi(), part.pid)

#---------------------------------------------------------------
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='hepmc to ALICE Ntuple format', prog=os.path.basename(__file__))
  parser.add_argument('-i', '--input', help='input file', default='', type=str, required=True)
  parser.add_argument('-o', '--output', help='output directory', default='.', type=str, required=True)
  parser.add_argument('-d', '--as-data', help='write as data - tree naming convention', action='store_true', default=False)
  parser.add_argument('--hepmc', help='what format 2 or 3', default=2, type=int)
  parser.add_argument('--nev', help='number of events', default=-1, type=int)
  parser.add_argument('-g', '--gen', help='generator type: pythia, herwig, jewel, jetscape, martini, hybrid', default='pythia', type=str, required=True)
  parser.add_argument('--no-progress-bar', help='whether to print progress bar', action='store_true', default=False)
  parser.add_argument('-p', '--include-parton', help='include additional tree of final-state partons', action='store_true', default=False)
  parser.add_argument('-s', '--event-split', help='define event split', default = 100000, type = int)
  args = parser.parse_args()

  converter = HepMC2antuple(input_file = args.input, output = args.output, as_data = args.as_data, hepmc = args.hepmc, 
                            nev = args.nev, event_split = args.event_split, gen = args.gen,
                            no_progress_bar = args.no_progress_bar, include_parton = args.include_parton)
  converter.main()