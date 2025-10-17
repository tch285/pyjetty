#!/usr/bin/env python3

# import pythia8
import pythiafjext
# import pythiaext
import fastjet as fj
# import fjcontrib
# import fjext
import os
import yaml
import ROOT
import argparse
import numpy as np
import itertools
# import logging
from pathlib import Path

from heppy.pythiautils import configuration as pyconf

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)
# Automatically set Sumw2 when creating new histograms
ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()

def linbins(xmin, xmax, nbins):
    return np.linspace(xmin, xmax, nbins+1)

def logbins(xmin, xmax, nbins):
    return np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)

def deltaR(p1, p2):
    return np.sqrt(p1.delta_phi_to(p2) ** 2 + (p1.eta() - p2.eta()) ** 2)

################################################################
class FastsimTreeBuilder:

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, args = None):
        # Read config file
        with open(args.config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        # Make sure directory of output file exists
        args.output_file.parent.mkdir(parents = True, exist_ok = True)
        self.output_file = args.output_file

        self.nev = args.nev

        # ALICE detector effects
        self.eta_min, self.eta_max = config.get("eta_acc")
        self.tracking_eff = config.get("tracking_eff")
        self.pT_resolution = config.get("pT_resolution")
        self.trk_pT_min = config.get("trk_pT_min")

        # EEC parameters
        self.jet_R = config.get("jet_R", 0.4)

    #---------------------------------------------------------------
    # Main processing function
    #---------------------------------------------------------------
    def generate_events(self, args):
        pythia = pyconf.create_and_init_pythia_from_args(args)

        # Initialize response histograms
        self.prepare()

        # print the banner first
        fj.ClusterSequence.print_banner()
        print()

        self.simulate(pythia)
        pythia.stat()
        print()

        self.finalize()

    #---------------------------------------------------------------
    # Prepare for simulation (initializing histograms, creating jet defs and selectors)
    #---------------------------------------------------------------
    def prepare(self):
        bins_jet_pT = np.array([20., 40., 60., 80.])
        # bins_jet_pT = linbins(0, 200, 200)
        nbins_jet_pT = len(bins_jet_pT) - 1
        bins_RL = logbins(1.0e-3, 1, 30)
        nbins_RL = len(bins_RL) - 1

        # dictionary for storing histograms
        self.hists = {
            'ev': ROOT.TH1I("hNevents", 'Number accepted events', 2, -0.5, 1.5),
            'evw': ROOT.TH1I("evw", 'Event weights;event weights;counts', 100, 0, 20),
        }
        for level in ['det', 'gen']:
            self.hists[f'EEC_{level}_20_40'] = ROOT.TH1D(f'EEC_{level}_20_40', f'EEC_{level}_20_40;R_{{L}}', nbins_RL, bins_RL)
            self.hists[f'EEC_{level}_40_60'] = ROOT.TH1D(f'EEC_{level}_40_60', f'EEC_{level}_40_60;R_{{L}}', nbins_RL, bins_RL)
            self.hists[f'EEC_{level}_60_80'] = ROOT.TH1D(f'EEC_{level}_60_80', f'EEC_{level}_60_80;R_{{L}}', nbins_RL, bins_RL)
            self.hists[f'jet_pT_{level}'] = ROOT.TH1D(f"jet_pT_{level}", f'{level} jet p_{{T}} distribution;jet p_{{T}} (GeV);Counts', nbins_jet_pT, bins_jet_pT)

        self.trees = {}

        self.jet_def = fj.JetDefinition(fj.antikt_algorithm, self.jet_R)

        # jet selector assumes symmetrical eta acceptance
        self.jet_selector = fj.SelectorPtMin(5) & fj.SelectorAbsEtaMax(self.eta_max - self.jet_R)
        # self.jet_selector = fj.SelectorPtMin(20) & fj.SelectorPtMax(80) & fj.SelectorAbsEtaMax(self.eta_max - self.jet_R)

        self.trk_selector = fj.SelectorPtMin(self.trk_pT_min) # selector for det-level particles
        self.eec_trk_selector = fj.SelectorPtMin(1.0) # selector

    #---------------------------------------------------------------
    # Calculate events and pass information on to jet finding
    #---------------------------------------------------------------
    def simulate(self, pythia):
        iev = 0
        while iev < self.nev:
            if not pythia.next():
                self.hists['ev'].Fill(1)
                continue

            self.event = pythia.event
            self.evw = pythia.info.weight()
            self.hists['evw'].Fill(self.evw)

            parts_gen = self.get_gen_parts(pythia)
            # self.save_gen_parts(parts_gen)
            self.analyze(parts_gen, "gen")
            parts_det = self.get_det_parts(parts_gen)
            # self.save_det_parts(parts_det)
            self.analyze(parts_det, "det")

            self.hists['ev'].Fill(0)

            iev += 1

    def get_gen_parts(self, pythia):
        return pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal, pythiafjext.kCharged], 0, True)
    def save_gen_parts(parts):
        pass
    def get_det_parts(self, parts_gen):
        # apply all detector effects, and return a new list of detector-level particles
        return parts_gen # just a placeholder for now so the code runs
    def save_det_parts(parts):
        pass

    def analyze(self, parts, level):
        # apply the track pT minimum even at gen-level, but only for analysis
        jets = self.jet_selector(self.jet_def(self.trk_selector(parts)))
        for jet in jets:
            self._analyze_eec(jet, level)

    def _analyze_eec(self, jet, level):
        # to get charge: pythiafjext.getPythia8Particle(c).charge()!=0:
        jet_pT = jet.pt()
        self.hists[f'jet_pT_{level}'].Fill(jet_pT, self.evw)

        # select jet constituents with 1 GeV cut
        eec_parts = self.eec_trk_selector(jet.constituents())
        for p1, p2 in itertools.permutations(eec_parts, 2):
            pair_ew = p1.pt() * p2.pt() / jet_pT / jet_pT
            pair_RL = deltaR(p1, p2)

            if jet_pT > 20 and jet_pT < 40:
                self.hists[f'EEC_{level}_20_40'].Fill(pair_RL, pair_ew * self.evw)
            elif jet_pT > 40 and jet_pT < 60:
                self.hists[f'EEC_{level}_40_60'].Fill(pair_RL, pair_ew * self.evw)
            elif jet_pT > 60 and jet_pT < 80:
                self.hists[f'EEC_{level}_60_80'].Fill(pair_RL, pair_ew * self.evw)

    def finalize(self):
        with ROOT.TFile(str(self.output_file), "RECREATE") as f:
            for h in self.hists.values():
                f.WriteTObject(h)
            for t in self.trees.values():
                f.WriteTObject(t)

################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pythia on-the-fly fast simulation',
                                     prog=os.path.basename(__file__))
    pyconf.add_standard_pythia_args(parser)
    parser.add_argument('-o', '--output-file', default="./results.root", type=Path,
                        help='Output file path for generated ROOT file(s)')
    parser.add_argument('-c', '--config_file', default='config.yaml', type=Path,
                        help="Path of config file")

    args = parser.parse_args()

    process = FastsimTreeBuilder(args = args)
    process.generate_events(args)