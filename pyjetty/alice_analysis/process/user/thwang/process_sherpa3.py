#!/usr/bin/env python3

"""
    Analyze Sherpa HepMC files.

    Author: Tucker Hwang
"""

# General
import os
import sys
import argparse
import itertools
import pyhepmc
from particle import PDGID
import numpy as np
import ROOT
import yaml
import logging

# Fastjet via python (from external library heppy)
import fastjet as fj
# import fjcontrib
# import fjtools
# import ecorrel

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'DEBUG': '\033[34m',
        'INFO': '\033[32m',
        'CRITICAL': '\033[35m'
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        if color:
            # Color the entire line
            formatted_msg = super().format(record)
            return f"{color}{formatted_msg}{self.RESET}"
        return super().format(record)

class PidZeroError(Exception):
    pass

logger = logging.getLogger(__name__)

def linbins(xmin, xmax, nbins):
    lspace = np.linspace(xmin, xmax, nbins+1)
    # arr = array.array('f', lspace)
    return lspace

def logbins(xmin, xmax, nbins):
    lspace = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
    # arr = array.array('f', lspace)
    return lspace

ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()

def charge(p):
    if p.pid == 0:
        raise PidZeroError
    return int(PDGID(p.pid).charge)

def isch(p):
    return charge(p) != 0

def to_psjv(parts):
    psjv = fj.vectorPJ()
    for p in parts:
        fv = p.momentum
        ch = charge(p)
        psj = fj.PseudoJet(fv.px, fv.py, fv.pz, fv.e)
        psj.set_user_index(ch)
        psjv.push_back(psj)
    return psjv

def deltaR(p1, p2):
    deta = p2.eta() - p1.eta()
    dphi = p1.delta_phi_to(p2)
    return np.sqrt(deta*deta + dphi*dphi)

################################################################
class ProcessSherpa3:

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, input_file='', config_file='', output_file=''):
        self.input_file = input_file
        self.output_file = output_file
        with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        self.pT_min, self.pT_max, self.pT_nbins = config["pT_binning"]
        self.RL_min, self.RL_max, self.RL_nbins = config["RL_binning"]
        self.pT_bins = linbins(self.pT_min,self.pT_max,self.pT_nbins)
        self.RL_bins = logbins(self.RL_min,self.RL_max,self.RL_nbins)
        self.ctypes = ["T", "P", "M", "PM", "Q"]
        self.jetR = config.get("jetR", 0.4)
        self.trk_pT_min = config.get("trk_pT_min", 0.15)
        self.trk_pT_thr = config.get("trk_pT_thr", 1.0)

        self.hists = {
            f"EEC_{ctype}": ROOT.TH2D(f"EEC_{ctype}", f"EEC_{ctype}", self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
            for ctype in self.ctypes
        }
        self.hists['jet_pT'] = ROOT.TH1D('jet_pT', 'jet_pT', self.pT_nbins, self.pT_bins)
        self.hists['nev'] = ROOT.TH1D('nev', 'nev', 2, -0.5, 1.5)
        self.jet_def = fj.JetDefinition(fj.antikt_algorithm, self.jetR)
        self.jet_selector = fj.SelectorPtMin(5.0) & fj.SelectorAbsEtaMax(0.9 - self.jetR)
        self.thr_selector = fj.SelectorPtMin(self.trk_pT_thr)


    def analyze(self):
        with pyhepmc.open(self.input_file) as f:
            for i, event in enumerate(f):
                self.evw = event.weight(0) / event.weight(2)
                try:
                    chp = [p for p in event.particles
                        if p.status == 1 and isch(p)
                        and p.momentum.abs_eta() < 0.9
                        and p.momentum.pt() > self.trk_pT_min]
                except PidZeroError as e:
                    logger.warning(f"Event {i} found with PDG ID 0; skipping event.")
                    continue
                psjv = to_psjv(chp)
                self.analyze_event(psjv)
                self.hists['nev'].Fill(1)
        self.save()

    def analyze_event(self, particles):
        cs = fj.ClusterSequence(particles, self.jet_def)
        jets = fj.sorted_by_pt(cs.inclusive_jets())
        jets_selected = self.jet_selector(jets)
        for jet in jets_selected:
            self.analyze_jet(jet)

    def analyze_jet(self, jet):
        self.hists['jet_pT'].Fill(jet.pt(), self.evw)
        parts_sel = self.thr_selector(jet.constituents())

        for p1, p2 in itertools.permutations(parts_sel, 2):
            ew = p1.pt() * p2.pt() / jet.pt() / jet.pt()
            angle = deltaR(p1, p2)
            q1 = p1.user_index()
            q2 = p2.user_index()
            self.hists["EEC_T"].Fill(jet.pt(), angle, ew * self.evw)
            self.hists["EEC_Q"].Fill(jet.pt(), angle, ew * self.evw * q1 * q2)
            if q1 > 0 and q2 > 0:
                ctype = "P"
            elif q1 * q2 < 0:
                ctype = "PM"
            else:
                ctype = "M"
            self.hists[f"EEC_{ctype}"].Fill(jet.pt(), angle, ew * self.evw)

    def save(self):
        with ROOT.TFile(self.output_file, 'RECREATE') as f:
            for h in self.hists.values():
                f.WriteTObject(h)
        logger.info(f"Histograms saved to {self.output_file}")

##################################################################
if __name__ == '__main__':
    # Define arguments
    parser = argparse.ArgumentParser(description='Process MC')
    parser.add_argument('-i', '--input-file', action='store',
                                            type=str, metavar='inputFile',
                                            default='/global/cfs/cdirs/alice/mhwang/mypyjetty/pyjetty/pyjetty/alice_analysis/generation/sherpa/lund.gz',
                                            help='Path of ROOT file containing TTrees')
    parser.add_argument('-c', '--config-file', action='store',
                                            type=str, metavar='configFile',
                                            default='/global/cfs/cdirs/alice/mhwang/mypyjetty/pyjetty/pyjetty/alice_analysis/config/cEEC/sherpa3.yaml',
                                            help="Path of config file for analysis")
    parser.add_argument('-o', '--output-file', action='store',
                                            type=str, metavar='outputDir',
                                            default='./TestOutput',
                                            help='Output file for output to be written to')

    # Parse the arguments
    args = parser.parse_args()

    handler = logging.StreamHandler()

    # Create a formatter and set it for the handler
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(funcName)s - %(message)s'))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.info('Configuring...')
    logger.info(f"Input file: \'{args.input_file}\'")
    logger.info(f"Config file: \'{args.config_file}\'")
    logger.info(f"Ouput directory: '{args.output_file}'")

    # If invalid inputFile is given, exit
    if not os.path.exists(args.input_file):
        logger.critical(f"Input file '{args.input_file}' does not exist, exiting.")
        sys.exit(1)

    # If invalid configFile is given, exit
    if not os.path.exists(args.config_file):
        logger.critical(f"Config file '{args.config_file}' does not exist, exiting.")
        sys.exit(1)

    analysis = ProcessSherpa3(input_file=args.input_file, config_file=args.config_file, output_file=args.output_file)
    analysis.analyze()