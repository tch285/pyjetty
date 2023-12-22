#!/usr/bin/env python

from __future__ import print_function

import fastjet as fj
import fjcontrib
import fjext

import ROOT

import tqdm
import yaml
import copy
from matplotlib import pyplot as plt
import argparse
import os
import array
import numpy as np
import math
import sys
import uproot

from pyjetty.mputils import *

from heppy.pythiautils import configuration as pyconf
import pythia8
import pythiafjext
import pythiaext
import ecorrel

from pyjetty.alice_analysis.process.base import process_base

def linbins(xmin, xmax, nbins):
    lspace = np.linspace(xmin, xmax, nbins+1)
    arr = array.array('f', lspace)
    centers = (arr[:-1] + arr[1:]) / 2
    return arr, centers

def logbins(xmin, xmax, nbins):
    arr = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
    centers = np.sqrt(arr[:-1] * arr[1:])
    lowerr = centers - arr[:-1]
    uperr = arr[1:] - centers
    return arr, centers, lowerr, uperr

class CEEC(process_base.ProcessBase):
    def __init__(self, input='', config_file='', output_dir='', debug_lvl=0, clargs=None, **kwargs):
        with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)
        

        super(CEEC, self).__init__(input, config_file, output_dir, debug_lvl, **kwargs)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        process_base.ProcessBase.initialize_config(self)

        # self.jet_levels = config["jet_levels"] # levels = ["p", "h", "ch"]
        # self.jetR_list = config["jetR"]
        self.jetR = 0.4
        self.jet_level="ch"
        self.nev = max(clargs.nev, 1)
        self.do_theory_check = config['do_theory_check']

        # particle level - ALICE tracking restriction
        self.max_eta_hadron = 0.9
        self.leading_pt = 0

        # ENC settings
        self.dphi_cut = -9999
        self.deta_cut = -9999
        self.npoint = 2
        self.npower = 1
        self.eec_type = clargs.eec_type
    
    def run(self, clargs):
        # outf_path = os.path.join(self.output_dir, clargs.tree_output_fname)
        # outf = ROOT.TFile(outf_path, 'recreate')
        # outf.cd()

        pythia_config=[]
        pythia_config.append("HadronLevel:all=off") # Turning off hadronization
        pythia = pyconf.create_and_init_pythia_from_args(clargs, pythia_config)

        self.init_hists()

        self.init_jet_tools()
        self.calculate_events(pythia)
        # pythia.stat()

        self.scale(pythia)

        # outf.Write()
        # outf.Close()

        self.save_output_objects()
    
    def init_hists(self):
        nbins=50
        self.RL_bins, self.RL_centers, self.lowerr, self.uperr = logbins(1e-4, 1, nbins)

        self.histTR, _ = np.histogram(a=[0], bins=self.RL_bins, weights=[0])
        self.histQQ, _ = np.histogram(a=[0], bins=self.RL_bins, weights=[0])
        self.histPM, _ = np.histogram(a=[0], bins=self.RL_bins, weights=[0])
        self.histPP, _ = np.histogram(a=[0], bins=self.RL_bins, weights=[0])
        self.histMM, _ = np.histogram(a=[0], bins=self.RL_bins, weights=[0])
    
    def init_jet_tools(self):
        self.jet_def = fj.JetDefinition(fj.antikt_algorithm, self.jetR)
        self.track_selector_ch = fj.SelectorPtMin(0.15)
        self.pfc_selector1 = fj.SelectorPtMin(1.)
        self.jet_selector = fj.SelectorPtMin(5) & fj.SelectorAbsEtaMax(self.max_eta_hadron - self.jetR)

    def calculate_events(self, pythia):
        self.forceHadronSurvive = 0
        for iev in range(self.nev):
            if not pythia.next():
                continue
            self.event = pythia.event
            # print(self.event)

            self.parts_pythia_p = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal], 0, True) # final stable partons

            hstatus = pythia.forceHadronLevel()
            if not hstatus:
                continue
            self.forceHadronSurvive += 1

            # full particle level
            self.parts_pythia_h = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal], 0, True)

            # charged particle level
            self.parts_pythia_ch = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal, pythiafjext.kCharged], 0, True)

            self.find_jets_fill_trees()
        
        print(f"Total events: {self.nev}")
        print(f"nAccepted events: {pythia.info.nAccepted()}")
        print(f"forceHadronSurvive events: {self.forceHadronSurvive}")

    def find_jets_fill_trees(self):
        jets_p = fj.sorted_by_pt(self.jet_selector(self.jet_def(self.parts_pythia_p)))
        jets_h = fj.sorted_by_pt(self.jet_selector(self.jet_def(self.parts_pythia_h)))
        jets_ch = fj.sorted_by_pt(self.jet_selector(self.jet_def(self.track_selector_ch(self.parts_pythia_ch))))

        # R_label = str(self.jetR).replace('.', '') + 'Scaled'

        if self.jet_level == "p":
            jets = jets_p
        if self.jet_level == "h":
            jets = jets_h
        if self.jet_level == "ch":
            jets = jets_ch

        #-------------------------------------------------------------
        # loop over jets and fill EEC histograms with jet constituents
        for j in jets:
            self.fill_jet_histograms(self.jet_level, j, self.jetR)

    def fill_jet_histograms(self, level, jet, jetR):
        if self.leading_pt > 0:
            constituents = fj.sorted_by_pt(jet.constituents())
            if constituents[0].perp() < self.leading_pt:
                return
        
        _c_select0 = fj.vectorPJ()
        for c in jet.constituents():
            if self.do_theory_check:
                if pythiafjext.getPythia8Particle(c).charge()!=0:
                    _c_select0.push_back(c)
            else:
                _c_select0.push_back(c)
        # cb0 = ecorrel.CorrelatorBuilder(_c_select0, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)
        cb0 = ecorrel.CorrelatorBuilder(_c_select0, self.npower, self.npoint)

        # select constituents with 1 GeV cut
        _c_select1 = fj.vectorPJ()
        for c in self.pfc_selector1(jet.constituents()):
            if self.do_theory_check:
                if pythiafjext.getPythia8Particle(c).charge()!=0:
                    _c_select1.push_back(c)
            else:
                _c_select1.push_back(c)
        # cb1 = ecorrel.CorrelatorBuilder(_c_select1, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)
        cb1 = ecorrel.CorrelatorBuilder(_c_select1, self.npower, self.npoint)
        for ipoint in range(2, self.npoint+1):
            assert cb0.correlator(ipoint).rs().size() == cb0.correlator(ipoint).weights().size()
            assert cb1.correlator(ipoint).rs().size() == cb1.correlator(ipoint).weights().size()
            print("the type of rs() is ", type(cb1.correlator(ipoint).rs()))
            self.calc_mod_weights(_c_select1, cb1.correlator(ipoint))
            self.histTR += np.histogram(cb1.correlator(ipoint).rs(), self.RL_bins, self.TR_weights)[0]
            self.histQQ += np.histogram(cb1.correlator(ipoint).rs(), self.RL_bins, self.QQ_weights)[0]
            self.histPM += np.histogram(cb1.correlator(ipoint).rs(), self.RL_bins, self.PM_weights)[0]
            self.histPP += np.histogram(cb1.correlator(ipoint).rs(), self.RL_bins, self.PP_weights)[0]
            self.histMM += np.histogram(cb1.correlator(ipoint).rs(), self.RL_bins, self.MM_weights)[0]
    
    def calc_mod_weights(self, const, corr):
        TR_weights=np.zeros(corr.weights().size())
        QQ_weights=np.zeros(corr.weights().size())
        PM_weights=np.zeros(corr.weights().size())
        PP_weights=np.zeros(corr.weights().size())
        MM_weights=np.zeros(corr.weights().size())
        for i in range(corr.rs().size()):
            part1 = int(corr.indices1()[i])
            part2 = int(corr.indices2()[i])
            c1 = const[part1]
            c2 = const[part2]
            charge_product = pythiafjext.getPythia8Particle(c1).charge() * pythiafjext.getPythia8Particle(c2).charge()
            TR_weights[i] = corr.weights()[i] if charge_product != 0 else 0
            QQ_weights[i] = corr.weights()[i] * charge_product
            PM_weights[i] = corr.weights()[i] if charge_product < 0 else 0    
            PP_weights[i] = corr.weights()[i] if c1 > 0 and c2 > 0 else 0    
            MM_weights[i] = corr.weights()[i] if c1 < 0 and c2 < 0 else 0    
    
    def scale(self, pythia):
        # Scale all jet histograms by the appropriate factor from generated cross section and the number of accepted events
        scale_f = pythia.info.sigmaGen() / self.hNevents.GetBinContent(1)
        print("scaling factor is",scale_f)

        self.histTR *= scale_f
        self.histQQ *= scale_f
        self.histPM *= scale_f
        self.histPP *= scale_f
        self.histMM *= scale_f

    def save_output_objects(self):
        # TODO: calculate y-errors
        # self.histTR_err = 
        # self.histQQ
        # self.histPM
        # self.histPP
        # self.histMM


        fig1, axs1 = plt.subplots(1, 3, sharey='row')
        axs1[0].errorbar(self.RL_centers, self.histTR, [self.lowerr, self.uperr], None, 'ro')
        axs1[0].errorbar(self.RL_centers, self.histQQ, [self.lowerr, self.uperr], None, 'ro')
        axs1[0].errorbar(self.RL_centers, self.histPM, [self.lowerr, self.uperr], None, 'ro')
        axs1.set_yscale('log')
        fig2, axs2 = plt.subplots(1, 3, sharey='row')
        axs1[0].errorbar(self.RL_centers, self.histPP, [self.lowerr, self.uperr], None, 'ro')
        axs1[0].errorbar(self.RL_centers, self.histMM, [self.lowerr, self.uperr], None, 'ro')
        axs2.set_yscale('log')

        fig1.savefig("pic1.png")
        fig2.savefig("pic2.png")











dir_path = os.path.dirname(os.path.realpath(__file__))
results_path = f"{dir_path}/results"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PYTHIA8+FASTJET for charged EECs',
                                     prog=os.path.basename(__file__))
    pyconf.add_standard_pythia_args(parser)
    # Could use --py-seed
    parser.add_argument('-o', '--output-dir', action='store', type=str, default='./', 
                        help='Output directory for generated ROOT file(s)')
    parser.add_argument('--tree-output-fname', default="AnalysisResults.root", type=str,
                        help="Filename for the (unscaled) generated particle ROOT TTree")
    parser.add_argument('-c', '--config-file', action='store', type=str, default='config/analysis_config.yaml',
                        help="Path of config file for observable configurations")
    parser.add_argument('--eec-type', action='store', default="q", type=str,
                        help="Type of EEC to calculate: 'qq' for E_Q, 'pm' for E+E-, 'pp' for E+E+, 'mm' for E-E-, 'tr' for E_trE_tr")
    clargs = parser.parse_args()

    process = CEEC(config_file=clargs.config_file, output_dir=clargs.output_dir, clargs=clargs)
    process.run(clargs)


class Histogram1D:
    """Class to efficiently and easily fill, store, and plot histograms.

    Contains conversions to ROOT because pyroot sux
    """
    def __init__(self, data = None, weights = None, nbins = 50, bin_min = 1e-4, bin_max = 1, binning = 'log', title = "") -> None:
        self.nbins = nbins
        self.binning = binning
        self.bin_min = bin_min
        self.bin_max = bin_max
        self._create_bins()
        if data is None:
            self.bin_contents = np.zeros(self.nbins)
            if weights is not None:
                print("Weights passed but no data, weights ignored.")
        elif isinstance(data, (int, float, np.ndarray, list)):
            assert len(data) == len(weights)
            self.bin_contents = np.histogram(data, bins = self.bin_edges, weights = weights)[0]
        else:
            raise TypeError("`bin_info` is not an acceptable type!")
        self.yerr = np.zeros(self.nbins) # no up or down error since it's the same both ways
        self.title = title
    
    def __add__(self, hist):
        self.bin_contents += hist.bin_contents
        return self
    
    def __mul__(self, f):
        assert isinstance(f, (float, int))
        self.yerr *= abs(f) # REVIEW: not sure if this behavior is desired or not actually...
        self.bin_contents *= f
        return self
    
    def __len__(self): return self.nbins
    def __getitem__(self, idx): return self.bin_contents[idx]
    def __setitem__(self, idx, val): self.bin_contents[idx] = val
    def __repr__(self): return f"{self.title}:{self.xtitle}:{self.ytitle}"
    def __truediv__(self, f): return self * (1 / f)
    def __sub__(self, hist): return self + (hist * -1)

    def _create_bins(self) -> None:
        if self.binning in ["linear", "lin", 1]:
            self.bin_edges, step = np.linspace(self.bin_min, self.bin_max, self.nbins + 1, endpoint = True, retstep = True)
            self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
            self.xlowerr = self.xuperr = np.full(self.nbins, step / 2)
        elif self.binning in ['logarithmic', 'log', 2]:
            self.bin_edges = np.logspace(np.log10(self.bin_min), np.log10(self.bin_max), self.nbins + 1)
            self.bin_centers = np.sqrt(self.bin_edges[:-1] * self.bin_edges[1:])
            self.xlowerr = self.bin_centers - self.bin_edges[:-1]
            self.xuperr = self.bin_edges[1:] - self.bin_centers
        else:
            raise ValueError("Binning option is not recognized: must be 'lin', 'log', 1, 'logarithmic', 'log', 2.")
    
    def fill(self, vals, weights):
        assert vals.size() == weights.size() # REVIEW: size? len? I think in principle len is probably the most robust option
        self.bin_contents += np.histogram(vals, bins = self.bin_edges, weights = weights)[0]

    def calc_error(self):
        self.yerr = np.sqrt(self.bin_contents)
    
    def set_bin_error(self, bin, err):
        self.yerr[bin] = err
        
    def normalize(self):
        self.scale(self.bin_contents.sum()) # FIXME: rewrite in terms of __mul__

    def toROOT(self, name): # try to avoid doing this until the very end
        self.set_error() # just as a last check
        hist = ROOT.TH1F(name, self.title, self.nbins, self.bin_edges)
        hist.GetXaxis().SetTitle(self.xtitle)
        hist.GetYaxis().SetTitle(self.ytitle)

        for i, freq in enumerate(self.bin_contents):
            hist.SetBinContent(i+1,freq)
            hist.SetBinError(i+1, self.yerr[i])
    
    def save(self, filename: str, show: bool):
        fig, ax = plt.subplots()
        ax.errorbar(self.bin_centers, self.bin_contents, self.yerr, [self.xlowerr, self.xuperr], 'ro')
        try:
            ax.set_title(self.title.split(":")[0])
            ax.set_xlabel(self.title.split(":")[1])
            ax.set_ylabel(self.title.split(":")[2])
        except IndexError:
            pass # HACK: not sure if this actually works; will it actually set the successful lines if it fails at L360?

        if self.binning in ['logarithmic', 'log', 2]:
            ax.set_yscale('log')

        fig.savefig(f"{results_path}/{filename}.png")
        if show:
            plt.show()

        