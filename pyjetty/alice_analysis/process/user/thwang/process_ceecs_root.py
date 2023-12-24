#!/usr/bin/env python

from __future__ import print_function

import fastjet as fj
import fjcontrib
import fjext

import ROOT

import yaml
import argparse
import os
import numpy as np
import sys

from pyjetty.mputils import *

from heppy.pythiautils import configuration as pyconf
import pythia8
import pythiafjext
import pythiaext
import ecorrel

from pyjetty.alice_analysis.process.base import process_base

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)
# Automatically set Sumw2 when creating new histograms
ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()

def linbins(xmin, xmax, nbins):
  arr = np.linspace(xmin, xmax, nbins+1)
#   arr = array.array('f', lspace)
  return arr

def logbins(xmin, xmax, nbins):
  arr = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
#   arr = array.array('f', lspace)
  return arr

################################################################
class PythiaGenENC(process_base.ProcessBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, args=None, **kwargs):

        super(PythiaGenENC, self).__init__(
            input_file, config_file, output_dir, debug_level, **kwargs)

        # Call base class initialization
        process_base.ProcessBase.initialize_config(self)

        # Read config file
        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.jet_levels = config["jet_levels"] # levels = ["p", "h", "ch"]
        self.jetR_list = config["jetR"]
        self.RL_min, self.RL_max, self.RL_nbins = config["RL_binning"]
        self.pT_min, self.pT_max, self.pT_nbins = config["pT_binning"]
        self.Nconst_min, self.Nconst_max, self.Nconst_nbins = config["Nconst_binning"]

        self.nev = args.nev
        self.do_theory_check = config["do_theory_check"]

        # particle level - ALICE tracking restriction
        self.max_eta_hadron = config["eta_max"] # used to be explicitly 0.9
        self.min_det_trk_pT = config["min_det_trk_pT"]
        self.min_jet_trk_pT = config["min_jet_trk_pT"]
        self.min_jet_pT = config["min_jet_pT"]
        self.leading_jet_const_pT = config["leading_jet_const_pT"]

        # ENC settings
        self.dphi_cut = -9999
        self.deta_cut = -9999
        self.npoint = config["npoint"]
        self.npower = config["npower"]

        

    #---------------------------------------------------------------
    # Main processing function
    #---------------------------------------------------------------
    def pythia_parton_hadron(self, args):
 
        # Create ROOT TTree file for storing raw PYTHIA particle information
        outf_path = os.path.join(self.output_dir, args.tree_output_fname)
        outf = ROOT.TFile(outf_path, 'recreate')
        outf.cd()

        mycfg = []
        mycfg.append("HadronLevel:all=off") # Turning off hadronization
        pythia = pyconf.create_and_init_pythia_from_args(args, mycfg)

        # Initialize response histograms
        self.initialize_hist()

        # print the banner first
        fj.ClusterSequence.print_banner()
        print()

        self.init_jet_tools()
        self.calculate_events(pythia)
        pythia.stat()
        print()
        
        self.scale_print_final_info(pythia)

        outf.Write()
        outf.Close()

        self.save_output_objects()

    #---------------------------------------------------------------
    # Initialize histograms
    #---------------------------------------------------------------
    def initialize_hist(self):

        pt_bins = linbins(self.pT_min, self.pT_max, self.pT_nbins)
        RL_bins = logbins(self.RL_min, self.RL_max, self.RL_nbins)
        Nconst_bins = linbins(self.Nconst_min, self.Nconst_max, self.Nconst_nbins)
        self.hNevents = ROOT.TH1I("hNevents", 'Number accepted events (unscaled)', 2, -0.5, 1.5)
        self.jetpT_ch = ROOT.TH1I("h_jetpT_ch", 'jet p_{T} distribution:jet p_{T} (GeV):Counts', self.pT_nbins, pt_bins)
        self.jetpT_h = ROOT.TH1I("h_jetpT_h", 'jet p_{T} distribution:jet p_{T} (GeV):Counts', self.pT_nbins, pt_bins)

        def add_ENC_histogram(name):
            print(f"Initializing histogram {name}")
            h = ROOT.TH2D(name, name, self.pT_nbins, pt_bins, self.RL_nbins, RL_bins)
            setattr(self, name, h)
            getattr(self, hist_list_name).append(h)

        for jetR in self.jetR_list:

            # Store a list of all the histograms just so that we can rescale them later
            R_label = str(jetR).replace('.', '') + 'Scaled'
            hist_list_name = "hist_list_R%s" % str(jetR).replace('.', '')
            setattr(self, hist_list_name, [])


            for jet_level in self.jet_levels:
                # ENC histograms (jet level == part level)
                for ipoint in range(2, self.npoint+1):
                    add_ENC_histogram(f'h_ENC{ipoint}TR_JetPt_{jet_level}_R{R_label}_trk00')
                    add_ENC_histogram(f'h_ENC{ipoint}TR_JetPt_{jet_level}_R{R_label}_trk10')
                    add_ENC_histogram(f'h_ENC{ipoint}QQ_JetPt_{jet_level}_R{R_label}_trk10')
                    add_ENC_histogram(f'h_ENC{ipoint}PP_JetPt_{jet_level}_R{R_label}_trk10')
                    add_ENC_histogram(f'h_ENC{ipoint}PM_JetPt_{jet_level}_R{R_label}_trk10')
                    add_ENC_histogram(f'h_ENC{ipoint}MM_JetPt_{jet_level}_R{R_label}_trk10')
                    # add_ENC_histogram(f'h_ENC{ipoint}AL_JetPt_{jet_level}_R{R_label}_trk10')

                    # only save charge separation for pT>1GeV for now
                    # if jet_level == "ch":
                    #     add_ENC_histogram(f'h_ENC{ipoint}_JetPt_{jet_level}_R{R_label}_unlike_trk10')
                    #     add_ENC_histogram(f'h_ENC{ipoint}_JetPt_{jet_level}_R{R_label}_like_trk10')

                # Jet pt vs N constituents
                name = f'h_Nconst_JetPt_{jet_level}_R{R_label}_trk00'
                print('Initialize histogram', name)
                h = ROOT.TH2D(name, f"{name}:jet pT (GeV):Nconst", self.pT_nbins, pt_bins, self.Nconst_nbins, Nconst_bins)
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)

                name = f'h_Nconst_JetPt_{jet_level}_R{R_label}_trk10'
                print('Initialize histogram', name)
                h = ROOT.TH2D(name, f"{name}:jet pT (GeV):Nconst", self.pT_nbins, pt_bins, self.Nconst_nbins, Nconst_bins)
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)


    #---------------------------------------------------------------
    # Initiate jet defs, selectors, and sd (if required)
    #---------------------------------------------------------------
    def init_jet_tools(self):
        
        for jetR in self.jetR_list:
            jetR_str = str(jetR).replace('.', '')      
            
            # set up our jet definition and a jet selector
            jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
            setattr(self, f"jet_def_R{jetR_str}", jet_def)

            jet_selector = fj.SelectorPtMin(self.min_jet_pT) & fj.SelectorAbsEtaMax(self.max_eta_hadron - jetR) # FIX ME: use 5 or lower? use it on all ch, h, p jets?
            setattr(self, f"jet_selector_R{jetR_str}", jet_selector)
            print(jet_def)

        # pwarning('max eta for particles after hadronization set to', self.max_eta_hadron)
        track_selector_ch = fj.SelectorPtMin(self.min_det_trk_pT)

        setattr(self, "track_selector_ch", track_selector_ch)

        # pfc_selector1 = fj.SelectorPtMin(1.)/
        pfc_selector1 = fj.SelectorPtMin(self.min_jet_trk_pT)
        setattr(self, "pfc_def_10", pfc_selector1)

            

    #---------------------------------------------------------------
    # Calculate events and pass information on to jet finding
    #---------------------------------------------------------------
    def calculate_events(self, pythia):
        
        iev = 0  # Event loop count
        while iev < self.nev:
            if iev % 100 == 0:
                print('ievt',iev)

            if not pythia.next():
                continue

            self.event = pythia.event
            # print(self.event)

            self.parts_pythia_p = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal], 0, True) # final stable partons

            hstatus = pythia.forceHadronLevel()
            if not hstatus:
                continue

            # full particle level
            self.parts_pythia_h = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal], 0, True)

            # charged particle level
            self.parts_pythia_ch = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal, pythiafjext.kCharged], 0, True)

            # Some "accepted" events don't survive hadronization step -- keep track here
            self.hNevents.Fill(0)

            self.find_jets_fill_trees()

            iev += 1

    #---------------------------------------------------------------
    # Find jets, do matching between levels, and fill histograms & trees
    #---------------------------------------------------------------
    def find_jets_fill_trees(self):
        # Loop over jet radii
        for jetR in self.jetR_list:
            jetR_str = str(jetR).replace('.', '')
            jet_selector = getattr(self, f"jet_selector_R{jetR_str}")
            jet_def = getattr(self, f"jet_def_R{jetR_str}")
            track_selector_ch = getattr(self, "track_selector_ch")

            jets_p = fj.sorted_by_pt(jet_selector(jet_def(self.parts_pythia_p)))
            jets_h = fj.sorted_by_pt(jet_selector(jet_def(self.parts_pythia_h)))
            jets_ch = fj.sorted_by_pt(jet_selector(jet_def(track_selector_ch(self.parts_pythia_ch))))

            

            for jet_level in self.jet_levels:
                # Get the jets at different levels
                if jet_level == "p":
                    jets = jets_p
                if jet_level == "h":
                    jets = jets_h
                    target_jetpT_hist = self.jetpT_h
                if jet_level == "ch":
                    jets = jets_ch
                    target_jetpT_hist = self.jetpT_ch

                #-------------------------------------------------------------
                # loop over jets and fill EEC histograms with jet constituents
                for j in jets:
                    target_jetpT_hist.Fill(j.perp())
                    self.fill_jet_histograms(jet_level, j, f"{jetR_str}Scaled")

    #---------------------------------------------------------------
    # Form EEC using jet constituents
    #---------------------------------------------------------------
    def fill_jet_histograms(self, level, jet, R_label):
        # leading track selection
        if self.leading_jet_const_pT > 0:
            constituents = fj.sorted_by_pt(jet.constituents())
            if constituents[0].perp() < self.leading_jet_const_pT:
                return

        # fill EEC histograms for jet constituents
        pfc_selector1 = getattr(self, "pfc_def_10")

        # select all constituents with no cut
        _c_select0 = fj.vectorPJ()
        for c in jet.constituents():
            if self.do_theory_check:
                if pythiafjext.getPythia8Particle(c).charge()!=0:
                    _c_select0.push_back(c)
            else:
                _c_select0.push_back(c)
        cb0 = ecorrel.CorrelatorBuilder(_c_select0, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)

        # select constituents with 1 GeV cut
        _c_select1 = fj.vectorPJ()
        for c in pfc_selector1(jet.constituents()):
            if self.do_theory_check:
                if pythiafjext.getPythia8Particle(c).charge()!=0:
                    _c_select1.push_back(c)
            else:
                _c_select1.push_back(c)
        cb1 = ecorrel.CorrelatorBuilder(_c_select1, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)

        for ipoint in range(2, self.npoint+1):
            for index in range(cb0.correlator(ipoint).rs().size()):
                    getattr(self, f'h_ENC{ipoint}TR_JetPt_{level}_R{R_label}_trk00').Fill(jet.perp(), cb0.correlator(ipoint).rs()[index], cb0.correlator(ipoint).weights()[index])
            for index in range(cb1.correlator(ipoint).rs().size()):
                    getattr(self, f'h_ENC{ipoint}TR_JetPt_{level}_R{R_label}_trk10').Fill(jet.perp(), cb1.correlator(ipoint).rs()[index], cb1.correlator(ipoint).weights()[index])
            
        # if analyzing charged jet, separare different charge combinations
        if level == "ch":
            for ipoint in range(2, self.npoint+1):
                # only fill trk pt > 1 GeV here for now
                for index in range(cb1.correlator(ipoint).rs().size()):
                    part1 = int(cb1.correlator(ipoint).indices1()[index])
                    part2 = int(cb1.correlator(ipoint).indices2()[index])
                    c1 = pythiafjext.getPythia8Particle(_c_select1[part1]).charge()
                    c2 = pythiafjext.getPythia8Particle(_c_select1[part2]).charge()
                    if c1 > 0 and c2 > 0:
                        getattr(self, f'h_ENC{ipoint}PP_JetPt_{level}_R{R_label}_trk10').Fill(jet.perp(), cb1.correlator(ipoint).rs()[index], cb1.correlator(ipoint).weights()[index])
                    elif c1 < 0 and c2 < 0:
                        getattr(self, f'h_ENC{ipoint}MM_JetPt_{level}_R{R_label}_trk10').Fill(jet.perp(), cb1.correlator(ipoint).rs()[index], cb1.correlator(ipoint).weights()[index])
                    else: # one is positive, one is negative
                        getattr(self, f'h_ENC{ipoint}PM_JetPt_{level}_R{R_label}_trk10').Fill(jet.perp(), cb1.correlator(ipoint).rs()[index], cb1.correlator(ipoint).weights()[index])
                    
                    getattr(self, f'h_ENC{ipoint}QQ_JetPt_{level}_R{R_label}_trk10').Fill(jet.perp(), cb1.correlator(ipoint).rs()[index], c1 * c2 * cb1.correlator(ipoint).weights()[index])

                    # if pythiafjext.getPythia8Particle(c1).charge()*pythiafjext.getPythia8Particle(c2).charge() < 0:
                    #     # print("unlike-sign pair ",pythiafjext.getPythia8Particle(c1).id(),pythiafjext.getPythia8Particle(c2).id())
                    #     getattr(self, f'h_ENC{ipoint}_JetPt_{level}_R{R_label}_unlike_trk10').Fill(jet.perp(), cb1.correlator(ipoint).rs()[index], cb1.correlator(ipoint).weights()[index])
                    # else:
                    #     # print("likesign pair ",pythiafjext.getPythia8Particle(c1).id(),pythiafjext.getPythia8Particle(c2).id())
                    #     getattr(self, f'h_ENC{ipoint}_JetPt_{level}_R{R_label}_like_trk10').Fill(jet.perp(), cb1.correlator(ipoint).rs()[index], cb1.correlator(ipoint).weights()[index])

        getattr(self, f'h_Nconst_JetPt_{level}_R{R_label}_trk00').Fill(jet.perp(), len(_c_select0))
        getattr(self, f'h_Nconst_JetPt_{level}_R{R_label}_trk10').Fill(jet.perp(), len(_c_select1))
       
    #---------------------------------------------------------------
    # Initiate scaling of all histograms and print final simulation info
    #---------------------------------------------------------------
    def scale_print_final_info(self, pythia):
        # Scale all jet histograms by the appropriate factor from generated cross section and the number of accepted events
        scale_f = pythia.info.sigmaGen() / self.hNevents.GetBinContent(1)
        print(f"Scaling factor (sigma/nev) = {scale_f}")
        with open(f"{self.output_dir}/scales.txt", 'w') as f:
            f.write(f"{scale_f}")

        for jetR in self.jetR_list:
            hist_list_name = f"hist_list_R{str(jetR).replace('.', '')}"
            for h in getattr(self, hist_list_name):
                h.Scale(scale_f)

        print(f"N total final events:{self.hNevents.GetBinContent(1)} with {pythia.info.nAccepted() - self.hNevents.GetBinContent(1)} events rejected at hadronization step")
        self.hNevents.SetBinError(1, 0)

################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='charged EEC simulation with PYTHIA8',
                                     prog=os.path.basename(__file__))
    pyconf.add_standard_pythia_args(parser)
    # Could use --py-seed
    parser.add_argument('-o', '--output-dir', action='store', type=str, default='./', 
                        help='Output directory for generated ROOT file(s)')
    parser.add_argument('--tree-output-fname', default="AnalysisResults.root", type=str,
                        help="Filename for the (unscaled) generated particle ROOT TTree")
    parser.add_argument('-c', '--config_file', action='store', type=str, default='config/analysis_config.yaml',
                        help="Path of config file for observable configurations")

    args = parser.parse_args()

    # Have at least 1 event
    if args.nev < 1:
        args.nev = 1

    process = PythiaGenENC(config_file=args.config_file, output_dir=args.output_dir, args=args)
    process.pythia_parton_hadron(args)