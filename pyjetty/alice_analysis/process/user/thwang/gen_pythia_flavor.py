#!/usr/bin/env python

from __future__ import print_function

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
import ecorrel

from pyjetty.mputils import *
from heppy.pythiautils import configuration as pyconf
from pyjetty.alice_analysis.process.base import process_base
import cEEC_utils.gen_utils as gutils

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)
# Automatically set Sumw2 when creating new histograms
ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()

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
        self.jetRs = config["jetR"]
        self.RL_min, self.RL_max, self.RL_nbins = config["RL_binning"]
        self.pT_min, self.pT_max, self.pT_nbins = config["pT_binning"]
        self.Nconst_min, self.Nconst_max, self.Nconst_nbins = config["Nconst_binning"]
        self.init_parton_ids = config["init_parton_ids"]
        self.init_parton_ids_nozero = [id for id in self.init_parton_ids if id != 0]

        self.nev = args.nev
        self.do_theory_check = config["do_theory_check"]
        self.jet_matching_distance = config["jet_matching_distance"]

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
    def generate(self, args):
        # Create ROOT TTree file for storing raw PYTHIA particle information
        outf_path = os.path.join(self.output_dir, args.tree_output_fname)
        outf = ROOT.TFile(outf_path, 'recreate')
        outf.cd()

        mycfg = ["HadronLevel:all=off"] # Turning off hadronization
        pythia = pyconf.create_and_init_pythia_from_args(args, mycfg)

        # Initialize response histograms
        self.prepare()

        # print the banner first
        fj.ClusterSequence.print_banner()
        print()

        self.simulate(pythia)
        pythia.stat()
        print()
        
        self.finalize(pythia)

        outf.Write()
        outf.Close()

    #---------------------------------------------------------------
    # Prepare for simulation (initializing histograms, creating jet defs and selectors)
    #---------------------------------------------------------------
    def prepare(self):
        pt_bins, _ = gutils.linbins(self.pT_min, self.pT_max, self.pT_nbins)
        RL_bins, _, _, _, _ = gutils.logbins(self.RL_min, self.RL_max, self.RL_nbins)
        self.hNevents = ROOT.TH1I("hNevents", 'Number accepted events (unscaled)', 2, -0.5, 1.5)

        # dictionary for storing histograms
        self.hists = {}

        self.hists['ENC'] = {}
        self.hists['jetpT'] = {}

        for jet_level in self.jet_levels:
            self.hists['ENC'][jet_level] = {}
            self.hists['jetpT'][jet_level] = {}
            for jetR in self.jetRs:
                R_label = str(jetR).replace('.', '') + '_sc'
                # self.hists['jetpT'][jet_level][jetR] = ROOT.TH1D(f"h_jetpT_{jet_level}_R{R_label}", 'jet p_{T} distribution;jet p_{T} (GeV);Counts', self.pT_nbins, pt_bins)
                self.hists['jetpT'][jet_level][jetR] = {}
                self.hists['ENC'][jet_level][jetR] = {}
                for id in self.init_parton_ids:
                    self.hists['jetpT'][jet_level][jetR][id] = ROOT.TH1D(f"h_jetpT_{jet_level}_R{R_label}_{id}", f'jet p_{{T}} distribution ({id});jet p_{{T}} (GeV);Counts', self.pT_nbins, pt_bins)
                    self.hists['ENC'][jet_level][jetR][id] = {}
                    for ipoint in range(2, self.npoint + 1):
                        self.hists['ENC'][jet_level][jetR][id][ipoint] = {}
                        for cEEC_type in gutils.cEEC_types:
                            self.hists['ENC'][jet_level][jetR][id][ipoint][cEEC_type] = ROOT.TH2D(f'h_ENC{ipoint}_{cEEC_type}_{id}_jetpT_{jet_level}_R{R_label}_1', f'ENC{ipoint}{cEEC_type} ({id});jet p_{{T}} (GeV);R_{{L}}', self.pT_nbins, pt_bins, self.RL_nbins, RL_bins)
                        if ipoint == 2:
                            self.hists['ENC'][jet_level][jetR][id][ipoint]['PM'] = ROOT.TH2D(f'h_ENC{ipoint}_PM_{id}_jetpT_{jet_level}_R{R_label}_1', f'ENC{ipoint}PM ({id});jet p_{{T}} (GeV);R_{{L}}', self.pT_nbins, pt_bins, self.RL_nbins, RL_bins)


        self.jet_def = {}
        self.jet_selector = {}
        for jetR in self.jetRs:
            # set up our jet definition and a jet selector
            jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
            self.jet_def[jetR] = jet_def

            jet_selector = fj.SelectorPtMin(self.min_jet_pT) & fj.SelectorAbsEtaMax(self.max_eta_hadron - jetR) # use 5 or lower? use it on all ch, h, p jets?
            self.jet_selector[jetR] = jet_selector
            print(jet_def)

        self.det_track_selector = fj.SelectorPtMin(self.min_det_trk_pT)
        self.jet_track_selector = fj.SelectorPtMin(self.min_jet_trk_pT)

    #---------------------------------------------------------------
    # Calculate events and pass information on to jet finding
    #---------------------------------------------------------------
    def simulate(self, pythia):
        iev = 0
        while iev < self.nev:
            if iev % 100 == 0:
                print('ievt',iev)

            if not pythia.next():
                continue

            self.event = pythia.event
            # print(self.event)
            leading_parton1 = fj.PseudoJet(pythia.event[5].px(),pythia.event[5].py(),pythia.event[5].pz(),pythia.event[5].e())
            leading_parton1.set_user_index(pythia.event[5].idAbs())
            leading_parton2 = fj.PseudoJet(pythia.event[6].px(),pythia.event[6].py(),pythia.event[6].pz(),pythia.event[6].e())
            leading_parton2.set_user_index(pythia.event[6].idAbs())
            # save index of the leading parton inside this event
            self.parton_parents = [leading_parton1, leading_parton2]

            
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

            self.collect_ENC_data()
            iev += 1

    #---------------------------------------------------------------
    # Find jets, do matching between levels, and fill histograms & trees
    #---------------------------------------------------------------
    def collect_ENC_data(self):
        # Loop over jet radii
        for jetR in self.jetRs:
            jet_selector = self.jet_selector[jetR]
            jet_def = self.jet_def[jetR]

            jets_p = fj.sorted_by_pt(jet_selector(jet_def(self.parts_pythia_p)))
            jets_h = fj.sorted_by_pt(jet_selector(jet_def(self.parts_pythia_h)))
            jets_ch = fj.sorted_by_pt(jet_selector(jet_def(self.det_track_selector(self.parts_pythia_ch))))

            for jet_level in self.jet_levels:
                # Get the jets at different levels
                if jet_level == "p":
                    jets = jets_p
                elif jet_level == "h":
                    jets = jets_h
                    # target_jetpT_hist = self.hists['jetpT'][jet_level][jetR]
                elif jet_level == "ch":
                    jets = jets_ch
                    # target_jetpT_hist = self.hists['jetpT'][jet_level][jetR]

                #-------------------------------------------------------------
                # loop over jets and fill EEC histograms with jet constituents
                for j in jets:
                    self._fill_hists(jet_level, j, jetR)

    #---------------------------------------------------------------
    # Calculate ENCs using jet constituents
    #---------------------------------------------------------------
    def _fill_hists(self, jet_level, jet, jetR):
        # ignore jet if constituents are all below cutoff
        if self.leading_jet_const_pT > 0:
            constituents = fj.sorted_by_pt(jet.constituents())
            if constituents[0].perp() < self.leading_jet_const_pT:
                return

        # select all constituents with no cut
        # _c_select0 = fj.vectorPJ()
        # for c in jet.constituents():
        #     if self.do_theory_check:
        #         if pythiafjext.getPythia8Particle(c).charge()!=0:
        #             _c_select0.push_back(c)
        #     else:
        #         _c_select0.push_back(c)
        # cb0 = ecorrel.CorrelatorBuilder(_c_select0, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)
            
        self.hists['jetpT'][jet_level][jetR][0].Fill(jet.perp())
        matched_parton_parents = []
        for parton_parent in self.parton_parents:
            if parton_parent.perp()/jet.perp() < 0.1 or parton_parent.perp()/jet.perp() > 10:
                continue
            if self._is_geo_matched(jet, parton_parent, jetR) and parton_parent.user_index() in self.init_parton_ids_nozero:
                matched_parton_parents.append(parton_parent.user_index())
        
        id = 0
        if len(matched_parton_parents) == 1: # accept if there is one match only (NB: but may be used multiple times)
            id = matched_parton_parents[0]
            self.hists['jetpT'][jet_level][jetR][id].Fill(jet.perp())
        elif len(matched_parton_parents) == 2:
            print("double match, doing nothing")
        # else:
            # print("no match, doing nothing")
        # print(id)



        # select constituents with 1 GeV cut
        jet_const = fj.vectorPJ()
        for c in self.jet_track_selector(jet.constituents()):
            if self.do_theory_check:
                if pythiafjext.getPythia8Particle(c).charge()!=0:
                    jet_const.push_back(c)
            else:
                jet_const.push_back(c)
        cb1 = ecorrel.CorrelatorBuilder(jet_const, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)
        for ipoint in range(2, self.npoint+1):
            for index in range(cb1.correlator(ipoint).rs().size()):
                self.hists['ENC'][jet_level][jetR][0][ipoint]['T'].Fill(jet.perp(), cb1.correlator(ipoint).rs()[index], cb1.correlator(ipoint).weights()[index])
                if id != 0:
                    self.hists['ENC'][jet_level][jetR][id][ipoint]['T'].Fill(jet.perp(), cb1.correlator(ipoint).rs()[index], cb1.correlator(ipoint).weights()[index])

        
        if jet_level == "ch":
            for ipoint in range(2, self.npoint+1):
                # only fill trk pt > 1 GeV here for now
                for indices, RL, weight in zip(cb1.correlator(ipoint).indices(), cb1.correlator(ipoint).rs(), cb1.correlator(ipoint).weights()):
                    charges = np.empty(ipoint, np.int8) # NOTE:? dtype might cause issues later
                    for i in range(ipoint):
                        charges[i] = pythiafjext.getPythia8Particle(jet_const[indices[i]]).charge()
                    if np.all(charges > 0):
                        self.hists['ENC'][jet_level][jetR][0][ipoint]['P'].Fill(jet.perp(), RL, weight)
                        if id != 0:
                            self.hists['ENC'][jet_level][jetR][id][ipoint]['P'].Fill(jet.perp(), RL, weight)

                    elif np.all(charges < 0):
                        self.hists['ENC'][jet_level][jetR][0][ipoint]['M'].Fill(jet.perp(), RL, weight)
                        if id != 0:
                            self.hists['ENC'][jet_level][jetR][id][ipoint]['M'].Fill(jet.perp(), RL, weight)
                    else:
                        if ipoint == 2:
                            self.hists['ENC'][jet_level][jetR][0][ipoint]['PM'].Fill(jet.perp(), RL, weight)
                            if id != 0:
                                self.hists['ENC'][jet_level][jetR][id][ipoint]['PM'].Fill(jet.perp(), RL, weight)
                    self.hists['ENC'][jet_level][jetR][0][ipoint]['Q'].Fill(jet.perp(), RL, np.prod(charges) * weight)
                    if id != 0:
                        self.hists['ENC'][jet_level][jetR][id][ipoint]['Q'].Fill(jet.perp(), RL, np.prod(charges) * weight)
    
    def _is_geo_matched(self, jet1, jet2, jetR):
        deltaR = jet1.delta_R(jet2)
        # Add a matching candidate to the list if it is within the geometrical cut
        if deltaR < self.jet_matching_distance * jetR:
            return True
        else:
            return False


    #---------------------------------------------------------------
    # Scale of all histograms and print final simulation info
    #---------------------------------------------------------------
    def finalize(self, pythia):
        # Scale all jet histograms by the appropriate factor from generated cross section and the number of accepted events
        scale_f = pythia.info.sigmaGen() / self.hNevents.GetBinContent(1)
        print(f"Scaling factor (sigma/nev) = {scale_f}")
        with open(f"{self.output_dir}/scales.txt", 'w') as f:
            f.write(f"{scale_f}")

        self._scale_hists(self.hists['ENC'], scale_f)

        print(f"N total final events:{self.hNevents.GetBinContent(1)} with {pythia.info.nAccepted() - self.hNevents.GetBinContent(1)} events rejected at hadronization step")
        self.hNevents.SetBinError(1, 0)
    
    def _scale_hists(self, value, scale):
        # need this funky way to get the ROOT histograms - iterate through dictionary values until you find a non-dictionary
        if not isinstance(value, dict):
            value.Scale(scale)
        else:
            for val in value.values():
                self._scale_hists(val, scale)

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
    process.generate(args)