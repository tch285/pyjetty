#!/usr/bin/env python3

"""
Tree class class to read a ROOT TTree of data track information
and produce TTrees of relevant particle, jet, and pair information

Author: Tucker Hwang (tucker_hwang@berkeley.edu)
"""

import argparse
import itertools
import logging
import os
import sys
from array import array

import fastjet as fj
import numpy as np
import ROOT
import yaml
from pyjetty.alice_analysis.process.user.substructure import process_data_base

logger = logging.getLogger(__name__)

ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()

################################################################
class Generator_Tree_Data_ENC(process_data_base.ProcessDataBase):
    # ---------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------
    def __init__(
        self,
        input_file="",
        config_file="",
        output_dir="",
        output_filename="",
        debug_level=0,
        **kwargs,
    ):
        # Initialize before so process_base correctly
        self.outputfilename = output_filename
        # Initialize base class
        super(Generator_Tree_Data_ENC, self).__init__(
            input_file, config_file, output_dir, debug_level, **kwargs
        )

        with open(self.config_file, "r") as stream:
            config = yaml.safe_load(stream)

        self.jet_branches = [
            (bname, btype)
            for bname, btype in (
                branch.split("/") for branch in config["jet_branches"]
            )
        ]
        self.part_branches = [
            (bname, btype)
            for bname, btype in (
                branch.split("/") for branch in config["part_branches"]
            )
        ]
        self.pair_branches = [
            (bname, btype)
            for bname, btype in (
                branch.split("/") for branch in config["pair_branches"]
            )
        ]

        self.tree2arr = {"I": "i", "F": "f", "D": "d", "L": "l", "O": "B", "B": "b"}

    # ---------------------------------------------------------------
    # Calculate pair distance of two fastjet particles
    # ---------------------------------------------------------------
    def calculate_distance(self, p0, p1):
        dphiabs = np.fabs(p0.phi() - p1.phi())
        dphi = dphiabs

        if dphiabs > np.pi:
            dphi = 2 * np.pi - dphiabs

        deta = p0.eta() - p1.eta()
        return np.sqrt(deta * deta + dphi * dphi)

    # ---------------------------------------------------------------
    # Calculate phistar distance of two fastjet particles
    # ---------------------------------------------------------------
    # def calc_phistar(self, p1, p2, q1, q2):
    # 	R = 1.1 # reference radius for TPC
    # 	Bz = 0.5
    # 	phi12 = p1.delta_phi_to(p2) # this calculates p2.phi() - p1.phi()
    # 	pt1 = p1.pt()
    # 	pt2 = p2.pt()
    # 	return phi12 + q1*np.arcsin(0.015*Bz*R/pt1) - q2*np.arcsin(0.015*Bz*R/pt2)
    def calc_phistar(self, p1, p2, q1, q2):
        R = 1.1  # reference radius for TPC
        Bz = -0.5  # extra minus
        dalpha = q1 * np.arcsin(-0.15 * Bz * R / p1.pt()) - q2 * np.arcsin(
            -0.15 * Bz * R / p2.pt()
        )

        return self.calculate_dphi(p1.phi(), p2.phi()) + dalpha

    def calculate_dphi(self, phi1, phi2):
        delta_phi = self.Phi_mpi_pi(phi1 - phi2)
        # if (delta_phi<-0.5*M_PI) delta_phi += 2*M_PI; // This should not be needed

        if delta_phi > np.pi or delta_phi < -np.pi:
            self.warning("Delta phi not inside desired range")

        return delta_phi

    def Phi_mpi_pi(self, dphi):
        while dphi >= np.pi:
            dphi -= np.pi * 2
        while dphi < -np.pi:
            dphi += np.pi * 2
        return dphi

    def calc_kt(self, p1, p2):
        # original formula below
        # return 0.5*np.sqrt( pow(p1.pt(),2)+pow(p1.pt(),2)+2*p1.pt()*p2.pt()*np.cos(p1.phi()-p2.phi()) )
        # fixed formula below
        return 0.5 * np.sqrt(
            pow(p1.pt(), 2)
            + pow(p2.pt(), 2)
            + 2 * p1.pt() * p2.pt() * np.cos(p1.phi() - p2.phi())
        )

    # ---------------------------------------------------------------
    # Initialize histograms
    # ---------------------------------------------------------------
    def initialize_user_output_objects(self):
        # store the ROOT TTree type names and the corresponding Python
        # array.array types in this dict
        for jetR in self.jetR_list:
            self.part_tree = ROOT.TTree("parts", "particles")
            self.jet_tree = ROOT.TTree("jets", f"jets (R={jetR})")
            self.pair_tree = ROOT.TTree("pairs", f"particle pairs (R={jetR})")

            self.add_branches(self.part_tree, self.part_branches, "part")
            self.add_branches(self.jet_tree, self.jet_branches, "jet")
            self.add_branches(self.pair_tree, self.pair_branches, "pair")

    def add_branches(self, tree, branches, prefix):
        for name, ttype in branches:
            atype = self.tree2arr[ttype]
            arr = array(atype, [0])
            setattr(self, f"{prefix}_{name}", tree.Branch(name, arr, f"{name}/{ttype}"))
            # setattr(self, name, arr)

    def analyze_event(self, fj_particles):
        self.event_number += 1
        if self.event_number > self.event_number_max:
            return
        if self.debug_level > 1:
            logger.debug(f"Event {self.event_number}:")

        self.fill_particle_trees(fj_particles)

        # Loop through jetR, and process event for each R
        for jetR in self.jetR_list:
            # Set jet definition and a jet selector
            jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
            # jet_selector = fj.SelectorPtMin(5.0) & fj.SelectorAbsRapMax(0.9 - jetR)
            jet_selector = fj.SelectorPtMin(5.0) & fj.SelectorAbsEtaMax(0.9 - jetR)
            logger.debug(f"Jet definition: {jet_def}")
            logger.debug(f"Jet selector: {jet_selector}")

            # Analyze
            # Find jets
            cs = fj.ClusterSequence(fj_particles, jet_def)
            jets = fj.sorted_by_pt(cs.inclusive_jets())
            jets_selected = jet_selector(jets)
            
            self.analyze_jets(jets_selected)

    def get_branch_type(self, name):
        return getattr(self, name).GetTitle().split("/")[-1]

    def fill_particle_trees(self, fj_particles):
        part_pT_arr = array(
            self.tree2arr[self.get_branch_type("part_pT")], [0.0]
        )
        getattr(self, "part_pT").SetAddress(part_pT_arr)
        part_eta_arr = array(
            self.tree2arr[self.get_branch_type("part_eta")], [0.0]
        )
        getattr(self, "part_eta").SetAddress(part_eta_arr)
        part_phi_arr = array(
            self.tree2arr[self.get_branch_type("part_phi")], [0.0]
        )
        getattr(self, "part_phi").SetAddress(part_phi_arr)
        part_q_arr = array(self.tree2arr[self.get_branch_type("part_q")], [0])
        getattr(self, "part_q").SetAddress(part_q_arr)

        for part in fj_particles:
            part_pT_arr[0] = part.pt()
            part_eta_arr[0] = part.eta()
            part_phi_arr[0] = part.phi()
            part_q_arr[0] = part.python_info().charge
            self.part_tree.Fill()

    def analyze_jets(self, jets):
        logger.debug(f"Number of jets: {len(jets)}")
        for jet in jets:        
            if not self.utils.is_det_jet_accepted(jet):
                continue

            self.fill_jet_trees(jet)
            self.fill_pair_trees(jet)

    def fill_jet_trees(self, jet):
        jet_pT_arr = array(self.tree2arr[self.get_branch_type("jet_pT")], [0.0])
        getattr(self, "jet_pT").SetAddress(jet_pT_arr)
        jet_nconst_arr = array(
            self.tree2arr[self.get_branch_type("jet_nconst")], [0]
        )
        getattr(self, "jet_nconst").SetAddress(jet_nconst_arr)
        jet_eta_arr = array(
            self.tree2arr[self.get_branch_type("jet_eta")], [0.0]
        )
        getattr(self, "jet_eta").SetAddress(jet_eta_arr)
        jet_phi_arr = array(
            self.tree2arr[self.get_branch_type("jet_phi")], [0.0]
        )
        getattr(self, "jet_phi").SetAddress(jet_phi_arr)

        jet_pT_arr[0] = jet.pt()
        # jet_eta_arr[0] = jet.rap()
        jet_eta_arr[0] = jet.eta()
        jet_phi_arr[0] = jet.phi()
        jet_nconst_arr[0] = len(jet.constituents())
        self.jet_tree.Fill()

    def fill_pair_trees(self, jet):
        jet_pT = jet.pt()

        pair_pT1_arr = array(
            self.tree2arr[self.get_branch_type("pair_pT1")], [0.0]
        )
        getattr(self, "pair_pT1").SetAddress(pair_pT1_arr)
        pair_pT2_arr = array(
            self.tree2arr[self.get_branch_type("pair_pT2")], [0.0]
        )
        getattr(self, "pair_pT2").SetAddress(pair_pT2_arr)
        pair_q1_arr = array(self.tree2arr[self.get_branch_type("pair_q1")], [0])
        getattr(self, "pair_q1").SetAddress(pair_q1_arr)
        pair_q2_arr = array(self.tree2arr[self.get_branch_type("pair_q2")], [0])
        getattr(self, "pair_q2").SetAddress(pair_q2_arr)

        pair_weight_arr = array(
            self.tree2arr[self.get_branch_type("pair_weight")], [0.0]
        )
        getattr(self, "pair_weight").SetAddress(pair_weight_arr)
        pair_RL_arr = array(
            self.tree2arr[self.get_branch_type("pair_RL")], [0.0]
        )
        getattr(self, "pair_RL").SetAddress(pair_RL_arr)
        pair_deta_arr = array(
            self.tree2arr[self.get_branch_type("pair_deta")], [0.0]
        )
        getattr(self, "pair_deta").SetAddress(pair_deta_arr)
        pair_dphi_arr = array(
            self.tree2arr[self.get_branch_type("pair_dphi")], [0.0]
        )
        getattr(self, "pair_dphi").SetAddress(pair_dphi_arr)

        pair_pTjet_arr = array(
            self.tree2arr[self.get_branch_type("pair_pTjet")], [jet_pT]
        )
        getattr(self, "pair_pTjet").SetAddress(pair_pTjet_arr)

        # for p1, p2 in itertools.product(jet.constituents(), repeat = 2):
        for p1, p2 in itertools.permutations(jet.constituents(), 2):
            pair_pT1_arr[0] = p1.pt()
            pair_pT2_arr[0] = p2.pt()
            pair_q1_arr[0] = p1.python_info().charge
            pair_q2_arr[0] = p2.python_info().charge
            pair_weight_arr[0] = p1.pt() * p2.pt() / (jet_pT * jet_pT)
            pair_RL_arr[0] = self.deltaR(p1, p2)
            pair_deta_arr[0] = p2.eta() - p1.eta()
            pair_dphi_arr[0] = p1.delta_phi_to(p2)

            self.pair_tree.Fill()


##################################################################
if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser(description="Process data for ENC trees.")
    parser.add_argument(
        "-i",
        "--input-file",
        action="store",
        type=str,
        metavar="input-file",
        default="AnalysisResults.root",
        help="Path of ROOT file containing particle and event TTrees",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        action="store",
        type=str,
        metavar="config-file",
        default="config/analysis_config.yaml",
        help="Path of config file for analysis",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        action="store",
        type=str,
        metavar="output-dir",
        default="./TestOutput",
        help="Output directory for output to be written to",
    )
    parser.add_argument(
        "-of",
        "--output-filename",
        action="store",
        type=str,
        metavar="output-filename",
        default="ceec_trees.root",
        help="Output filename for output to be written to",
    )

    # Parse the arguments
    args = parser.parse_args()

    handler = logging.StreamHandler()

    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(funcName)s - %(message)s"
    )
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.info("Configuring...")
    logger.info(f"Input file: '{args.input_file}'")
    logger.info(f"Config file: '{args.config_file}'")
    logger.info(f"Output directory: '{args.output_dir}'")
    logger.info(f"Output filename: '{args.output_filename}'")

    # If invalid inputFile is given, exit
    if not os.path.exists(args.input_file):
        logger.critical(f"Input file '{args.input_file}' does not exist, exiting.")
        sys.exit(1)

    # If invalid configFile is given, exit
    if not os.path.exists(args.config_file):
        logger.critical(f"Config file '{args.config_file}' does not exist, exiting.")
        sys.exit(1)

    # ROOT.EnableImplicitMT()
    # n_threads = ROOT.ROOT.GetThreadPoolSize()
    # print(f"Number of threads: {n_threads}")
    # n_slots = df.GetNSlots()
    # print(f"Number of processing slots: {n_slots}")

    analysis = Generator_Tree_Data_ENC(
        input_file=args.input_file,
        config_file=args.config_file,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
    )
    analysis.process_data()
