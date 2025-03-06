#!/usr/bin/env python3

"""
    Analysis class to read a ROOT TTree of track information
    and do jet-finding, and save basic histograms.
    
    Author: James Mulligan (james.mulligan@berkeley.edu)
"""

# General
import os
import sys
import argparse
import logging
import itertools
# import sys

# Data analysis and plotting
import ROOT
import yaml
import numpy as np
from array import array

# Fastjet via python (from external library heppy)
import fastjet as fj

# Base class
from pyjetty.alice_analysis.process.user.substructure import process_data_base_pPb
from pyjetty.alice_analysis.process.base import jet_info

def linbins(xmin, xmax, nbins):
    lspace = np.linspace(xmin, xmax, nbins+1)
    arr = array.array('f', lspace)
    return arr

def logbins(xmin, xmax, nbins):
    lspace = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
    arr = array.array('f', lspace)
    return arr

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

logger = logging.getLogger(__name__)

################################################################
class Generator_Tree_Data_ENC(process_data_base_pPb.ProcessDataBase):
    def __init__(self, input_file="", config_file="", output_dir="", output_filename="", debug_level=0, **kwargs):
        # Initialize base class
        self.outputfilename = output_filename
        super(Generator_Tree_Data_ENC, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)

        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        self.event_branches = [
            (bname, btype)
            for bname, btype in (
                branch.split("/") for branch in config["event_branches"]
            )
        ]
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
        self.prep()

    def prep(self):
        self.tree2arr = {"I": "i", "F": "f", "D": "d", "L": "l", "O": "B", "B": "b"}
        self.jet_def = {jetR: fj.JetDefinition(fj.antikt_algorithm, jetR) for jetR in self.jetR_list}
        self.jet_selector = {jetR: fj.SelectorPtMin(5.0) & fj.SelectorAbsEtaMax(0.9 - jetR) for jetR in self.jetR_list}
        self.Cjet_selector = {jetR: fj.SelectorAbsEtaMax(0.9 - jetR) & (~fj.SelectorIsPureGhost()) for jetR in self.jetR_list}

    def initialize_user_output_objects(self):
        # store the ROOT TTree type names and the corresponding Python
        # array.array types in this dict
        for jetR in self.jetR_list:
            self.event_tree = ROOT.TTree("events", "events")
            self.part_tree = ROOT.TTree("parts", "particles")
            self.jet_tree = ROOT.TTree("jets", f"jets (R={jetR})")
            self.pair_tree = ROOT.TTree("pairs", f"particle pairs (R={jetR})")

            self.add_branches(self.event_tree, self.event_branches, "event")
            self.add_branches(self.part_tree, self.part_branches, "part")
            self.add_branches(self.jet_tree, self.jet_branches, "jet")
            self.add_branches(self.pair_tree, self.pair_branches, "pair")

    def add_branches(self, tree, branches, prefix):
        for name, ttype in branches:
            atype = self.tree2arr[ttype]
            arr = array(atype, [0])
            setattr(self, f"{prefix}_{name}", tree.Branch(name, arr, f"{name}/{ttype}"))

    def analyze_event(self, fj_particles, mult):
        if len(fj_particles) > 1:
            if np.abs(fj_particles[0].pt() - fj_particles[1].pt()) <  1e-10:
                logger.warning('Duplicate particles may be present:')
                logger.warning([p.user_index() for p in fj_particles])
                logger.warning([p.pt() for p in fj_particles])
        # [part.set_user_index(0) for part in fj_particles]
        # logger.critical([part.user_index() for part in fj_particles])
        self.fill_particle_trees(fj_particles)

        # Loop through jetR, and process event for each R
        for jetR in self.jetR_list:
            jet_def = self.jet_def[jetR]
            jet_selector = self.jet_selector[jetR]

            if self.do_median_subtraction:
                csa_medsub = fj.ClusterSequenceArea(fj_particles, self.jet_def_medsub[jetR], fj.AreaDefinition(fj.active_area_explicit_ghosts))
                self.median_subtractor[jetR].set_cluster_sequence(csa_medsub)
                rho = self.median_subtractor[jetR].rho()

                Cjet_selector = self.Cjet_selector[jetR]
                # selected_jets = fj.sorted_by_pt(Cjet_selector(csa_medsub.inclusive_jets()))
                selected_jets = Cjet_selector(csa_medsub.inclusive_jets())
                C_area = np.sum([jet.area() for jet in selected_jets]) / (2 * np.pi * 0.9 * 2)
            else:
                rho = 0
                C_area = 0

            # Jet finding
            cs = fj.ClusterSequenceArea(fj_particles, jet_def, fj.AreaDefinition(fj.VoronoiAreaSpec()))
            jets = fj.sorted_by_pt(cs.inclusive_jets())
            jets_selected = jet_selector(jets)

            jets_reselected = [jet for jet in jets_selected if jet.perp() - rho * C_area * jet.area() > 5 and self.utils.is_det_jet_accepted(jet)]
            if self.do_perpendicular_cone:
                jets_with_perpcones = [self.attach_perp_cones(fj_particles, jet, coneR = jetR) for jet in jets_reselected]
                self.analyze_jets(jets_with_perpcones, rho_bge=rho*C_area)
            else:
                self.analyze_jets(jets_reselected, rho_bge=rho*C_area)
        self.fill_event_trees(rho, C_area, mult)

    def attach_perp_cones(self, parts, jet, coneR):
        cone_1 = fj.vectorPJ()
        cone_2 = fj.vectorPJ()
        angle = np.pi / 2 if not self.randomize_cone else np.random.uniform(np.pi / 3, 2 * np.pi / 3)
        # jet_phi_1, jet_phi_2 = (jet.phi() + angle) % (2 * np.pi), (jet.phi() - angle) % (2 * np.pi)
        rot_jet_1 = fj.PseudoJet()
        rot_jet_1.reset_PtYPhiM(jet.pt(), jet.rap(), jet.phi() + angle, jet.m())
        rot_jet_2 = fj.PseudoJet()
        rot_jet_2.reset_PtYPhiM(jet.pt(), jet.rap(), jet.phi() - angle, jet.m())
        for part in parts:
            # if np.sqrt((rot_jet_1.rap() - part.rap()) ** 2 + (rot_jet_1.phi() - part.phi()) ** 2) <= coneR:
            if np.sqrt((rot_jet_1.eta() - part.eta()) ** 2 + (rot_jet_1.delta_phi_to(part)) ** 2) <= coneR:
                rot_part = fj.PseudoJet()
                rot_part.reset_PtYPhiM(part.pt(), part.rap(), part.phi() - angle, part.m())
                rot_part.set_user_index(1)
                rot_part.set_python_info(part.python_info()) # copy python_info from original particle
                cone_1.push_back(rot_part)
            # particles can only be in one cone or the other, so we can use elif
            # elif np.sqrt((rot_jet_2.rap() - part.rap()) ** 2 + (rot_jet_2.phi() - part.phi()) ** 2) <= coneR:
            elif np.sqrt((rot_jet_2.eta() - part.eta()) ** 2 + (rot_jet_2.delta_phi_to(part)) ** 2) <= coneR:
                rot_part = fj.PseudoJet()
                rot_part.reset_PtYPhiM(part.pt(), part.rap(), part.phi() + angle, part.m())
                rot_part.set_user_index(-1)
                rot_part.set_python_info(part.python_info())
                cone_2.push_back(rot_part)

        info = jet_info.JetInfo()
        info.perpcone1 = cone_1
        info.perpcone2 = cone_2
        jet.set_python_info(info)
        return jet

    def get_branch_type(self, name):
        return getattr(self, name).GetTitle().split("/")[-1]

    def fill_event_trees(self, rho, areaC, mult):
        event_rho_arr = array(self.tree2arr[self.get_branch_type("event_rho")], [rho])
        getattr(self, "event_rho").SetAddress(event_rho_arr)
        event_areaC_arr = array(self.tree2arr[self.get_branch_type("event_areaC")], [areaC])
        getattr(self, "event_areaC").SetAddress(event_areaC_arr)
        event_mult_arr = array(self.tree2arr[self.get_branch_type("event_mult")], [mult])
        getattr(self, "event_mult").SetAddress(event_mult_arr)

        self.event_tree.Fill()

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

    def analyze_jets(self, jets, rho_bge):
        for jet in jets:
            self.fill_jet_trees(jet, rho_bge)
            self.fill_pair_trees(jet, rho_bge)

    def fill_jet_trees(self, jet, rho_bge):
        jet_pT_arr = array(self.tree2arr[self.get_branch_type("jet_pT")], [0.0])
        getattr(self, "jet_pT").SetAddress(jet_pT_arr)
        jet_pTsub_arr = array(self.tree2arr[self.get_branch_type("jet_pTsub")], [0.0])
        getattr(self, "jet_pTsub").SetAddress(jet_pTsub_arr)
        jet_area_arr = array(self.tree2arr[self.get_branch_type("jet_area")], [0.0])
        getattr(self, "jet_area").SetAddress(jet_area_arr)
        jet_nconst_arr = array(self.tree2arr[self.get_branch_type("jet_nconst")], [0])
        getattr(self, "jet_nconst").SetAddress(jet_nconst_arr)
        jet_eta_arr = array(self.tree2arr[self.get_branch_type("jet_eta")], [0.0])
        getattr(self, "jet_eta").SetAddress(jet_eta_arr)
        jet_phi_arr = array(self.tree2arr[self.get_branch_type("jet_phi")], [0.0])
        getattr(self, "jet_phi").SetAddress(jet_phi_arr)

        jet_pT_arr[0] = jet.pt()
        jet_pTsub_arr[0] = jet.pt() - rho_bge * jet.area()
        jet_area_arr[0] = jet.area()
        # jet_eta_arr[0] = jet.rap()
        jet_eta_arr[0] = jet.eta()
        jet_phi_arr[0] = jet.phi()
        jet_nconst_arr[0] = len(jet.constituents())
        self.jet_tree.Fill()

    def fill_pair_trees(self, jet, rho_bge):
        jetwcone1 = fj.vectorPJ()
        jetwcone2 = fj.vectorPJ()
        for part in jet.constituents():
            part.set_user_index(0)
            jetwcone1.push_back(part)
            jetwcone2.push_back(part)
        [jetwcone1.push_back(part) for part in jet.python_info().perpcone1]
        [jetwcone2.push_back(part) for part in jet.python_info().perpcone2]

        jet_pT = jet.pt()
        jet_pTsub = jet.pt() - rho_bge * jet.area()

        pair_pT1_arr = array(self.tree2arr[self.get_branch_type("pair_pT1")], [0.0])
        getattr(self, "pair_pT1").SetAddress(pair_pT1_arr)
        pair_pT2_arr = array(self.tree2arr[self.get_branch_type("pair_pT2")], [0.0])
        getattr(self, "pair_pT2").SetAddress(pair_pT2_arr)
        pair_q1_arr = array(self.tree2arr[self.get_branch_type("pair_q1")], [0])
        getattr(self, "pair_q1").SetAddress(pair_q1_arr)
        pair_q2_arr = array(self.tree2arr[self.get_branch_type("pair_q2")], [0])
        getattr(self, "pair_q2").SetAddress(pair_q2_arr)
        pair_id1_arr = array(self.tree2arr[self.get_branch_type("pair_id1")], [0])
        getattr(self, "pair_id1").SetAddress(pair_id1_arr)
        pair_id2_arr = array(self.tree2arr[self.get_branch_type("pair_id2")], [0])
        getattr(self, "pair_id2").SetAddress(pair_id2_arr)

        pair_weight_arr = array(self.tree2arr[self.get_branch_type("pair_weight")], [0.0])
        getattr(self, "pair_weight").SetAddress(pair_weight_arr)
        pair_RL_arr = array(self.tree2arr[self.get_branch_type("pair_RL")], [0.0])
        getattr(self, "pair_RL").SetAddress(pair_RL_arr)
        pair_deta_arr = array(self.tree2arr[self.get_branch_type("pair_deta")], [0.0])
        getattr(self, "pair_deta").SetAddress(pair_deta_arr)
        pair_dphi_arr = array(self.tree2arr[self.get_branch_type("pair_dphi")], [0.0])
        getattr(self, "pair_dphi").SetAddress(pair_dphi_arr)

        pair_pTjet_arr = array(self.tree2arr[self.get_branch_type("pair_pTjet")], [jet_pT])
        getattr(self, "pair_pTjet").SetAddress(pair_pTjet_arr)
        pair_pTjetsub_arr = array(self.tree2arr[self.get_branch_type("pair_pTjetsub")], [jet_pTsub])
        getattr(self, "pair_pTjetsub").SetAddress(pair_pTjetsub_arr)

        # for p1, p2 in itertools.product(jetwcone1, repeat = 2):
        for p1, p2 in itertools.permutations(jetwcone1, 2):
            pair_pT1_arr[0] = p1.pt()
            pair_pT2_arr[0] = p2.pt()
            pair_q1_arr[0] = p1.python_info().charge
            pair_q2_arr[0] = p2.python_info().charge
            pair_id1_arr[0] = p1.user_index()
            pair_id2_arr[0] = p2.user_index()
            pair_weight_arr[0] = p1.pt() * p2.pt() / (jet_pTsub * jet_pTsub)
            pair_RL_arr[0] = self.deltaR(p1, p2)
            pair_deta_arr[0] = p2.eta() - p1.eta()
            pair_dphi_arr[0] = p1.delta_phi_to(p2)

            self.pair_tree.Fill()
        # for p1, p2 in itertools.product(jetwcone2, repeat = 2):
        for p1, p2 in itertools.permutations(jetwcone2, 2):
            pair_id1_arr[0] = p1.user_index()
            pair_id2_arr[0] = p2.user_index()
            if pair_id1_arr[0] == 0 and pair_id2_arr[0] == 0:
                continue
            pair_pT1_arr[0] = p1.pt()
            pair_pT2_arr[0] = p2.pt()
            pair_q1_arr[0] = p1.python_info().charge
            pair_q2_arr[0] = p2.python_info().charge
            pair_weight_arr[0] = p1.pt() * p2.pt() / (jet_pTsub * jet_pTsub)
            pair_RL_arr[0] = self.deltaR(p1, p2)
            pair_deta_arr[0] = p2.eta() - p1.eta()
            pair_dphi_arr[0] = p1.delta_phi_to(p2)

            self.pair_tree.Fill()

        for p1 in jet.python_info().perpcone1:
            for p2 in jet.python_info().perpcone2:
                pair_id1_arr[0] = p1.user_index()
                pair_id2_arr[0] = p2.user_index()
                pair_pT1_arr[0] = p1.pt()
                pair_pT2_arr[0] = p2.pt()
                pair_q1_arr[0] = p1.python_info().charge
                pair_q2_arr[0] = p2.python_info().charge
                pair_weight_arr[0] = p1.pt() * p2.pt() / (jet_pTsub * jet_pTsub)
                pair_RL_arr[0] = self.deltaR(p1, p2)
                pair_deta_arr[0] = p2.eta() - p1.eta()
                pair_dphi_arr[0] = p1.delta_phi_to(p2)

                self.pair_tree.Fill()

                # Swap to reverse order and fill again
                pair_id1_arr[0] = p2.user_index()
                pair_id2_arr[0] = p1.user_index()
                pair_pT1_arr[0] = p2.pt()
                pair_pT2_arr[0] = p1.pt()
                pair_q1_arr[0] = p2.python_info().charge
                pair_q2_arr[0] = p1.python_info().charge
                pair_deta_arr[0] = p1.eta() - p2.eta()
                pair_dphi_arr[0] = p2.delta_phi_to(p1)

                self.pair_tree.Fill()
    # def deltaR(self, p1, p2):
    #     return np.sqrt(p1.delta_phi_to(p2) ** 2 + (p1.eta() - p2.eta()) ** 2)

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

    args = parser.parse_args()

    handler = logging.StreamHandler()

    handler.setFormatter(ColoredFormatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(funcName)s - %(message)s'))

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

    analysis = Generator_Tree_Data_ENC(
        input_file=args.input_file,
        config_file=args.config_file,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
    )
    analysis.process_data()