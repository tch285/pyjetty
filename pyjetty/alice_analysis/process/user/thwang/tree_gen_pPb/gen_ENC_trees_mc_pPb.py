#!/usr/bin/env python3

"""
    Analysis class to read a ROOT TTree of MC track information
    and do jet-finding, and save response histograms.
    
    Author: James Mulligan (james.mulligan@berkeley.edu)
"""

# General
import os
import sys
import argparse
import itertools
import logging

# Data analysis and plotting
import numpy as np
import ROOT
import yaml
from array import array

# Fastjet via python (from external library heppy)
import fastjet as fj

# Analysis utilities
from pyjetty.alice_analysis.process.base import jet_info
from pyjetty.alice_analysis.process.user.substructure import process_mc_base_pPb

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
class Generator_Tree_MC_ENC(process_mc_base_pPb.ProcessMCBase):
    def __init__(self, input_file="", config_file="", output_dir="", output_filename="", debug_level=0, **kwargs):
        # Initialize base class
        self.outputfilename = output_filename
        super(Generator_Tree_MC_ENC, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)

        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        self.event_branches = [
            (bname, btype)
            for bname, btype in (
                branch.split("/") for branch in config["event_branches"]
            )
        ]
        self.jet_det_branches = [
            (bname, btype)
            for bname, btype in (
                branch.split("/") for branch in config["jet_det_branches"]
            )
        ]
        self.jet_gen_branches = [
            (bname, btype)
            for bname, btype in (
                branch.split("/") for branch in config["jet_gen_branches"]
            )
        ]
        self.part_det_branches = [
            (bname, btype)
            for bname, btype in (
                branch.split("/") for branch in config["part_det_branches"]
            )
        ]
        self.part_gen_branches = [
            (bname, btype)
            for bname, btype in (
                branch.split("/") for branch in config["part_gen_branches"]
            )
        ]
        self.pair_gen_branches = [
            (bname, btype)
            for bname, btype in (
                branch.split("/") for branch in config["pair_gen_branches"]
            )
        ]
        self.pair_det_branches = [
            (bname, btype)
            for bname, btype in (
                branch.split("/") for branch in config["pair_det_branches"]
            )
        ]
        self.tree2arr = {"I": "i", "F": "f", "D": "d", "L": "l", "O": "B", "B": "b"}

    def calc_phistar(self, p1, p2, q1, q2):
        R = 1.1 # reference radius for TPC
        Bz = 0.5 # extra minus REMOVED for pPb
        dalpha = q1*np.arcsin(-0.15*Bz*R/p1.pt()) - q2*np.arcsin(-0.15*Bz*R/p2.pt())

        return self.calculate_dphi(p1.phi(), p2.phi()) + dalpha

    def calculate_dphi(self, phi1, phi2):
        delta_phi = self.Phi_mpi_pi(phi1-phi2)
        # if (delta_phi < -0.5*M_PI) delta_phi += 2*M_PI; // This should not be needed

        if (delta_phi>np.pi or delta_phi < -np.pi):
            logger.warning("Delta phi not inside desired range")

        return delta_phi

    def Phi_mpi_pi(self, dphi):
        while dphi >= np.pi:
            dphi -= (np.pi * 2)
        while dphi < -np.pi:
            dphi += (np.pi * 2)
        return dphi

    def calc_kt(self, p1, p2):
        # original formula below
        # return 0.5*np.sqrt( pow(p1.pt(),2)+pow(p1.pt(),2)+2*p1.pt()*p2.pt()*np.cos(p1.phi()-p2.phi()) )
        # fixed formula below
        return 0.5*np.sqrt( pow(p1.pt(),2)+pow(p2.pt(),2)+ 2*p1.pt()*p2.pt()*np.cos(p1.phi()-p2.phi()) )
        # NOTE: to self: this is actually law of cosines but vector causes plus not minus on cos term
        # e.g. consider case when phis are equal, then sum has double magnitude, which is only possible
        # with + cos not - cos (proper angle is 180-theta not theta)
    #---------------------------------------------------------------
    # Initialize histograms
    #---------------------------------------------------------------
    def initialize_user_output_objects_R(self, jetR):
        self.event_tree = ROOT.TTree("events", "events")
        self.jet_det_tree = ROOT.TTree("jets_det", f"det jets (R={jetR})")
        self.jet_gen_tree = ROOT.TTree("jets_gen", f"gen jets (R={jetR})")
        self.part_det_tree = ROOT.TTree("parts_det", "det particles")
        self.part_gen_tree = ROOT.TTree("parts_gen", "gen particles")
        self.pair_det_tree = ROOT.TTree(
            "pairs_det", f"det particle pairs (R={jetR})"
        )
        self.pair_gen_tree = ROOT.TTree(
            "pairs_gen", f"gen particle pairs (R={jetR})"
        )

        self.add_branches(self.event_tree, self.event_branches, "event")
        self.add_branches(self.jet_det_tree, self.jet_det_branches, "jet_det")
        self.add_branches(self.jet_gen_tree, self.jet_gen_branches, "jet_gen")
        self.add_branches(self.part_det_tree, self.part_det_branches, "part_det")
        self.add_branches(self.part_gen_tree, self.part_gen_branches, "part_gen")
        self.add_branches(self.pair_det_tree, self.pair_det_branches, "pair_det")
        self.add_branches(self.pair_gen_tree, self.pair_gen_branches, "pair_gen")

    def add_branches(self, tree, branches, prefix):
        for name, ttype in branches:
            atype = self.tree2arr[ttype]
            arr = array(atype, [0])
            setattr(self, f"{prefix}_{name}", tree.Branch(name, arr, f"{name}/{ttype}"))
    def get_branch_type(self, name):
        return getattr(self, name).GetTitle().split("/")[-1]
    def get_pair_type(self, uidx1, uidx2):
        # jet parts have uidx 0, cone1 (+angle) has uidx +1, cone2 (-angle) has uidx -1
        if uidx1 == 0 and uidx2 == 0:
            return '_jj'
        elif uidx1 * uidx2 == 0: # take advantage of lazy if
            return '_jp'
        else:
            return '_pp'

    def analyze_event(self, fj_particles_det, fj_particles_truth, mult = None, fj_particles_det_holes=None,
                      fj_particles_truth_holes=None, particles_mcid_det=None, particles_pid_truth=None):
        self.event_number += 1
        if self.event_number > self.event_number_max:
            return
        if len(fj_particles_truth) == 0 or len(fj_particles_det) == 0:
            logger.warning(f"Event {self.event_number} has {len(fj_particles_det)} det tracks and {len(fj_particles_truth)} truth tracks.")

        for i_truth in range( len(fj_particles_truth) ):
            particle_truth = fj_particles_truth[i_truth]
            info_truth = particle_truth.python_info()
            mcid_truth = info_truth.mcid
            candidates = []
            candidates_mcid = []
            candidates_idx = []
            for i_det in range( len(fj_particles_det) ):
                particle_det = fj_particles_det[i_det]
                info_det = particle_det.python_info()
                if np.abs(info_det.mcid) == mcid_truth:
                    candidates.append(particle_det)
                    candidates_mcid.append(info_det.mcid)
                    candidates_idx.append(i_det)

            if candidates:
                if len(candidates) == 1:
                    matched_det_idx = candidates_idx[0]
                else:
                    logger.warning(f"Found 2+ det particles with matching abs mcid: {candidates_mcid}")
                    if all(id > 0 for id in candidates_mcid) or all(id < 0 for id in candidates_mcid) or all(id == 0 for id in candidates_mcid):
                        # mcids are all positive or all negative
                        deltaR = [self.deltaR(particle_truth, part_det) for part_det in candidates]
                        matched_det_idx = candidates_idx[np.argmin(deltaR)]
                    else:
                        # mcids are mixed +ve and -ve
                        candidates_abs = [(part_det, idx) for part_det, mcid, idx in zip(candidates, candidates_mcid, candidates_idx) if mcid > 0]
                        deltaR = [self.deltaR(particle_truth, part_det) for part_det, idx in candidates_abs]
                        _, matched_det_idx = candidates_abs[np.argmin(deltaR)]

                matched_det = fj_particles_det[matched_det_idx]
                matched_det_info = matched_det.python_info()

                info_truth.particle_det = matched_det
                matched_det_info.particle_truth = particle_truth

                fj_particles_truth[i_truth].set_python_info(info_truth)
                fj_particles_det[matched_det_idx].set_python_info(matched_det_info)

        self.fill_particle_trees(fj_particles_det, fj_particles_truth)

        for jetR in self.jetR_list:
            # Set jet definition and a jet selector
            jet_def = self.jet_defs[jetR]
            jet_selector_det = self.jet_selectors_det[jetR]
            
            if self.do_median_subtraction:
                csa_medsub = fj.ClusterSequenceArea(fj_particles_det, self.jet_def_medsub[jetR], fj.AreaDefinition(fj.active_area_explicit_ghosts))
                self.median_subtractor[jetR].set_cluster_sequence(csa_medsub)
                rho_det = self.median_subtractor[jetR].rho()

                Cjet_selector = self.Cjet_selectors[jetR]
                medsub_selected_jets = Cjet_selector(csa_medsub.inclusive_jets())
                C_area_det = np.sum([jet.area() for jet in medsub_selected_jets]) / (2 * np.pi * 0.9 * 2)

                csa_medsub_truth = fj.ClusterSequenceArea(fj_particles_truth, self.jet_def_medsub[jetR], fj.AreaDefinition(fj.active_area_explicit_ghosts))
                self.median_subtractor_truth[jetR].set_cluster_sequence(csa_medsub_truth)
                rho_truth = self.median_subtractor_truth[jetR].rho()
                medsub_selected_jets_truth = Cjet_selector(csa_medsub_truth.inclusive_jets())
                C_area_truth = np.sum([jet.area() for jet in medsub_selected_jets_truth]) / (2 * np.pi * 2 * 0.9)
            else:
                rho_det = 0
                rho_truth = 0
                C_area_det = 0
                C_area_truth = 0
            
            cs_det = fj.ClusterSequenceArea(fj_particles_det, jet_def, fj.AreaDefinition(fj.VoronoiAreaSpec()))
            jets_det = fj.sorted_by_pt(jet_selector_det(cs_det.inclusive_jets()))
            jets_det_selected = [jet for jet in jets_det if jet.pt() - rho_det * jet.area() * C_area_det > 5]

            cs_truth = fj.ClusterSequenceArea(fj_particles_truth, jet_def, fj.AreaDefinition(fj.VoronoiAreaSpec()))
            jets_truth = fj.sorted_by_pt(jet_selector_det(cs_truth.inclusive_jets()))
            if self.do_perpendicular_cone:
                jets_det_wpcone = [self.attach_perp_cones(fj_particles_det, jet, coneR = jetR) for jet in jets_det_selected]
                # jets_combined_reselected_wpcone = [self.attach_perp_cones(fj_particles_det, jet, coneR = jetR) for jet in jets_combined_selected]
                jets_truth_wpcone = [self.attach_perp_cones(fj_particles_truth, jet, coneR = jetR) for jet in jets_truth]
                self.analyze_jets(jets_det_wpcone, jets_truth_wpcone, rho_det * C_area_det, rho_truth * C_area_truth)
            else:
                self.analyze_jets(jets_det_selected, jets_truth, rho_det * C_area_det, rho_truth * C_area_truth)

        self.fill_event_trees(rho_det, rho_truth, C_area_det, C_area_truth, mult)

    def fill_event_trees(self, rho_det, rho_gen, areaC_det, areaC_gen, mult):
        event_rho_det_arr = array(self.tree2arr[self.get_branch_type("event_rho_det")], [rho_det])
        getattr(self, "event_rho_det").SetAddress(event_rho_det_arr)
        event_rho_gen_arr = array(self.tree2arr[self.get_branch_type("event_rho_gen")], [rho_gen])
        getattr(self, "event_rho_gen").SetAddress(event_rho_gen_arr)
        event_areaC_det_arr = array(self.tree2arr[self.get_branch_type("event_areaC_det")], [areaC_det])
        getattr(self, "event_areaC_det").SetAddress(event_areaC_det_arr)
        event_areaC_gen_arr = array(self.tree2arr[self.get_branch_type("event_areaC_gen")], [areaC_gen])
        getattr(self, "event_areaC_gen").SetAddress(event_areaC_gen_arr)
        event_mult_arr = array(self.tree2arr[self.get_branch_type("event_mult")], [mult])
        getattr(self, "event_mult").SetAddress(event_mult_arr)

        self.event_tree.Fill()

    def fill_particle_trees(self, fj_particles_det, fj_particles_truth):
        part_det_pT_arr = array(
            self.tree2arr[self.get_branch_type("part_det_pT")], [0.0]
        )
        getattr(self, "part_det_pT").SetAddress(part_det_pT_arr)
        part_det_eta_arr = array(
            self.tree2arr[self.get_branch_type("part_det_eta")], [0.0]
        )
        getattr(self, "part_det_eta").SetAddress(part_det_eta_arr)
        part_det_phi_arr = array(
            self.tree2arr[self.get_branch_type("part_det_phi")], [0.0]
        )
        getattr(self, "part_det_phi").SetAddress(part_det_phi_arr)
        part_det_q_arr = array(self.tree2arr[self.get_branch_type("part_det_q")], [0])
        getattr(self, "part_det_q").SetAddress(part_det_q_arr)
        part_det_match_pT_arr = array(
            self.tree2arr[self.get_branch_type("part_det_match_pT")], [0]
        )
        getattr(self, "part_det_match_pT").SetAddress(part_det_match_pT_arr)

        for part in fj_particles_det:
            part_det_pT_arr[0] = part.pt()
            part_det_eta_arr[0] = part.eta()
            part_det_phi_arr[0] = part.phi()
            part_det_q_arr[0] = part.python_info().charge
            part_det_match_pT_arr[0] = (
                part.python_info().particle_truth.pt()
                if part.python_info().particle_truth is not None
                else -1
            )
            self.part_det_tree.Fill()

        part_gen_pT_arr = array(
            self.tree2arr[self.get_branch_type("part_gen_pT")], [0.0]
        )
        getattr(self, "part_gen_pT").SetAddress(part_gen_pT_arr)
        part_gen_eta_arr = array(
            self.tree2arr[self.get_branch_type("part_gen_eta")], [0.0]
        )
        getattr(self, "part_gen_eta").SetAddress(part_gen_eta_arr)
        part_gen_phi_arr = array(
            self.tree2arr[self.get_branch_type("part_gen_phi")], [0.0]
        )
        getattr(self, "part_gen_phi").SetAddress(part_gen_phi_arr)
        part_gen_q_arr = array(self.tree2arr[self.get_branch_type("part_gen_q")], [0])
        getattr(self, "part_gen_q").SetAddress(part_gen_q_arr)
        part_gen_match_pT_arr = array(
            self.tree2arr[self.get_branch_type("part_gen_match_pT")], [0]
        )
        getattr(self, "part_gen_match_pT").SetAddress(part_gen_match_pT_arr)

        for part in fj_particles_truth:
            part_gen_pT_arr[0] = part.pt()
            part_gen_eta_arr[0] = part.eta()
            part_gen_phi_arr[0] = part.phi()
            part_gen_q_arr[0] = part.python_info().charge
            part_gen_match_pT_arr[0] = (
                part.python_info().particle_det.pt()
                if part.python_info().particle_det is not None
                else -1
            )
            self.part_gen_tree.Fill()

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
            # if np.sqrt((rot_jet_1.eta() - part.eta()) ** 2 + (rot_jet_1.phi() - part.phi()) ** 2) <= coneR:
            if np.sqrt((rot_jet_1.eta() - part.eta()) ** 2 + (rot_jet_1.delta_phi_to(part)) ** 2) <= coneR:
                rot_part = fj.PseudoJet()
                rot_part.reset_PtYPhiM(part.pt(), part.rap(), part.phi() - angle, part.m())
                rot_part.set_user_index(1)
                rot_part.set_python_info(part.python_info()) # copy python_info from original particle
                cone_1.push_back(rot_part)
            # particles can only be in one cone or the other, so we can use elif
            # elif np.sqrt((rot_jet_2.eta() - part.eta()) ** 2 + (rot_jet_2.phi() - part.phi()) ** 2) <= coneR:
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

    def analyze_jets(self, jets_det_selected, jets_truth, rho_bge_det, rho_bge_truth):
        for jet_det in jets_det_selected:
            self.fill_jet_det_tree(jet_det, rho_bge_det)
            self.fill_pair_det_tree(jet_det, rho_bge_det)
        for jet_truth in jets_truth:
            self.fill_jet_gen_tree(jet_truth, rho_bge_truth)
            self.fill_pair_gen_tree(jet_truth, rho_bge_truth)

    def fill_jet_gen_tree(self, jet, rho_bge):
        jet_gen_pT_arr = array(self.tree2arr[self.get_branch_type("jet_gen_pT")], [0.0])
        getattr(self, "jet_gen_pT").SetAddress(jet_gen_pT_arr)
        jet_gen_pTsub_arr = array(self.tree2arr[self.get_branch_type("jet_gen_pTsub")], [0.0])
        getattr(self, "jet_gen_pTsub").SetAddress(jet_gen_pTsub_arr)
        jet_gen_area_arr = array(self.tree2arr[self.get_branch_type("jet_gen_area")], [0.0])
        getattr(self, "jet_gen_area").SetAddress(jet_gen_area_arr)
        jet_gen_nconst_arr = array(self.tree2arr[self.get_branch_type("jet_gen_nconst")], [0])
        getattr(self, "jet_gen_nconst").SetAddress(jet_gen_nconst_arr)
        jet_gen_eta_arr = array(self.tree2arr[self.get_branch_type("jet_gen_eta")], [0.0])
        getattr(self, "jet_gen_eta").SetAddress(jet_gen_eta_arr)
        jet_gen_phi_arr = array(self.tree2arr[self.get_branch_type("jet_gen_phi")], [0.0])
        getattr(self, "jet_gen_phi").SetAddress(jet_gen_phi_arr)

        jet_gen_pT_arr[0] = jet.pt()
        jet_gen_pTsub_arr[0] = jet.pt() - rho_bge * jet.area()
        jet_gen_area_arr[0] = jet.area()
        # jet_gen_eta_arr[0] = jet.rap()
        jet_gen_eta_arr[0] = jet.eta()
        jet_gen_phi_arr[0] = jet.phi()
        jet_gen_nconst_arr[0] = len(jet.constituents())
        self.jet_gen_tree.Fill()

    def fill_jet_det_tree(self, jet, rho_bge):
        jet_det_pT_arr = array(self.tree2arr[self.get_branch_type("jet_det_pT")], [0.0])
        getattr(self, "jet_det_pT").SetAddress(jet_det_pT_arr)
        jet_det_pTsub_arr = array(self.tree2arr[self.get_branch_type("jet_det_pTsub")], [0.0])
        getattr(self, "jet_det_pTsub").SetAddress(jet_det_pTsub_arr)
        jet_det_area_arr = array(self.tree2arr[self.get_branch_type("jet_det_area")], [0.0])
        getattr(self, "jet_det_area").SetAddress(jet_det_area_arr)
        jet_det_nconst_arr = array(self.tree2arr[self.get_branch_type("jet_det_nconst")], [0])
        getattr(self, "jet_det_nconst").SetAddress(jet_det_nconst_arr)
        jet_det_eta_arr = array(self.tree2arr[self.get_branch_type("jet_det_eta")], [0.0])
        getattr(self, "jet_det_eta").SetAddress(jet_det_eta_arr)
        jet_det_phi_arr = array(self.tree2arr[self.get_branch_type("jet_det_phi")], [0.0])
        getattr(self, "jet_det_phi").SetAddress(jet_det_phi_arr)

        jet_det_pT_arr[0] = jet.pt()
        jet_det_pTsub_arr[0] = jet.pt() - rho_bge * jet.area()
        jet_det_area_arr[0] = jet.area()
        # jet_det_eta_arr[0] = jet.rap()
        jet_det_eta_arr[0] = jet.eta()
        jet_det_phi_arr[0] = jet.phi()
        jet_det_nconst_arr[0] = len(jet.constituents())
        self.jet_det_tree.Fill()

    def fill_pair_det_tree(self, jet, rho_bge):
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

        pair_det_pT1_arr = array(self.tree2arr[self.get_branch_type("pair_det_pT1")], [0.0])
        getattr(self, "pair_det_pT1").SetAddress(pair_det_pT1_arr)
        pair_det_pT2_arr = array(self.tree2arr[self.get_branch_type("pair_det_pT2")], [0.0])
        getattr(self, "pair_det_pT2").SetAddress(pair_det_pT2_arr)
        pair_det_q1_arr = array(self.tree2arr[self.get_branch_type("pair_det_q1")], [0])
        getattr(self, "pair_det_q1").SetAddress(pair_det_q1_arr)
        pair_det_q2_arr = array(self.tree2arr[self.get_branch_type("pair_det_q2")], [0])
        getattr(self, "pair_det_q2").SetAddress(pair_det_q2_arr)
        pair_det_id1_arr = array(self.tree2arr[self.get_branch_type("pair_det_id1")], [0])
        getattr(self, "pair_det_id1").SetAddress(pair_det_id1_arr)
        pair_det_id2_arr = array(self.tree2arr[self.get_branch_type("pair_det_id2")], [0])
        getattr(self, "pair_det_id2").SetAddress(pair_det_id2_arr)

        pair_det_weight_arr = array(self.tree2arr[self.get_branch_type("pair_det_weight")], [0.0])
        getattr(self, "pair_det_weight").SetAddress(pair_det_weight_arr)
        pair_det_RL_arr = array(self.tree2arr[self.get_branch_type("pair_det_RL")], [0.0])
        getattr(self, "pair_det_RL").SetAddress(pair_det_RL_arr)
        pair_det_deta_arr = array(self.tree2arr[self.get_branch_type("pair_det_deta")], [0.0])
        getattr(self, "pair_det_deta").SetAddress(pair_det_deta_arr)
        pair_det_dphi_arr = array(self.tree2arr[self.get_branch_type("pair_det_dphi")], [0.0])
        getattr(self, "pair_det_dphi").SetAddress(pair_det_dphi_arr)

        pair_det_pTjet_arr = array(self.tree2arr[self.get_branch_type("pair_det_pTjet")], [jet_pT])
        getattr(self, "pair_det_pTjet").SetAddress(pair_det_pTjet_arr)
        pair_det_pTjetsub_arr = array(self.tree2arr[self.get_branch_type("pair_det_pTjetsub")], [jet_pTsub])
        getattr(self, "pair_det_pTjetsub").SetAddress(pair_det_pTjetsub_arr)

        # for p1, p2 in itertools.product(jetwcone1, repeat = 2):
        for p1, p2 in itertools.permutations(jetwcone1, 2):
            pair_det_pT1_arr[0] = p1.pt()
            pair_det_pT2_arr[0] = p2.pt()
            pair_det_q1_arr[0] = p1.python_info().charge
            pair_det_q2_arr[0] = p2.python_info().charge
            pair_det_id1_arr[0] = p1.user_index()
            pair_det_id2_arr[0] = p2.user_index()
            pair_det_weight_arr[0] = p1.pt() * p2.pt() / (jet_pTsub * jet_pTsub)
            pair_det_RL_arr[0] = self.deltaR(p1, p2)
            pair_det_deta_arr[0] = p2.eta() - p1.eta()
            pair_det_dphi_arr[0] = p1.delta_phi_to(p2)

            self.pair_det_tree.Fill()
        # for p1, p2 in itertools.product(jetwcone2, repeat = 2):
        for p1, p2 in itertools.permutations(jetwcone2, 2):
            pair_det_id1_arr[0] = p1.user_index()
            pair_det_id2_arr[0] = p2.user_index()
            if pair_det_id1_arr[0] == 0 and pair_det_id2_arr[0] == 0:
                continue
            pair_det_pT1_arr[0] = p1.pt()
            pair_det_pT2_arr[0] = p2.pt()
            pair_det_q1_arr[0] = p1.python_info().charge
            pair_det_q2_arr[0] = p2.python_info().charge
            pair_det_weight_arr[0] = p1.pt() * p2.pt() / (jet_pTsub * jet_pTsub)
            pair_det_RL_arr[0] = self.deltaR(p1, p2)
            pair_det_deta_arr[0] = p2.eta() - p1.eta()
            pair_det_dphi_arr[0] = p1.delta_phi_to(p2)

            self.pair_det_tree.Fill()
    
    def fill_pair_gen_tree(self, jet, rho_bge):
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

        pair_gen_pT1_arr = array(self.tree2arr[self.get_branch_type("pair_gen_pT1")], [0.0])
        getattr(self, "pair_gen_pT1").SetAddress(pair_gen_pT1_arr)
        pair_gen_pT2_arr = array(self.tree2arr[self.get_branch_type("pair_gen_pT2")], [0.0])
        getattr(self, "pair_gen_pT2").SetAddress(pair_gen_pT2_arr)
        pair_gen_q1_arr = array(self.tree2arr[self.get_branch_type("pair_gen_q1")], [0])
        getattr(self, "pair_gen_q1").SetAddress(pair_gen_q1_arr)
        pair_gen_q2_arr = array(self.tree2arr[self.get_branch_type("pair_gen_q2")], [0])
        getattr(self, "pair_gen_q2").SetAddress(pair_gen_q2_arr)
        pair_gen_id1_arr = array(self.tree2arr[self.get_branch_type("pair_gen_id1")], [0])
        getattr(self, "pair_gen_id1").SetAddress(pair_gen_id1_arr)
        pair_gen_id2_arr = array(self.tree2arr[self.get_branch_type("pair_gen_id2")], [0])
        getattr(self, "pair_gen_id2").SetAddress(pair_gen_id2_arr)

        pair_gen_weight_arr = array(self.tree2arr[self.get_branch_type("pair_gen_weight")], [0.0])
        getattr(self, "pair_gen_weight").SetAddress(pair_gen_weight_arr)
        pair_gen_RL_arr = array(self.tree2arr[self.get_branch_type("pair_gen_RL")], [0.0])
        getattr(self, "pair_gen_RL").SetAddress(pair_gen_RL_arr)
        pair_gen_deta_arr = array(self.tree2arr[self.get_branch_type("pair_gen_deta")], [0.0])
        getattr(self, "pair_gen_deta").SetAddress(pair_gen_deta_arr)
        pair_gen_dphi_arr = array(self.tree2arr[self.get_branch_type("pair_gen_dphi")], [0.0])
        getattr(self, "pair_gen_dphi").SetAddress(pair_gen_dphi_arr)

        pair_gen_pTjet_arr = array(self.tree2arr[self.get_branch_type("pair_gen_pTjet")], [jet_pT])
        getattr(self, "pair_gen_pTjet").SetAddress(pair_gen_pTjet_arr)
        pair_gen_pTjetsub_arr = array(self.tree2arr[self.get_branch_type("pair_gen_pTjetsub")], [jet_pTsub])
        getattr(self, "pair_gen_pTjetsub").SetAddress(pair_gen_pTjetsub_arr)

        # for p1, p2 in itertools.product(jetwcone1, repeat = 2):
        for p1, p2 in itertools.permutations(jetwcone1, 2):
            pair_gen_pT1_arr[0] = p1.pt()
            pair_gen_pT2_arr[0] = p2.pt()
            pair_gen_q1_arr[0] = p1.python_info().charge
            pair_gen_q2_arr[0] = p2.python_info().charge
            pair_gen_id1_arr[0] = p1.user_index()
            pair_gen_id2_arr[0] = p2.user_index()
            pair_gen_weight_arr[0] = p1.pt() * p2.pt() / (jet_pTsub * jet_pTsub)
            pair_gen_RL_arr[0] = self.deltaR(p1, p2)
            pair_gen_deta_arr[0] = p2.eta() - p1.eta()
            pair_gen_dphi_arr[0] = p1.delta_phi_to(p2)

            self.pair_gen_tree.Fill()
        # for p1, p2 in itertools.product(jetwcone2, repeat = 2):
        for p1, p2 in itertools.permutations(jetwcone2, 2):
            pair_gen_id1_arr[0] = p1.user_index()
            pair_gen_id2_arr[0] = p2.user_index()
            if pair_gen_id1_arr[0] == 0 and pair_gen_id2_arr[0] == 0:
                continue
            pair_gen_pT1_arr[0] = p1.pt()
            pair_gen_pT2_arr[0] = p2.pt()
            pair_gen_q1_arr[0] = p1.python_info().charge
            pair_gen_q2_arr[0] = p2.python_info().charge
            pair_gen_weight_arr[0] = p1.pt() * p2.pt() / (jet_pTsub * jet_pTsub)
            pair_gen_RL_arr[0] = self.deltaR(p1, p2)
            pair_gen_deta_arr[0] = p2.eta() - p1.eta()
            pair_gen_dphi_arr[0] = p1.delta_phi_to(p2)

            self.pair_gen_tree.Fill()

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

    analysis = Generator_Tree_MC_ENC(
        input_file=args.input_file,
        config_file=args.config_file,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
    )
    analysis.process_mc()