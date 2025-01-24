#!/usr/bin/env python3

"""
Tree class class to read a ROOT TTree of MC track information
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
from particle import PDGID
from pyjetty.alice_analysis.process.base import jet_info
from pyjetty.alice_analysis.process.user.substructure import process_mc_base

logger = logging.getLogger(__name__)

ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()

################################################################
class Generator_Tree_MC_ENC(process_mc_base.ProcessMCBase):
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
        super(Generator_Tree_MC_ENC, self).__init__(
            input_file, config_file, output_dir, debug_level, **kwargs
        )

        with open(self.config_file, "r") as stream:
            config = yaml.safe_load(stream)
            
        # if 'pair_fcn' in config:
        #     if config['pair_fcn'] in ['perm', 'permutations']:
        #         self.pairs = itertools.permutations
        #     elif config['pair_fcn'] in ['prod', 'product']:
        #         self.pairs = itertools.product
        #     else:
        #         logger.warning(f"Pairing function '{config['pair_fcn']}' unrecognized, default to permutations.")
        #         self.pairs = itertools.permutations
        # else:
        #     logger.warning("Pairing function not set, defaulting to permutations.")
        #     self.pairs = itertools.permutations

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
    def initialize_output_objects_R(self, jetR):
        # store the ROOT TTree type names and the corresponding Python
        # array.array types in this dict
        self.tree2arr = {"I": "i", "F": "f", "D": "d", "L": "l", "O": "B", "B": "b"}
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
            # setattr(self, name, arr)

    def analyze_event(
        self,
        fj_particles_det,
        fj_particles_truth,
        fj_particles_det_holes=None,
        fj_particles_truth_holes=None,
        particles_mcid_det=None,
        particles_pid_truth=None,
    ):
        self.event_number += 1
        if self.event_number > self.event_number_max:
            return
        if self.debug_level > 1:
            logger.debug(f"event {self.event_number}")

        if not self.ENC_fastsim:
            # if not isinstance(fj_particles_truth, fj.vectorPJ):
            if len(fj_particles_truth) == 0:
                # fj_particles_truth = fj.vectorPJ()
                logger.warning(
                    f"Event {self.event_number} has {len(fj_particles_det)} det tracks and {len(fj_particles_truth)} truth tracks."
                )
            # if not isinstance(fj_particles_det, fj.vectorPJ):
            if len(fj_particles_det) == 0:
                # fj_particles_det = fj.vectorPJ()
                logger.warning(
                    f"Event {self.event_number} has {len(fj_particles_det)} det tracks and {len(fj_particles_truth)} truth tracks."
                )
        else:
            if len(fj_particles_truth) == 0:
                logger.warning(
                    f"Event {self.event_number} has {len(fj_particles_det)} det tracks and {len(fj_particles_truth)} truth tracks."
                )
                particles_pid_truth = []

        if self.ENC_fastsim:
            # make charge array from pid info, needed for pair efficiency determination
            particles_charge_truth = np.array([])
            for pid in particles_pid_truth:
                # charged hadrons
                if abs(pid) in [211, 321, 2212, 3222]:
                    if pid > 0:
                        particles_charge_truth = np.append(particles_charge_truth, 1)
                    else:
                        particles_charge_truth = np.append(particles_charge_truth, -1)
                # electrons and muons
                elif abs(pid) in [11, 13, 3112, 3312, 3334]:
                    if pid > 0:
                        particles_charge_truth = np.append(particles_charge_truth, -1)
                    else:
                        particles_charge_truth = np.append(particles_charge_truth, 1)
                # long lived weak decay particles (<2% of the total number of charged particles)
                # for now mark as charge 0 and later NOT applying pair efficiency for 0-charged or 0-0 pairs
                # NB: this can be avoided by decaying these paritcles within the generation step
                else:
                    particles_charge_truth = np.append(particles_charge_truth, 0)
                # print(PDGID(pid).charge, particles_charge_truth[-1])
                if PDGID(pid).charge != particles_charge_truth[-1]:
                    print(
                        "MISMATCHED CHARGE----------------------------------------------------------------------------------------"
                    )
        else:  # is pythia, so we match MC truth to det here:
            logger.debug("Starting MC ID matching.")
            for i_truth in range(len(fj_particles_truth)):
                particle_truth = fj_particles_truth[i_truth]
                info_truth = particle_truth.python_info()
                mcid_truth = info_truth.mcid
                candidates = []
                candidates_mcid = []
                candidates_idx = []
                for i_det in range(len(fj_particles_det)):
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
                        logger.warning(
                            f"Found 2+ det particles with matching abs mcid: {candidates_mcid}"
                        )
                        if all(id > 0 for id in candidates_mcid) or all(
                            id < 0 for id in candidates_mcid
                        ):
                            # mcids are all positive or all negative
                            deltaR = [
                                self.deltaR(particle_truth, part_det)
                                for part_det in candidates
                            ]
                            matched_det_idx = candidates_idx[np.argmin(deltaR)]
                        else:
                            # mcids are mixed +ve and -ve
                            candidates_abs = [
                                (part_det, idx)
                                for part_det, mcid, idx in zip(
                                    candidates, candidates_mcid, candidates_idx
                                )
                                if mcid > 0
                            ]
                            deltaR = [
                                self.deltaR(particle_truth, part_det)
                                for part_det, idx in candidates_abs
                            ]
                            _, matched_det_idx = candidates_abs[np.argmin(deltaR)]

                    matched_det = fj_particles_det[matched_det_idx]
                    matched_det_info = matched_det.python_info()

                    info_truth.particle_det = matched_det
                    matched_det_info.particle_truth = particle_truth

                    fj_particles_truth[i_truth].set_python_info(info_truth)
                    fj_particles_det[matched_det_idx].set_python_info(matched_det_info)
            logger.debug("MC ID matching completed.")

        # add associated truth info and charge info in fj_particles_det using the JetInfo object
        # HACK: don't need mcid here really, and charge is already being converted so no need for this
        if self.ENC_fastsim:
            for index, mcid in enumerate(particles_mcid_det):
                if fj_particles_det[index].has_user_info():
                    ecorr_user_info = fj_particles_det[index].python_info()
                    logger.warning("User info already found?")
                else:
                    ecorr_user_info = jet_info.JetInfo()
                if mcid < 0 or mcid >= len(fj_particles_truth):
                    logger.warning(
                        f"MCIndex out of range: {mcid} with index {index}, max {len(fj_particles_truth)}"
                    )
                else:
                    ecorr_user_info.mcid = int(mcid)
                    ecorr_user_info.particle_truth = fj_particles_truth[int(mcid)]
                    # ecorr_user_info.charge = particles_charge_truth[int(mcid)]
                    ecorr_user_info.charge = PDGID(
                        particles_pid_truth[int(mcid)]
                    ).charge
                fj_particles_det[index].set_python_info(ecorr_user_info)

            for index in range(len(fj_particles_truth)):
                if fj_particles_truth[index].has_user_info():
                    ecorr_user_info = fj_particles_truth[index].python_info()
                    logger.warning("User info already found?")
                else:
                    ecorr_user_info = jet_info.JetInfo()
                ecorr_user_info.particle_truth = fj_particles_truth[index]
                ecorr_user_info.mcid = fj_particles_truth[index]
                # ecorr_user_info.charge = particles_charge_truth[index]
                ecorr_user_info.charge = PDGID(particles_pid_truth[index]).charge
                fj_particles_truth[index].set_python_info(ecorr_user_info)
                # fj_particles_truth[index].set_user_index(int(index))

        if self.dry_run:
            return

        self.fill_particle_trees(fj_particles_det, fj_particles_truth)

        # Loop through jetR, and process event for each R
        for jetR in self.jetR_list:
            # Set jet definition and a jet selector
            jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
            # jet_selector_det = fj.SelectorPtMin(5.0) & fj.SelectorAbsRapMax(0.9 - jetR)
            # jet_selector_truth_matched = fj.SelectorPtMin(5.0) & fj.SelectorAbsRapMax(0.9)
            jet_selector_det = fj.SelectorPtMin(5.0) & fj.SelectorAbsEtaMax(0.9 - jetR)
            jet_selector_truth_matched = fj.SelectorPtMin(5.0) & fj.SelectorAbsEtaMax(0.9)
            logger.debug(f"Jet definition: {jet_def}")
            logger.debug(f"Jet selector (det, gen): {jet_selector_det}")
            logger.debug(
                f"Jet selector (gen match): {jet_selector_truth_matched}",
            )

            # Analyze
            if self.is_pp:
                # Find pp det and truth jets
                if self.ENC_fastsim:
                    # FIX ME: should treat long lived charged particle differently (check how the existing fast herwig and pythia handles it)
                    fj_particles_det_ch = fj.vectorPJ()
                    for part in fj_particles_det:
                        if (
                            part.python_info().charge != 0
                        ):  # only use charged particles HACK: commented out line, using the one after it
                            fj_particles_det_ch.append(part)
                    cs_det = fj.ClusterSequence(fj_particles_det_ch, jet_def)
                else:
                    cs_det = fj.ClusterSequence(fj_particles_det, jet_def)

                jets_det_pp = fj.sorted_by_pt(cs_det.inclusive_jets())
                
                # make sure the user info (on the jet side) for jets are all empty right after the jet-clustering
                for jet in jets_det_pp:
                    if jet.has_user_info():
                        jet.python_info().clear_jet_info()
                jets_det_pp_selected = jet_selector_det(jets_det_pp)

                if self.ENC_fastsim:
                    # FIXME: should treat long lived charged particle differently (check how the existing fast herwig and pythia handles it)
                    fj_particles_truth_ch = fj.vectorPJ()
                    for part in fj_particles_truth:
                        if (part.python_info().charge != 0):
                            fj_particles_truth_ch.append(part)
                    cs_truth = fj.ClusterSequence(fj_particles_truth_ch, jet_def)
                else:
                    cs_truth = fj.ClusterSequence(fj_particles_truth, jet_def)

                jets_truth = fj.sorted_by_pt(cs_truth.inclusive_jets())
                # make sure the user info (on the jet side) for jets are all empty right after the jet-clustering
                for jet in jets_truth:
                    if jet.has_user_info():
                        jet.python_info().clear_jet_info()

                    # logger.warning("too large")
                jets_truth_selected = jet_selector_det(jets_truth)
                jets_truth_selected_matched = jet_selector_truth_matched(jets_truth)

                self.analyze_jets(
                    jets_det_pp_selected,
                    jets_truth_selected,
                    jets_truth_selected_matched,
                    jetR,
                )

    def get_branch_type(self, name):
        return getattr(self, name).GetTitle().split("/")[-1]

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

    def analyze_jets(
        self,
        jets_det_selected,
        jets_truth_selected,
        jets_truth_selected_matched,
        jetR,
        jets_det_pp_selected=None,
        R_max=None,
        fj_particles_det_holes=None,
        fj_particles_truth_holes=None,
        rho_bge=0,
        fj_particles_det_cones=None,
        fj_particles_truth_cones=None,
    ):
        logger.debug(f"Number of det-level jets: {len(jets_det_selected)}")

        # Loop through jets and set jet matching candidates for each jet in user_info
        if self.is_pp:
            # [[self.set_matching_candidates(jet_det, jet_truth, jetR, 'hDeltaR_All_R{}'.format(jetR)) for jet_truth in jets_truth_selected_matched] for jet_det in jets_det_selected]
            [
                [
                    self.set_matching_candidates(jet_det, jet_truth, jetR, "")
                    for jet_truth in jets_truth_selected_matched
                ]
                for jet_det in jets_det_selected
            ]

        # Loop through jets and set accepted matches
        if self.is_pp:
            [self.set_matches_pp(jet_det, "") for jet_det in jets_det_selected]

        # make sure truth jets have det-match information attached
        for jet_det in jets_det_selected:
            if jet_det.has_user_info():
                if jet_det.python_info().match:
                    jet_truth = jet_det.python_info().match
                    truth_info = (
                        jet_truth.python_info()
                        if jet_truth.has_user_info()
                        else jet_info.JetInfo()
                    )
                    truth_info.match = jet_det
                    jet_truth.set_python_info(truth_info)

        for jet_det in jets_det_selected:
            # Check additional acceptance criteria
            # skip event if not satisfied -- since first jet in event is highest pt
            # if not self.utils.is_det_jet_accepted(jet_det):
            #     self.hNevents.Fill(0)
            #     logger.warning('Jets in this event rejected due to jet acceptance.')
            #     return

            self.fill_jet_det_trees(jet_det, jetR, R_max, rho_bge)
            self.fill_pair_det_trees(jet_det)

        # Fill truth-level jet histograms (before matching)
        if self.is_pp or self.fill_Rmax_indep_hists:
            for jet_truth in jets_truth_selected_matched:
                self.fill_jet_gen_trees(jet_truth, jetR)
                self.fill_pair_gen_trees(jet_truth, jetR)

    def fill_jet_gen_trees(self, jet, jetR):
        jet_gen_pT_arr = array(self.tree2arr[self.get_branch_type("jet_gen_pT")], [0.0])
        getattr(self, "jet_gen_pT").SetAddress(jet_gen_pT_arr)
        jet_gen_nconst_arr = array(
            self.tree2arr[self.get_branch_type("jet_gen_nconst")], [0]
        )
        getattr(self, "jet_gen_nconst").SetAddress(jet_gen_nconst_arr)
        jet_gen_eta_arr = array(
            self.tree2arr[self.get_branch_type("jet_gen_eta")], [0.0]
        )
        getattr(self, "jet_gen_eta").SetAddress(jet_gen_eta_arr)
        jet_gen_phi_arr = array(
            self.tree2arr[self.get_branch_type("jet_gen_phi")], [0.0]
        )
        getattr(self, "jet_gen_phi").SetAddress(jet_gen_phi_arr)
        jet_gen_match_pT_arr = array(
            self.tree2arr[self.get_branch_type("jet_gen_match_pT")], [0]
        )
        getattr(self, "jet_gen_match_pT").SetAddress(jet_gen_match_pT_arr)

        jet_gen_pT_arr[0] = jet.pt()
        # jet_gen_eta_arr[0] = jet.rap()
        jet_gen_eta_arr[0] = jet.eta()
        jet_gen_phi_arr[0] = jet.phi()
        jet_gen_nconst_arr[0] = len(jet.constituents())
        jet_gen_match_pT_arr[0] = (
            jet.python_info().match.pt()
            if jet.has_user_info() and jet.python_info().match is not None
            else -1
        )
        self.jet_gen_tree.Fill()

    def fill_jet_det_trees(self, jet, jetR, R_max, rho_bge=0):
        jet_det_pT_arr = array(self.tree2arr[self.get_branch_type("jet_det_pT")], [0.0])
        getattr(self, "jet_det_pT").SetAddress(jet_det_pT_arr)
        jet_det_nconst_arr = array(
            self.tree2arr[self.get_branch_type("jet_det_nconst")], [0]
        )
        getattr(self, "jet_det_nconst").SetAddress(jet_det_nconst_arr)
        jet_det_eta_arr = array(
            self.tree2arr[self.get_branch_type("jet_det_eta")], [0.0]
        )
        getattr(self, "jet_det_eta").SetAddress(jet_det_eta_arr)
        jet_det_phi_arr = array(
            self.tree2arr[self.get_branch_type("jet_det_phi")], [0.0]
        )
        getattr(self, "jet_det_phi").SetAddress(jet_det_phi_arr)
        jet_det_match_pT_arr = array(
            self.tree2arr[self.get_branch_type("jet_det_match_pT")], [0]
        )
        getattr(self, "jet_det_match_pT").SetAddress(jet_det_match_pT_arr)

        jet_det_pT_arr[0] = jet.pt()
        jet_det_eta_arr[0] = jet.eta()
        jet_det_phi_arr[0] = jet.phi()
        jet_det_nconst_arr[0] = len(jet.constituents())
        jet_det_match_pT_arr[0] = (
            jet.python_info().match.pt()
            if jet.has_user_info() and jet.python_info().match is not None
            else -1
        )
        self.jet_det_tree.Fill()

    def fill_pair_gen_trees(self, jet, jetR):
        jet_pT = jet.pt()
        match_jet_pT = (
            jet.python_info().match.pt()
            if jet.has_user_info() and jet.python_info().match is not None
            else -1
        )
        jet_in_acceptance = abs(jet.eta()) < (0.9 - jetR)
        # jet_in_acceptance = abs(jet.rap()) < (0.9 - jetR)

        pair_gen_pT1_arr = array(
            self.tree2arr[self.get_branch_type("pair_gen_pT1")], [0.0]
        )
        getattr(self, "pair_gen_pT1").SetAddress(pair_gen_pT1_arr)
        pair_gen_pT2_arr = array(
            self.tree2arr[self.get_branch_type("pair_gen_pT2")], [0.0]
        )
        getattr(self, "pair_gen_pT2").SetAddress(pair_gen_pT2_arr)
        pair_gen_q1_arr = array(self.tree2arr[self.get_branch_type("pair_gen_q1")], [0])
        getattr(self, "pair_gen_q1").SetAddress(pair_gen_q1_arr)
        pair_gen_q2_arr = array(self.tree2arr[self.get_branch_type("pair_gen_q2")], [0])
        getattr(self, "pair_gen_q2").SetAddress(pair_gen_q2_arr)

        pair_gen_weight_arr = array(
            self.tree2arr[self.get_branch_type("pair_gen_weight")], [0.0]
        )
        getattr(self, "pair_gen_weight").SetAddress(pair_gen_weight_arr)
        pair_gen_RL_arr = array(
            self.tree2arr[self.get_branch_type("pair_gen_RL")], [0.0]
        )
        getattr(self, "pair_gen_RL").SetAddress(pair_gen_RL_arr)
        pair_gen_deta_arr = array(
            self.tree2arr[self.get_branch_type("pair_gen_deta")], [0.0]
        )
        getattr(self, "pair_gen_deta").SetAddress(pair_gen_deta_arr)
        pair_gen_dphi_arr = array(
            self.tree2arr[self.get_branch_type("pair_gen_dphi")], [0.0]
        )
        getattr(self, "pair_gen_dphi").SetAddress(pair_gen_dphi_arr)

        pair_gen_pTjet_arr = array(
            self.tree2arr[self.get_branch_type("pair_gen_pTjet")], [jet_pT]
        )
        getattr(self, "pair_gen_pTjet").SetAddress(pair_gen_pTjet_arr)
        pair_gen_match_pT_arr = array(
            self.tree2arr[self.get_branch_type("pair_gen_match_pT")], [match_jet_pT]
        )
        getattr(self, "pair_gen_match_pT").SetAddress(pair_gen_match_pT_arr)
        pair_gen_jet_in_acceptance_arr = array(
            self.tree2arr[self.get_branch_type("pair_gen_jet_in_acceptance")],
            [jet_in_acceptance],
        )
        getattr(self, "pair_gen_jet_in_acceptance").SetAddress(
            pair_gen_jet_in_acceptance_arr
        )

        # # for p1, p2 in itertools.product(jet.constituents(), repeat = 2):
        for p1, p2 in itertools.permutations(jet.constituents(), 2):

            pair_gen_pT1_arr[0] = p1.pt()
            pair_gen_pT2_arr[0] = p2.pt()
            pair_gen_q1_arr[0] = p1.python_info().charge
            pair_gen_q2_arr[0] = p2.python_info().charge
            pair_gen_weight_arr[0] = p1.pt() * p2.pt() / (jet_pT * jet_pT)
            pair_gen_RL_arr[0] = self.deltaR(p1, p2)
            pair_gen_deta_arr[0] = p2.eta() - p1.eta()
            pair_gen_dphi_arr[0] = p1.delta_phi_to(p2)

            self.pair_gen_tree.Fill()

    def fill_pair_det_trees(self, jet):
        jet_pT = jet.pt()
        match_jet_pT = (
            jet.python_info().match.pt()
            if jet.has_user_info() and jet.python_info().match is not None
            else -1
        )

        pair_det_pT1_arr = array(
            self.tree2arr[self.get_branch_type("pair_det_pT1")], [0.0]
        )
        getattr(self, "pair_det_pT1").SetAddress(pair_det_pT1_arr)
        pair_det_pT2_arr = array(
            self.tree2arr[self.get_branch_type("pair_det_pT2")], [0.0]
        )
        getattr(self, "pair_det_pT2").SetAddress(pair_det_pT2_arr)
        pair_det_q1_arr = array(self.tree2arr[self.get_branch_type("pair_det_q1")], [0])
        getattr(self, "pair_det_q1").SetAddress(pair_det_q1_arr)
        pair_det_q2_arr = array(self.tree2arr[self.get_branch_type("pair_det_q2")], [0])
        getattr(self, "pair_det_q2").SetAddress(pair_det_q2_arr)

        pair_det_weight_arr = array(
            self.tree2arr[self.get_branch_type("pair_det_weight")], [0.0]
        )
        getattr(self, "pair_det_weight").SetAddress(pair_det_weight_arr)
        pair_det_RL_arr = array(
            self.tree2arr[self.get_branch_type("pair_det_RL")], [0.0]
        )
        getattr(self, "pair_det_RL").SetAddress(pair_det_RL_arr)
        pair_det_deta_arr = array(
            self.tree2arr[self.get_branch_type("pair_det_deta")], [0.0]
        )
        getattr(self, "pair_det_deta").SetAddress(pair_det_deta_arr)
        pair_det_dphi_arr = array(
            self.tree2arr[self.get_branch_type("pair_det_dphi")], [0.0]
        )
        getattr(self, "pair_det_dphi").SetAddress(pair_det_dphi_arr)

        pair_det_pTjet_arr = array(
            self.tree2arr[self.get_branch_type("pair_det_pTjet")], [jet_pT]
        )
        getattr(self, "pair_det_pTjet").SetAddress(pair_det_pTjet_arr)
        pair_det_match_pT_arr = array(
            self.tree2arr[self.get_branch_type("pair_det_match_pT")], [match_jet_pT]
        )
        getattr(self, "pair_det_match_pT").SetAddress(pair_det_match_pT_arr)

        # for p1, p2 in itertools.product(jet.constituents(), repeat = 2):
        for p1, p2 in itertools.permutations(jet.constituents(), 2):

            pair_det_pT1_arr[0] = p1.pt()
            pair_det_pT2_arr[0] = p2.pt()
            pair_det_q1_arr[0] = p1.python_info().charge
            pair_det_q2_arr[0] = p2.python_info().charge
            pair_det_weight_arr[0] = p1.pt() * p2.pt() / (jet_pT * jet_pT)
            pair_det_RL_arr[0] = self.deltaR(p1, p2)
            pair_det_deta_arr[0] = p2.eta() - p1.eta()
            pair_det_dphi_arr[0] = p1.delta_phi_to(p2)

            self.pair_det_tree.Fill()


##################################################################
if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser(description="Process MC for ENC trees.")
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

    analysis = Generator_Tree_MC_ENC(
        input_file=args.input_file,
        config_file=args.config_file,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
    )
    analysis.process_mc()
