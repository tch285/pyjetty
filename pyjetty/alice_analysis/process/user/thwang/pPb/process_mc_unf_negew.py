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

# Data analysis and plotting
import numpy as np
import ROOT
import yaml

# Fastjet via python (from external library heppy)
import fastjet as fj
import ecorrel

# Analysis utilities
# from pyjetty.alice_analysis.process.base import process_io
from pyjetty.alice_analysis.process.user.substructure import process_mc_base_pPb

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

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

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(funcName)s - %(message)s')
handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'))

logger.addHandler(handler)
logger.setLevel(logging.INFO)

def linbins(xmin, xmax, nbins):
  return np.linspace(xmin, xmax, nbins+1)

def logbins(xmin, xmax, nbins):
  return np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)

# n_bins_truth = [20, 22, 7] # WARNING RooUnfold seg faults if too many bins used
# # these are the truth level binnings
# binnings_truth = [np.logspace(-5,0,n_bins_truth[0]+1), \
#             np.logspace(-2.299,0,n_bins_truth[1]+1), \
#             np.array([5, 10, 20, 40, 60, 80, 100, 150]).astype(float) ]
# # slight difference for reco pt bin
# n_bins = [20, 22, 6]
# binnings = [np.logspace(-5,0,n_bins[0]+1), \
#             np.logspace(-2.299,0,n_bins[1]+1), \
#             np.array([10, 20, 40, 60, 80, 100, 150]).astype(float) ]

################################################################
class EEC_pair:
  def __init__(self, _mcid1, _mcid2, _weight, _r, _pt, has_match: bool, ctype: str, ptype: str):
    self.mcid1 = _mcid1
    self.mcid2 = _mcid2
    self.weight = _weight
    self.r = _r
    self.pt = _pt
    self.has_match = has_match
    if ctype not in ['PM', 'P', 'M']:
      raise ValueError(f"Invalid ctype {ctype}")
    else:
      self.ctype = ctype
    if ptype not in ['jj', 'jp', 'pp', 'mx']:
      raise ValueError(f"Invalid ptype {ptype}")
    else:
      self.ptype = ptype

  def is_equal(self, other): # check ptype
    return self.has_match and other.has_match and self.is_same_ptype(other) and \
      ((self.mcid1 == other.mcid1 and self.mcid2 == other.mcid2) \
      or (self.mcid1 == other.mcid2 and self.mcid2 == other.mcid1))
  def is_same_ptype(self, other):
    return self.ptype == other.ptype
  def is_same_ctype(self, other):
    return self.ctype == other.ctype

  def __str__(self):
    return "EEC pair with (mcid1, mcid2, weight, RL, pt) = (" + \
      str(self.mcid1) + ", " + str(self.mcid2) + ", " + str(self.weight) + \
      ", " + str(self.r) + ", " + str(self.pt) + ")"

################################################################
class ProcessMC_ENC(process_mc_base_pPb.ProcessMCBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
  
    # Initialize base class
    super(ProcessMC_ENC, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)
    # find pt_hat for set of events in input_file, assumes all events in input_file are in the same pt_hat bin
    if "LHC23a3" in input_file or "LHC18b8" in input_file:
      self.pt_hat_bin = int(input_file.split('/')[len(input_file.split('/'))-4]) # depends on exact format of input_file name
      with open("/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/data/LHC18b8/scaleFactors.yaml", 'r') as stream:
        pt_hat_yaml = yaml.safe_load(stream)
      self.pt_hat = pt_hat_yaml[self.pt_hat_bin]
      logger.info(f"pt hat bin: {self.pt_hat_bin}")
      logger.info(f"pt hat weight: {self.pt_hat}")
    elif "LHC18f3" in input_file:
      self.pt_hat_bin = 0
      self.pt_hat = 1
      logger.info("No pT hat weight, using unity weight.")
    else:
      raise ValueError("Can't find the right pT hat bin from input file.")
    with open(self.config_file, 'r') as stream:
      config = yaml.safe_load(stream)
    self.thrd = config['unfold']['config1']['thrd']
    
    self.weight_min, self.weight_max, self.weight_nbins = config["weight_binning"]
    posbins = logbins(self.weight_min, self.weight_max, self.weight_nbins)
    negbins = - np.flip(logbins(self.weight_min, self.weight_max, self.weight_nbins))
    self.weight_bins = np.append(negbins, posbins)
    self.weight_nbins = len(self.weight_bins) - 1
    # self.weight_bins = logbins(self.weight_min, self.weight_max, self.weight_nbins)

    self.RL_min, self.RL_max, self.RL_nbins = config["RL_binning"]
    self.RL_bins = logbins(self.RL_min, self.RL_max, self.RL_nbins)

    self.pT_truth_bins = np.array(config['pT_truth_bins'])
    self.pT_truth_nbins = len(self.pT_truth_bins) - 1
    self.pT_det_bins = np.array(config['pT_det_bins'])
    self.pT_det_nbins = len(self.pT_det_bins) - 1

    self.jetpt_min_det = config['jetpt_min_det']
    self.jetpt_min_truth = config['jetpt_min_truth']

  #---------------------------------------------------------------
  # Calculate pair distance of two fastjet particles
  #---------------------------------------------------------------
  def deltaR(self, p1, p2):
    return np.sqrt(p1.delta_phi_to(p2) ** 2 + (p1.eta() - p2.eta()) ** 2)

  #---------------------------------------------------------------
  # Initialize histograms
  #---------------------------------------------------------------
  def initialize_user_output_objects_R(self, jetR):
    for ctype in ['P', 'M', 'PM', 'T']:
      h3_reco = ROOT.TH3D(f"reco_{ctype}", f"reco_{ctype}", self.weight_nbins, self.weight_bins, self.RL_nbins, self.RL_bins, self.pT_det_nbins, self.pT_det_bins)
      setattr(self, f"reco_{ctype}", h3_reco)
      h3_gen = ROOT.TH3D(f"gen_{ctype}", f"gen_{ctype}", self.weight_nbins, self.weight_bins, self.RL_nbins, self.RL_bins, self.pT_truth_nbins, self.pT_truth_bins)
      setattr(self, f"gen_{ctype}", h3_gen)
      # for purity correction
      name = f'reco_unmatched_{ctype}'
      h = ROOT.TH3D(name, name, self.weight_nbins, self.weight_bins, self.RL_nbins, self.RL_bins, self.pT_det_nbins, self.pT_det_bins)
      setattr(self, name, h)

      name = f"response_{ctype}"
      response = ROOT.RooUnfoldResponse(h3_reco, h3_gen, name, name)
      setattr(self, name, response)

    h1_reco = ROOT.TH1D("reco1D", "reco1D", self.pT_det_nbins, self.pT_det_bins)
    setattr(self, "reco1D", h1_reco)
    h1_gen = ROOT.TH1D("gen1D", "gen1D", self.pT_truth_nbins, self.pT_truth_bins)
    setattr(self, "gen1D", h1_gen)

    response_matrix = ROOT.TH2D("response1D", "response1D", 
                                self.pT_det_nbins, self.pT_det_bins,
                                self.pT_truth_nbins, self.pT_truth_bins)
    response1D = ROOT.RooUnfoldResponse(h1_reco, h1_gen, response_matrix, "response1D", "response1D")
    setattr(self, "response1D", response1D)

    # efficiency and purity check for 1D jet pT unfolding
    name = 'gen1D_unmatched'
    h = ROOT.TH1D("gen1D_unmatched", "gen1D_unmatched", self.pT_truth_nbins, self.pT_truth_bins)
    setattr(self, name, h)

    name = 'reco1D_unmatched'
    h = ROOT.TH1D("reco1D_unmatched", "reco1D_unmatched", self.pT_det_nbins, self.pT_det_bins)
    setattr(self, name, h)

  def analyze_event(self, fj_particles_det, fj_particles_truth, fj_particles_det_holes=None, fj_particles_truth_holes=None,
                    particles_mcid_det=None, particles_pid_truth=None, mult = None):
    self.event_number += 1
    # if self.event_number > self.event_number_max:
    #     return

    if len(fj_particles_truth) == 0:
      logger.warning(f"Event {self.event_number} has {len(fj_particles_det)} det tracks and {len(fj_particles_truth)} truth tracks, skipping.")
      return
    if len(fj_particles_det) == 0:
      logger.warning(f"Event {self.event_number} has {len(fj_particles_det)} det tracks and {len(fj_particles_truth)} truth tracks.")

    fj_particles_det_pass = fj.vectorPJ()
    for part in fj_particles_det:
      if part.perp() > 0.15:
        fj_particles_det_pass.push_back(part)
      else:
        logger.warning('Found particle with pT less than 150 MeV in det sample.')
    fj_particles_det = fj_particles_det_pass

    fj_particles_truth_pass = fj.vectorPJ()
    for part in fj_particles_truth:
      if part.perp() > 0.15:
        fj_particles_truth_pass.push_back(part)
      else:
        logger.warning('Found particle with pT less than 150 MeV in truth sample.')
    fj_particles_truth = fj_particles_truth_pass

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
            # mcids are all +ve or all -ve
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

    # if len(fj_particles_truth) > 1:
    #   if np.abs(fj_particles_truth[0].pt() - fj_particles_truth[1].pt()) <  1e-10:
    #     logger.warning('Duplicate particles may be present'
    #                   f'{[p.user_index() for p in fj_particles_truth]}'
    #                   f'{[p.pt() for p in fj_particles_truth]}')

    #HACK: pretty sure the old bug is here still
    # but how to make sure the multiplicities match up when stitching det and truth?
    # self.mult_det = self.df_events_det['V0Amult'].values[self.event_number-1]
    # self.mult_truth = self.df_events_truth['V0Amult'].values[self.event_number-1]
    #HACK: placeholder for now
    self.mult_det = 0
    self.mult_truth = 0

    # Loop through jetR, and process event for each R
    for jetR in self.jetR_list:
      jet_def = self.jet_defs[jetR]
      jet_selector_det = self.jet_selectors_det[jetR]
      Cjet_selector = self.Cjet_selectors[jetR]

      if self.do_median_subtraction:
        csa_medsub_det = fj.ClusterSequenceArea(fj_particles_det, self.jet_def_medsub[jetR], fj.AreaDefinition(fj.active_area_explicit_ghosts))
        self.median_subtractor[jetR].set_cluster_sequence(csa_medsub_det)
        rho_det = self.median_subtractor[jetR].rho()

        # medsub_selected_jets = fj.sorted_by_pt(Cjet_selector(csa_medsub.inclusive_jets()))
        medsub_selected_jets_det = Cjet_selector(csa_medsub_det.inclusive_jets())
        C_area_det = np.sum([jet.area() for jet in medsub_selected_jets_det]) / (2 * np.pi * 0.9 * 2)

        csa_medsub_truth = fj.ClusterSequenceArea(fj_particles_truth, self.jet_def_medsub[jetR], fj.AreaDefinition(fj.active_area_explicit_ghosts))
        self.median_subtractor_truth[jetR].set_cluster_sequence(csa_medsub_truth)
        rho_truth = self.median_subtractor_truth[jetR].rho()
        # medsub_selected_jets_truth = fj.sorted_by_pt(Cjet_selector(csa_medsub_truth.inclusive_jets()))
        medsub_selected_jets_truth = Cjet_selector(csa_medsub_truth.inclusive_jets())
        C_area_truth = np.sum([jet.area() for jet in medsub_selected_jets_truth]) / (2 * np.pi * 2 * 0.9)
      else:
        rho_det, rho_truth, C_area_det, C_area_truth = 0, 0, 0, 0

      # Analyze
      cs_det = fj.ClusterSequenceArea(fj_particles_det, jet_def, fj.AreaDefinition(fj.VoronoiAreaSpec()))
      jets_det = fj.sorted_by_pt(cs_det.inclusive_jets())
      jets_det_selected = jet_selector_det(jets_det)

      # cs_truth = fj.ClusterSequenceArea(fj_particles_truth, jet_def, fj.AreaDefinition(fj.active_area_explicit_ghosts))
      # cs_truth = fj.ClusterSequenceArea(fj_particles_truth, jet_def, fj.AreaDefinition(fj.active_area))
      cs_truth = fj.ClusterSequenceArea(fj_particles_truth, jet_def, fj.AreaDefinition(fj.VoronoiAreaSpec()))
      jets_truth = fj.sorted_by_pt(cs_truth.inclusive_jets())
      # jets_truth_selected = jet_selector_det(jets_truth)
      jets_truth_selected_matched = self.jet_selector_truth_matched(jets_truth)

      if self.do_median_subtraction:
        jets_det_reselected = fj.vectorPJ()
        [jets_det_reselected.push_back(jet) for jet in jets_det_selected if jet.pt() - rho_det * jet.area() * C_area_det > self.jetpt_min_det]
        jets_truth_reselected = fj.vectorPJ()
        [jets_truth_reselected.push_back(jet) for jet in jets_truth_selected_matched if jet.pt() - rho_truth * jet.area() * C_area_truth > self.jetpt_min_truth]

        if self.do_perpendicular_cone:
          jets_det_reselected_wpcone = [self.attach_perp_cones(fj_particles_det, jet, coneR = jetR) for jet in jets_det_reselected]
          jets_truth_selected_wpcone = [self.attach_perp_cones(fj_particles_truth, jet, coneR = jetR) for jet in jets_truth_reselected]
          self.analyze_jets(jets_det_reselected_wpcone, jets_truth_selected_wpcone, jetR, rho_det * C_area_det, rho_truth * C_area_truth)

  #---------------------------------------------------------------
  # Analyze jets of a given event.
  #---------------------------------------------------------------
  def analyze_jets(self, jets_det, jets_truth, jetR, rho_bge_det, rho_bge_truth):
    for jet in jets_det:
      getattr(self, 'reco1D_unmatched').Fill(jet.perp() - rho_bge_det * jet.area(), self.pt_hat)

    ############################## JET MATCHING ##############################
    # perform jet-matching, every det jet has a guaranteed truth jet match
    det_used = []
    for t_jet in jets_truth:
      candidates = []
      candidates_pt = []
      t_jet_pt_sub = t_jet.pt() - rho_bge_truth * t_jet.area()

      if t_jet.eta() < 0.9 - jetR:
        getattr(self, 'gen1D_unmatched').Fill(t_jet_pt_sub, self.pt_hat)

      # for i in range(jets_det.size()):
      #   d_jet = jets_det[i]

      #   if self.deltaR(t_jet, d_jet) < 0.2 and d_jet not in det_used:
      #     candidates.append(d_jet)
      #     candidates_pt.append(d_jet.perp() - rho_bge_det * d_jet.area())
      candidates = [d_jet for d_jet in jets_det if self.deltaR(t_jet, d_jet) < 0.2 and d_jet not in det_used]
      candidates_pt = np.array([d_jet.perp() - rho_bge_det * d_jet.area() for d_jet in candidates])

      # if match found
      if candidates:
        idx = np.argmin(np.abs(candidates_pt - t_jet_pt_sub))
        det_match = candidates[idx]
        det_match_pt = candidates_pt[idx]
        det_used.append(det_match)

        getattr(self, "reco1D").Fill(det_match_pt, self.pt_hat) # already subtracted
        getattr(self, "gen1D").Fill(t_jet_pt_sub, self.pt_hat)
        getattr(self, "response1D").Fill(det_match_pt, t_jet_pt_sub, self.pt_hat)

        self.fill_matched_jet_histograms(det_match, t_jet, rho_bge_det, rho_bge_truth)

      # if match not found, DONT DO ANYTHING, we had a whole conversation about this...

  def fill_matched_jet_histograms(self, det_jet, truth_jet, rho_bge_det, rho_bge_truth):
    # truth level EEC pairs
    truth_pairs = self.get_EEC_pairs(truth_jet, ipoint=2, is_truth = True, rho_bge = rho_bge_truth)

    # det level EEC pairs
    det_pairs = self.get_EEC_pairs(det_jet, ipoint=2, is_truth = False, rho_bge = rho_bge_det)

    ######### purity correction #########
    # calculate det EEC cross section irregardless if truth match exists

    for d_pair in det_pairs:
       ptype_f = self.get_pair_type_factor(d_pair.ptype)
       ctype = d_pair.ctype
       getattr(self,  "reco_unmatched_T").Fill(d_pair.weight * ptype_f, d_pair.r, d_pair.pt, self.pt_hat)
       getattr(self, f"reco_unmatched_{ctype}").Fill(d_pair.weight * ptype_f, d_pair.r, d_pair.pt, self.pt_hat)

    ########################## TTree output generation #########################
    # composite of truth and smeared pairs, fill the TTree preprocessed
    # dummyval = -9999

    # pair matching
    for t_pair in truth_pairs:
      ptype_f = self.get_pair_type_factor(t_pair.ptype)
      getattr(self, "gen_T").Fill(t_pair.weight * ptype_f, t_pair.r, t_pair.pt, self.pt_hat)

      for d_pair in det_pairs:
        if d_pair.is_equal(t_pair): # no need to check ctype, but ptype is checked
          getattr(self, "reco_T").Fill(d_pair.weight * ptype_f, d_pair.r, d_pair.pt, self.pt_hat)
          getattr(self, "response_T").Fill(d_pair.weight * ptype_f, d_pair.r, d_pair.pt, t_pair.weight * ptype_f, t_pair.r, t_pair.pt, self.pt_hat)
          break
      else: # miss if break is not encountered
        getattr(self, "response_T").Miss(t_pair.weight * ptype_f, t_pair.r, t_pair.pt, self.pt_hat)

    # run again for ctype check since else statement can't distinguish between matching ptype and 
    # mismatched ctype (ok for incl. EEC) and matched ptype AND ctype (needed for charged EEC)
    for t_pair in truth_pairs:
      ctype = t_pair.ctype
      ptype_f = self.get_pair_type_factor(t_pair.ptype)
      getattr(self, f"gen_{ctype}").Fill(t_pair.weight * ptype_f, t_pair.r, t_pair.pt, self.pt_hat)

      for d_pair in det_pairs:
        if d_pair.is_equal(t_pair) and d_pair.is_same_ctype(t_pair):
          getattr(self, f"reco_{ctype}").Fill(d_pair.weight * ptype_f, d_pair.r, d_pair.pt, self.pt_hat)
          getattr(self, f"response_{ctype}").Fill(d_pair.weight * ptype_f, d_pair.r, d_pair.pt, t_pair.weight * ptype_f, t_pair.r, t_pair.pt, self.pt_hat)
          break
      else: # miss if break is not encountered
        getattr(self, f"response_{ctype}").Miss(t_pair.weight * ptype_f, t_pair.r, t_pair.pt, self.pt_hat)

  def get_EEC_pairs(self, jet, ipoint, is_truth, rho_bge):
    pairs = []

    jet_pt_sub = jet.perp() - rho_bge * jet.area()

    # push constituents to a vector in python
    # reapply pt cut incase of ghosts
    # _v = fj.vectorPJ()
    # for c in jet.constituents():
    #   if c.perp() > 1:
    #      _v.push_back(c)

    constituents = fj.sorted_by_pt(jet.constituents())
    c_select = fj.vectorPJ()

    for c in constituents:
      if c.pt() < self.thrd:
        break
      c.set_user_index(0)
      c_select.push_back(c)
    if self.do_perpendicular_cone:
      c_select2 = fj.vectorPJ()
      for c in constituents:
        if c.pt() < self.thrd:
          break
        c.set_user_index(0)
        c_select2.push_back(c)

      for part in jet.python_info().perpcone1:
        if part.pt() > self.thrd:
          c_select.push_back(part)
      for part in jet.python_info().perpcone2:
        if part.pt() > self.thrd:
          c_select2.push_back(part)

    max_npoint = 2
    weight_power = 1
    cb = ecorrel.CorrelatorBuilder(c_select, jet_pt_sub, max_npoint, weight_power, -9999, -9999)
    EEC_cb = cb.correlator(ipoint)

    for indices, rL, weight in zip(EEC_cb.indices(), EEC_cb.rs(), EEC_cb.weights()):
      p1, p2 = [c_select[idx] for idx in indices]
      mcid1, mcid2 = [np.abs(c_select[idx].python_info().mcid) for idx in indices]

      uidx1, uidx2 = [c_select[idx].user_index() for idx in indices]
      ptype = self.get_pair_type(uidx1, uidx2)
      q1, q2 = [c_select[idx].python_info().charge for idx in indices]
      ctype = self.get_charge_type(q1, q2)

      if is_truth:
        has_match = p1.python_info().particle_det is not None and p2.python_info().particle_det is not None 
      else:
        has_match = p1.python_info().particle_truth is not None and p2.python_info().particle_truth is not None 
      pairs.append(EEC_pair(mcid1, mcid2, weight, rL, jet_pt_sub, has_match, ctype, ptype))

    cb2 = ecorrel.CorrelatorBuilder(c_select2, jet_pt_sub, max_npoint, weight_power, -9999, -9999)
    EEC_cb2 = cb2.correlator(ipoint)
    for indices, rL, weight in zip(EEC_cb2.indices(), EEC_cb2.rs(), EEC_cb2.weights()):
      p1, p2 = [c_select2[idx] for idx in indices]
      mcid1, mcid2 = [np.abs(c_select2[idx].python_info().mcid) for idx in indices]

      uidx1, uidx2 = [c_select2[idx].user_index() for idx in indices]
      ptype = self.get_pair_type(uidx1, uidx2)
      if ptype == 'jj':
        continue
      q1, q2 = [c_select2[idx].python_info().charge for idx in indices]
      ctype = self.get_charge_type(q1, q2)

      if is_truth:
        has_match = p1.python_info().particle_det is not None and p2.python_info().particle_det is not None 
      else:
        has_match = p1.python_info().particle_truth is not None and p2.python_info().particle_truth is not None 
      pairs.append(EEC_pair(mcid1, mcid2, weight, rL, jet_pt_sub, has_match, ctype, ptype))

    cone1_w_ptcut = [p for p in jet.python_info().perpcone1 if p.pt() > self.thrd]
    cone2_w_ptcut = [p for p in jet.python_info().perpcone2 if p.pt() > self.thrd]
    for p1, p2 in itertools.product(cone1_w_ptcut, cone2_w_ptcut):
      pair_weight = p1.pt() * p2.pt() / (jet_pt_sub ** 2)
      pair_RL = np.sqrt((p1.eta() - p2.eta()) ** 2 + (p1.delta_phi_to(p2)) ** 2)
      mcid1, mcid2 = np.abs(p1.python_info().mcid), np.abs(p2.python_info().mcid)

      q1 = p1.python_info().charge
      q2 = p2.python_info().charge
      ptype = "mx"
      ctype = self.get_charge_type(q1, q2)
      if is_truth:
        has_match = p1.python_info().particle_det is not None and p2.python_info().particle_det is not None 
      else:
        has_match = p1.python_info().particle_truth is not None and p2.python_info().particle_truth is not None 

      pairs.append(EEC_pair(mcid1, mcid2, pair_weight, pair_RL, jet_pt_sub, has_match, ctype, "mx"))
      # fill twice to get reverse pair order as well
      pairs.append(EEC_pair(mcid2, mcid1, pair_weight, pair_RL, jet_pt_sub, has_match, ctype, "mx"))

    return pairs

  def get_charge_type(self, q1, q2):
    if q1 > 0 and q2 > 0:
      return 'P'
    elif q1 * q2 == -1:
      return 'PM'
    else:
      return 'M'

  def get_pair_type(self, uidx1, uidx2):
    # jet parts have uidx 0, cone1 (+angle) has uidx +1, cone2 (-angle) has uidx -1
    if uidx1 == 0 and uidx2 == 0:
      return 'jj'
    elif uidx1 * uidx2 == 0: # take advantage of lazy if
      return 'jp'
    else:
      return 'pp'

  def get_pair_type_factor(self, ptype):
    if ptype == 'jj' or ptype == 'mx':
      return 1
    else:
      return -1


##################################################################
if __name__ == '__main__':
  # Define arguments
  parser = argparse.ArgumentParser(description='Process MC')
  parser.add_argument('-i', '--input-file', action='store',
                      type=str, metavar='inputFile',
                      default='AnalysisResults.root',
                      help='Path of ROOT file containing TTrees')
  parser.add_argument('-c', '--config-file', action='store',
                      type=str, metavar='configFile',
                      default='config/analysis_config.yaml',
                      help="Path of config file for analysis")
  parser.add_argument('-o', '--output-dir', action='store',
                      type=str, metavar='outputDir',
                      default='./TestOutput',
                      help='Output directory for output to be written to')
  
  # Parse the arguments
  args = parser.parse_args()
  
  logger.info('Configuring...')
  logger.info(f'Input file: {args.input_file}')
  logger.info(f'Config file: {args.config_file}')
  logger.info(f'Output dir: {args.output_dir}')

  # If invalid input file is given, exit
  if not os.path.exists(args.input_file):
    logger.critical(f'File {args.input_file} does not exist! Exiting!')
    sys.exit(2)
  
  # If invalid config_file is given, exit
  if not os.path.exists(args.config_file):
    logger.critical(f'File {args.config_file} does not exist! Exiting!')
    sys.exit(2)

  # perform analysis
  analysis = ProcessMC_ENC(input_file=args.input_file, config_file=args.config_file, output_dir=args.output_dir)
  analysis.process_mc()