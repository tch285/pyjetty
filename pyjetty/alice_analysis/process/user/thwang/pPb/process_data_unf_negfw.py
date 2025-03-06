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

# Data analysis and plotting
import ROOT
import numpy as np
import math
import yaml
import logging

# Fastjet via python (from external library heppy)
import fastjet as fj
import ecorrel

# Base class
from pyjetty.alice_analysis.process.user.substructure import process_data_base_pPb

def linbins(xmin, xmax, nbins):
  return np.linspace(xmin, xmax, nbins+1)

def logbins(xmin, xmax, nbins):
  return np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)

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
      formatted_msg = super().format(record)
      return f"{color}{formatted_msg}{self.RESET}"
    return super().format(record)

logger = logging.getLogger(__name__)

################################################################
class ProcessData_ENC(process_data_base_pPb.ProcessDataBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
    # Initialize base class
    super(ProcessData_ENC, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)

    self.observable = self.observable_list[0]

    with open(self.config_file, 'r') as stream:
      config = yaml.safe_load(stream)

    self.weight_min, self.weight_max, self.weight_nbins = config["weight_binning"]
    self.weight_bins = logbins(self.weight_min, self.weight_max, self.weight_nbins)

    self.RL_min, self.RL_max, self.RL_nbins = config["RL_binning"]
    self.RL_bins = logbins(self.RL_min, self.RL_max, self.RL_nbins)
    self.pT_det_bins = np.array(config['pT_det_bins'])
    self.pT_det_nbins = len(self.pT_det_bins) - 1

    self.jetpt_min_det = config['jetpt_min_det'] if 'jetpt_min_det' in config else 5
    self.jetpt_min_det_subtracted = config['jetpt_min_det_subtracted'] if 'jetpt_min_det_subtracted' in config else 5

    self.thrd = config['unfold']['config1']['thrd']

  #---------------------------------------------------------------
  # Initialize histograms
  #---------------------------------------------------------------
  def initialize_user_output_objects(self):
    h1_raw =  ROOT.TH1D("raw1D", "raw1D", self.pT_det_nbins, self.pT_det_bins)
    setattr(self, "raw1D", h1_raw)

    for ctype in ['P', 'M', 'PM', 'T']:
      h3_raw = ROOT.TH3D(f"raw_{ctype}", f"raw_{ctype}", self.weight_nbins, self.weight_bins, self.RL_nbins, self.RL_bins, self.pT_det_nbins, self.pT_det_bins)
      setattr(self, f"raw_{ctype}", h3_raw)

      h2_raw_eec = ROOT.TH2D(f"raw_eec_{ctype}", f"raw_eec_{ctype}", self.RL_nbins, self.RL_bins, self.pT_det_nbins, self.pT_det_bins)
      setattr(self, f"raw_eec_{ctype}", h2_raw_eec)

  #---------------------------------------------------------------
  # Calculate pair distance of two fastjet particles
  #---------------------------------------------------------------
  def calculate_distance(self, p0, p1):
    dphiabs = math.fabs(p0.phi() - p1.phi())
    dphi = dphiabs

    if dphiabs > math.pi:
      dphi = 2*math.pi - dphiabs

    deta = p0.eta() - p1.eta()
    return math.sqrt(deta*deta + dphi*dphi)

  #---------------------------------------------------------------
  # Analyze jets of a given event.
  #---------------------------------------------------------------
  def analyze_jets(self, jets_selected, jetR, R_max = None, rho_bge = 0):
    for jet in jets_selected:
      jet_pt_subtracted = jet.pt() - rho_bge*jet.area()
      if jet_pt_subtracted <= self.jetpt_min_det_subtracted:
        logger.warning("Did not pass 5 GeV cut")
        continue

      self.fill_jet_tables(jet, jet_pt_subtracted)

  def fill_jet_tables(self, jet, jet_pt_sub):
    getattr(self, "raw1D").Fill(jet_pt_sub)

    constituents = fj.sorted_by_pt(jet.constituents())
    c_select = fj.vectorPJ()

    for c in constituents:
      if c.pt() < self.thrd:
        break
      c.set_user_index(0)
      c_select.append(c)
    if self.do_perpendicular_cone:
      c_select2 = fj.vectorPJ()
      for c in constituents:
        if c.pt() < self.thrd:
          break
        c.set_user_index(0)
        c_select2.append(c)

      for part in jet.python_info().perpcone1:
        if part.pt() > self.thrd:
          c_select.append(part)
      for part in jet.python_info().perpcone2:
        if part.pt() > self.thrd:
          c_select2.append(part)

    ipoint = 2
    correl = ecorrel.CorrelatorBuilder(c_select, jet_pt_sub, ipoint, 1, -9999, -9999)
    corr = correl.correlator(ipoint)
    for indices, RL, weight in zip(corr.indices(), corr.rs(), corr.weights()):
      # print(indices, RL, weight, jet_pt_sub)
      uidx1, uidx2 = [c_select[idx].user_index() for idx in indices]
      ptype_f = self.get_pair_type_factor(uidx1, uidx2)
      q1, q2 = [c_select[idx].python_info().charge for idx in indices]
      ctype = self.get_charge_type(q1, q2)

      getattr(self,  "raw_T").Fill(weight, RL, jet_pt_sub, ptype_f)
      getattr(self,  "raw_eec_T").Fill(RL, jet_pt_sub, weight * ptype_f)
      getattr(self, f"raw_{ctype}").Fill(weight, RL, jet_pt_sub, ptype_f)
      getattr(self, f"raw_eec_{ctype}").Fill(RL, jet_pt_sub, weight * ptype_f)

    if self.do_perpendicular_cone:
      correl2 = ecorrel.CorrelatorBuilder(c_select2, jet_pt_sub, ipoint, 1, -9999, -9999)
      corr2 = correl2.correlator(ipoint)
      for indices, RL, weight in zip(corr2.indices(), corr2.rs(), corr2.weights()):
        uidx1, uidx2 = [c_select2[idx].user_index() for idx in indices]
        ptype_f = self.get_pair_type_factor(uidx1, uidx2)
        if ptype_f == 1:
          continue
        q1, q2 = [c_select2[idx].python_info().charge for idx in indices]
        ctype = self.get_charge_type(q1, q2)

        getattr(self,  "raw_T").Fill(weight, RL, jet_pt_sub, ptype_f)
        getattr(self,  "raw_eec_T").Fill(RL, jet_pt_sub, weight * ptype_f)
        getattr(self, f"raw_{ctype}").Fill(weight, RL, jet_pt_sub, ptype_f)
        getattr(self, f"raw_eec_{ctype}").Fill(RL, jet_pt_sub, weight * ptype_f)
    if self.mixed_cone:
      cone1_w_ptcut = [p for p in jet.python_info().perpcone1 if p.pt() > self.thrd]
      cone2_w_ptcut = [p for p in jet.python_info().perpcone2 if p.pt() > self.thrd]
      for p1 in cone1_w_ptcut:
        for p2 in cone2_w_ptcut:
          pair_weight = p1.pt() * p2.pt() / (jet_pt_sub ** 2)
          pair_RL = np.sqrt((p1.eta() - p2.eta()) ** 2 + (p1.delta_phi_to(p2)) ** 2)
          q1 = p1.python_info().charge
          q2 = p2.python_info().charge
          ctype = self.get_charge_type(q1, q2)
          # pair type factor is +1 for mixed cone
          getattr(self,  "raw_T").Fill(pair_weight, pair_RL, jet_pt_sub)
          getattr(self,  "raw_eec_T").Fill(pair_RL, jet_pt_sub, pair_weight)
          getattr(self, f"raw_{ctype}").Fill(pair_weight, pair_RL, jet_pt_sub)
          getattr(self, f"raw_eec_{ctype}").Fill(pair_RL, jet_pt_sub, pair_weight)
          # we fill twice to get the reverse pair order as well
          getattr(self,  "raw_T").Fill(pair_weight, pair_RL, jet_pt_sub)
          getattr(self,  "raw_eec_T").Fill(pair_RL, jet_pt_sub, pair_weight)
          getattr(self, f"raw_{ctype}").Fill(pair_weight, pair_RL, jet_pt_sub)
          getattr(self, f"raw_eec_{ctype}").Fill(pair_RL, jet_pt_sub, pair_weight)

  def get_charge_type(self, q1, q2):
    if q1 > 0 and q2 > 0:
      return 'P'
    elif q1 * q2 == -1:
      return 'PM'
    else:
      return 'M'

  # def get_pair_type(self, uidx1, uidx2):
  #   # jet parts have uidx 0, cone1 (+angle) has uidx +1, cone2 (-angle) has uidx -1
  #   if uidx1 == 0 and uidx2 == 0:
  #     return 'jj'
  #   elif uidx1 * uidx2 == 0: # take advantage of lazy if
  #     return 'jp'
  #   else:
  #     return 'pp'
  def get_pair_type_factor(self, uidx1, uidx2):
    # jet parts have uidx 0, cone1 (+angle) has uidx +1, cone2 (-angle) has uidx -1
    if uidx1 == 0 and uidx2 == 0: # jet-jet
      return 1
    else: # jet-perp and perp-perp
      return -1
    # mixed cone not checked here, but would be +1 factor

##################################################################
if __name__ == '__main__':
  # Define arguments
  parser = argparse.ArgumentParser(description='Process data')
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
  
  handler = logging.StreamHandler()

  # Create a formatter and set it for the handler
  try:
    os.environ['SLURM_SUBMIT_DIR']
    handler.setFormatter(logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(funcName)s - %(message)s'))
  except KeyError:
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(funcName)s - %(message)s'))

  handler.setLevel(logging.INFO)
  logger.addHandler(handler)
  logger.setLevel(logging.INFO)

  logger.info('Configuring...')
  logger.info(f"Input file: \'{args.input_file}\'")
  logger.info(f"Config file: \'{args.config_file}\'")
  logger.info(f"Ouput directory: '{args.output_dir}'")

  # If invalid inputFile is given, exit
  if not os.path.exists(args.input_file):
    logger.critical(f'File {args.input_file} does not exist! Exiting!')
    sys.exit(1)

  # If invalid configFile is given, exit
  if not os.path.exists(args.config_file):
    logger.critical(f'File {args.config_file} does not exist! Exiting!')
    sys.exit(1)

  analysis = ProcessData_ENC(input_file=args.input_file, config_file=args.config_file, output_dir=args.output_dir)
  analysis.process_data()