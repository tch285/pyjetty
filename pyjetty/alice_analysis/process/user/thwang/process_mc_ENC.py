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
import array
import math
# from array import *

# Fastjet via python (from external library heppy)
import fastjet as fj
# import fjcontrib
# import fjtools
import ecorrel
import uproot as ur

# Analysis utilities
# from pyjetty.alice_analysis.process.base import process_io
# from pyjetty.alice_analysis.process.base import process_io_emb
# from pyjetty.alice_analysis.process.base import jet_info
from pyjetty.alice_analysis.process.user.substructure import process_mc_base
# from pyjetty.alice_analysis.process.base import thermal_generator
# from pyjetty.mputils.csubtractor import CEventSubtractor

import logging
logger = logging.getLogger(__name__)

def linbins(xmin, xmax, nbins):
	lspace = np.linspace(xmin, xmax, nbins+1)
	arr = array.array('f', lspace)
	return arr

def logbins(xmin, xmax, nbins):
	lspace = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
	arr = array.array('f', lspace)
	return arr

ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()

################################################################
class ProcessMC_ENC(process_mc_base.ProcessMCBase):

	#---------------------------------------------------------------
	# Constructor
	#---------------------------------------------------------------
	def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):

		# Initialize base class
		super(ProcessMC_ENC, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)

		self.observable = self.observable_list[0]

		with open(self.config_file, 'r') as stream:
			config = yaml.safe_load(stream)

		self.pT_min, self.pT_max, self.pT_nbins = config["pT_binning"]
		self.RL_min, self.RL_max, self.RL_nbins = config["RL_binning"]
		self.pTRL_min, self.pTRL_max, self.pTRL_nbins = config["pTRL_binning"]
		self.pT_bins = linbins(self.pT_min,self.pT_max,self.pT_nbins)
		self.RL_bins = logbins(self.RL_min,self.RL_max,self.RL_nbins)
		self.pTRL_bins = logbins(self.pTRL_min,self.pTRL_max,self.pTRL_nbins)
		if "kT_binning" in config.keys():
			self.kT_min, self.kT_max, self.kT_nbins = config["kT_binning"]
			self.kT_bins = linbins(self.kT_min,self.kT_max,self.kT_nbins)
		if "pairdist_binning" in config.keys():
			self.pairdist_min, self.pairdist_max, self.pairdist_nbins = config["pairdist_binning"]
			self.pairdist_bins = linbins(self.pairdist_min,self.pairdist_max,self.pairdist_nbins)
		if "trk_pt_binning" in config.keys():
			self.trk_pt_min, self.trk_pt_max, self.trk_pt_nbins = config["trk_pt_binning"]
			self.trk_pt_bins = logbins(self.trk_pt_min,self.trk_pt_max,self.trk_pt_nbins)
		if "pt_rsn_binning" in config.keys():
			self.rsn_min, self.rsn_max, self.rsn_nbins = config["pt_rsn_binning"]
			self.rsn_bins = linbins(self.rsn_min,self.rsn_max,self.rsn_nbins)
		if "dp_edges" in config.keys():
			self.dp_bins = array.array('f', config['dp_edges'])
			self.dp_nbins = len(self.dp_bins) - 1
		if "logRL_binning" in config.keys():
			self.logRL_min, self.logRL_max, self.logRL_nbins = config['logRL_binning']
			self.logRL_bins = linbins(self.logRL_min, self.logRL_max, self.logRL_nbins) # x edges
		if 'pair_eff_on' in config.keys() and self.ENC_fastsim:
			self.pair_eff_on = config['pair_eff_on']
		else:
			self.pair_eff_on = False

		if self.ENC_fastsim and self.pair_eff_on:
			self.pair_effs_qpt, self.eff_qpt_edges, self.eff_logRL_edges = self.get_effs_qpt_from_file(config['pair_eff_file_qpt'])
			self.pair_effs_phet, self.eff_kt_edges, self.eff_phist_edges, self.eff_deta_edges = self.get_effs_phet_from_file(config['pair_eff_file_phet'])
			
	def get_effs_phet_from_file(self, filename):
		with ur.open(filename) as file:
			h = file['pair_eff_T_kt_ph_et']
			ktedges = h.axis(0).edges()
			phedges = h.axis(1).edges()
			etedges = h.axis(2).edges()
			effs = {}
			for ptype in ["P", "M", "PM"]:
				effs[ptype] = file[f'pair_eff_{ptype}_kt_ph_et'].values()
		return effs, ktedges, phedges, etedges
	def get_effs_qpt_from_file(self, filename):
		with ur.open(filename) as file:
			h = file['pair_eff_T']
			logRLedges = h.axis(0).edges()
			qpTedges = h.axis(1).edges()
			effs = {}
			for ptype in ["P", "M", "PM"]:
				effs[ptype] = file[f'pair_eff_{ptype}'].values()
		return effs, qpTedges, logRLedges

	def get_pair_eff_qpt(self, RL, dp, ptype):
		eff = self.pair_effs_qpt[ptype]

		x_bin = np.searchsorted(self.eff_logRL_edges, np.log10(RL), side='left') - 1
		y_bin = np.searchsorted(self.eff_qpt_edges, dp, side='left') - 1
		
		# Check if point is within bounds
		if (0 <= x_bin < eff.shape[0] and 
			0 <= y_bin < eff.shape[1]):
			return eff[x_bin, y_bin]
		else:
			return 1
	def get_pair_eff_phet(self, kt, dphistar, deta, ptype):
		eff = self.pair_effs_phet[ptype]

		x_bin = np.searchsorted(self.eff_kt_edges, kt, side='left') - 1
		y_bin = np.searchsorted(self.eff_phist_edges, dphistar, side='left') - 1
		z_bin = np.searchsorted(self.eff_deta_edges, deta, side='left') - 1
		
		# Check if point is within bounds
		if (0 <= x_bin < eff.shape[0] and 
			0 <= y_bin < eff.shape[1] and
			0 <= z_bin < eff.shape[2]):
			return eff[x_bin, y_bin, z_bin]
		else:
			return 1

	#---------------------------------------------------------------
	# Calculate pair distance of two fastjet particles
	#---------------------------------------------------------------
	def calculate_distance(self, p0, p1):
		dphiabs = np.fabs(p0.phi() - p1.phi())
		dphi = dphiabs

		if dphiabs > np.pi:
			dphi = 2*np.pi - dphiabs

		deta = p0.eta() - p1.eta()
		return np.sqrt(deta*deta + dphi*dphi)

	#---------------------------------------------------------------
	# Calculate phistar distance of two fastjet particles
	#---------------------------------------------------------------
	# def calc_phistar(self, p1, p2, q1, q2):
	# 	R = 1.1 # reference radius for TPC
	# 	Bz = 0.5
	# 	phi12 = p1.delta_phi_to(p2) # this calculates p2.phi() - p1.phi()
	# 	pt1 = p1.pt()
	# 	pt2 = p2.pt()
	# 	return phi12 + q1*np.arcsin(0.015*Bz*R/pt1) - q2*np.arcsin(0.015*Bz*R/pt2)
	def calc_phistar(self, p1, p2, q1, q2):
		R = 1.1 # reference radius for TPC
		Bz = -0.5 # extra minus
		dalpha = q1*np.arcsin(-0.15*Bz*R/p1.pt()) - q2*np.arcsin(-0.15*Bz*R/p2.pt())

		return self.calculate_dphi(p1.phi(), p2.phi()) + dalpha

	def calculate_dphi(self, phi1, phi2):
		delta_phi = self.Phi_mpi_pi(phi1-phi2)
		# if (delta_phi<-0.5*M_PI) delta_phi += 2*M_PI; // This should not be needed

		if (delta_phi>np.pi or delta_phi<-np.pi):
			self.warning("Delta phi not inside desired range")

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
		for observable in self.observable_list:
			for trk_thrd in self.obs_settings[observable]:
				obs_label = self.utils.obs_label(trk_thrd, None)

				if self.ENC_fastsim and self.pair_eff_on:
					self.pair_type_labels = ['_qpt', '_phet']
				else:
					self.pair_type_labels = ['']
				if self.do_rho_subtraction or self.do_constituent_subtraction:
					self.pair_type_labels = ['_bb','_sb','_ss']

				# Init ENC histograms (both det and truth level)
				for pair_type_label in self.pair_type_labels:
					if 'E2C' in observable or 'E3C' in observable:
						name = 'h_{}{}_JetPt_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
						h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
						h.GetXaxis().SetTitle('p_{T,ch jet}')
						h.GetYaxis().SetTitle('R_{L}')
						setattr(self, name, h)

						# name = 'h_{}{}Pt_JetPt_R{}_{}'.format(observable, pair_type_label, jetR, obs_label) # pt scaled histograms (currently only for unmatched jets)
						# h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.pTRL_nbins, self.pTRL_bins)
						# h.GetXaxis().SetTitle('p_{T,ch jet}')
						# h.GetYaxis().SetTitle('p_{T,ch jet}R_{L}') # NB: y axis scaled by jet pt (applied jet by jet)
						# setattr(self, name, h)

						# Truth histograms
						name = 'h_{}{}_JetPt_Truth_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
						h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
						h.GetXaxis().SetTitle('p_{T,ch jet}^{truth}')
						h.GetYaxis().SetTitle('R_{L}')
						setattr(self, name, h)

						# name = 'h_{}{}Pt_JetPt_Truth_R{}_{}'.format(observable, pair_type_label, jetR, obs_label) # pt scaled histograms (currently only for unmatched jets)
						# h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.pTRL_nbins, self.pTRL_bins)
						# h.GetXaxis().SetTitle('p_{T,ch jet}')
						# h.GetYaxis().SetTitle('p_{T,ch jet}R_{L}') # NB: y axis scaled by jet pt (applied jet by jet)
						# setattr(self, name, h)

						# Matched det histograms
						# name = 'h_matched_{}{}_JetPt_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
						# h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
						# h.GetXaxis().SetTitle('p_{T,ch jet}')
						# h.GetYaxis().SetTitle('R_{L}')
						# setattr(self, name, h)

						# Matched det histograms (with matched truth jet pT filled to the other axis)
						# name = 'h_matched_extra_{}{}_JetPt_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
						# h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
						# h.GetXaxis().SetTitle('p_{T,ch jet}^{truth}')
						# h.GetYaxis().SetTitle('R_{L}')
						# setattr(self, name, h)

						# Matched truth histograms
						# truth pairs with truth jet pt weight and selection
						# name = 'h_matched_{}{}_JetPt_Truth_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
						# h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
						# h.GetXaxis().SetTitle('p_{T,ch jet}^{truth}')
						# h.GetYaxis().SetTitle('R_{L}')
						# setattr(self, name, h)

						# # det pairs with truth jet pt weight and truth jet pt selection
						# name = 'h_matched_{}{}_JetPt_TruthJetPtWeightSel_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
						# h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
						# h.GetXaxis().SetTitle('p_{T,ch jet}^{truth}')
						# h.GetYaxis().SetTitle('R_{L}')
						# setattr(self, name, h)

						# # det pairs with truth jet pt weight and det jet pt selection
						# name = 'h_matched_{}{}_JetPt_TruthJetPtWeight_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
						# h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
						# h.GetXaxis().SetTitle('p_{T,ch jet}')
						# h.GetYaxis().SetTitle('R_{L}')
						# setattr(self, name, h)

						# # det pairs with det jet pt weight and truth jet pt selection
						# name = 'h_matched_{}{}_JetPt_TruthJetPtSel_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
						# h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
						# h.GetXaxis().SetTitle('p_{T,ch jet}^{truth}')
						# h.GetYaxis().SetTitle('R_{L}')
						# setattr(self, name, h)

						# # truth pairs with det jet pt weight and det jet pt selection
						# name = 'h_matched_{}{}_JetPt_TruthPairs_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
						# h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
						# h.GetXaxis().SetTitle('p_{T,ch jet}')
						# h.GetYaxis().SetTitle('R_{L}')
						# setattr(self, name, h)

						# if self.do_jetcone:
						# 	for jetcone_R in self.jetcone_R_list:
						# 		# Matched det histograms
						# 		name = 'h_jetcone{}_matched_{}{}{}_JetPt_R{}_{}'.format(jetcone_R, observable, ipoint, pair_type_label, jetR, obs_label)
						# 		h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
						# 		h.GetXaxis().SetTitle('p_{T,ch jet}')
						# 		h.GetYaxis().SetTitle('R_{L}')
						# 		setattr(self, name, h)

						# 		# Matched det histograms (with matched truth jet pT filled to the other axis)
						# 		name = 'h_jetcone{}_matched_extra_{}{}{}_JetPt_R{}_{}'.format(jetcone_R, observable, ipoint, pair_type_label, jetR, obs_label)
						# 		h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
						# 		h.GetXaxis().SetTitle('p_{T,ch jet}^{truth}')
						# 		h.GetYaxis().SetTitle('R_{L}')
						# 		setattr(self, name, h)

						# 		# Matched truth histograms
						# 		name = 'h_jetcone{}_matched_{}{}{}_JetPt_Truth_R{}_{}'.format(jetcone_R, observable, ipoint, pair_type_label, jetR, obs_label)
						# 		h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
						# 		h.GetXaxis().SetTitle('p_{T,ch jet}')
						# 		h.GetYaxis().SetTitle('R_{L}')
						# 		setattr(self, name, h)
						# if self.thermal_model:
						# 		for R_max in self.max_distance:
						# 			name = 'h_{}{}{}_JetPt_R{}_{}_Rmax{}'.format(observable, ipoint, pair_type_label, jetR, obs_label, R_max)
						# 			h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
						# 			h.GetXaxis().SetTitle('p_{T,ch jet}')
						# 			h.GetYaxis().SetTitle('R_{L}')
						# 			setattr(self, name, h)
					if 'jet_pairdist' in observable:
						# for mpref in ['', '_matched']:
						for mpref in ['_matched']:
							for data_class in ['', "Truth_"]:
								for xaxis, (x_nbins, x_bins), xtitle in zip(['JetPt', 'PairKt'], [(self.pT_nbins, self.pT_bins), (self.kT_nbins, self.kT_bins)], ['p_{T,ch jet}', 'pair k_{T}']):
									for yaxis in ['phi', 'phistar', 'eta']:
										name = 'h{}_{}_{}_{}_{}R{}_{}'.format(mpref, observable, yaxis, xaxis, data_class, jetR, trk_thrd)
										h = ROOT.TH2D(name, name, x_nbins, x_bins, self.pairdist_nbins, self.pairdist_bins)
										h.GetXaxis().SetTitle(xtitle)
										h.GetYaxis().SetTitle(f'{yaxis}')
										setattr(self, name, h)
									name = 'h{}_{}_{}_{}_{}R{}_{}'.format(mpref, observable, 'RL', xaxis, data_class, jetR, trk_thrd)
									h = ROOT.TH2D(name, name, x_nbins, x_bins, self.RL_nbins, self.RL_bins)
									h.GetXaxis().SetTitle(xtitle)
									h.GetYaxis().SetTitle('RL')
									setattr(self, name, h)
									name = 'h{}_{}_{}_{}_{}R{}_{}'.format(mpref, observable, 'phistar_eta', xaxis, data_class, jetR, trk_thrd)
									h = ROOT.TH3F(name, name, x_nbins, x_bins, self.pairdist_nbins, self.pairdist_bins, self.pairdist_nbins, self.pairdist_bins)
									h.GetXaxis().SetTitle(xtitle)
									h.GetYaxis().SetTitle('phistar')
									h.GetZaxis().SetTitle('eta')
									setattr(self, name, h)

					if 'pair_eff' in observable:
						for dtype in ['_2miss', '_1miss', '_0miss', '_Truth']:
							name = f'h_{observable}{dtype}'
							h = ROOT.TH2D(name, name, self.logRL_nbins, self.logRL_bins, self.dp_nbins, self.dp_bins)
							h.GetXaxis().SetTitle('log(R_{L})')
							h.GetYaxis().SetTitle('#delta q/p_{T}')
							setattr(self, name, h)

					if 'ENC' in observable:
						for ipoint in range(2, 3):
							name = 'h_{}{}{}_JetPt_R{}_{}'.format(observable, ipoint, pair_type_label, jetR, obs_label)
							h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
							h.GetXaxis().SetTitle('p_{T,ch jet}')
							h.GetYaxis().SetTitle('R_{L}')
							setattr(self, name, h)

							name = 'h_{}{}{}Pt_JetPt_R{}_{}'.format(observable, ipoint, pair_type_label, jetR, obs_label) # pt scaled histograms (currently only for unmatched jets)
							h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.pTRL_nbins, self.pTRL_bins)
							h.GetXaxis().SetTitle('p_{T,ch jet}')
							h.GetYaxis().SetTitle('p_{T,ch jet}R_{L}') # NB: y axis scaled by jet pt (applied jet by jet)
							setattr(self, name, h)

							# Truth histograms
							name = 'h_{}{}{}_JetPt_Truth_R{}_{}'.format(observable, ipoint, pair_type_label, jetR, obs_label)
							h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
							h.GetXaxis().SetTitle('p_{T,ch jet}')
							h.GetYaxis().SetTitle('R_{L}')
							setattr(self, name, h)

							name = 'h_{}{}{}Pt_JetPt_Truth_R{}_{}'.format(observable, ipoint, pair_type_label, jetR, obs_label) # pt scaled histograms (currently only for unmatched jets)
							h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.pTRL_nbins, self.pTRL_bins)
							h.GetXaxis().SetTitle('p_{T,ch jet}')
							h.GetYaxis().SetTitle('p_{T,ch jet}R_{L}') # NB: y axis scaled by jet pt (applied jet by jet)
							setattr(self, name, h)

							# Matched det histograms
							name = 'h_matched_{}{}{}_JetPt_R{}_{}'.format(observable, ipoint, pair_type_label, jetR, obs_label)
							h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
							h.GetXaxis().SetTitle('p_{T,ch jet}')
							h.GetYaxis().SetTitle('R_{L}')
							setattr(self, name, h)

							# Matched det histograms (with matched truth jet pT filled to the other axis)
							name = 'h_matched_extra_{}{}{}_JetPt_R{}_{}'.format(observable, ipoint, pair_type_label, jetR, obs_label)
							h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
							h.GetXaxis().SetTitle('p_{T,ch jet}^{truth}')
							h.GetYaxis().SetTitle('R_{L}')
							setattr(self, name, h)

							# Matched truth histograms
							name = 'h_matched_{}{}{}_JetPt_Truth_R{}_{}'.format(observable, ipoint, pair_type_label, jetR, obs_label)
							h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
							h.GetXaxis().SetTitle('p_{T,ch jet}')
							h.GetYaxis().SetTitle('R_{L}')
							setattr(self, name, h)

							if self.do_jetcone:
								for jetcone_R in self.jetcone_R_list:
									# Matched det histograms
									name = 'h_jetcone{}_matched_{}{}{}_JetPt_R{}_{}'.format(jetcone_R, observable, ipoint, pair_type_label, jetR, obs_label)
									h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
									h.GetXaxis().SetTitle('p_{T,ch jet}')
									h.GetYaxis().SetTitle('R_{L}')
									setattr(self, name, h)

									# Matched det histograms (with matched truth jet pT filled to the other axis)
									name = 'h_jetcone{}_matched_extra_{}{}{}_JetPt_R{}_{}'.format(jetcone_R, observable, ipoint, pair_type_label, jetR, obs_label)
									h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
									h.GetXaxis().SetTitle('p_{T,ch jet}^{truth}')
									h.GetYaxis().SetTitle('R_{L}')
									setattr(self, name, h)

									# Matched truth histograms
									name = 'h_jetcone{}_matched_{}{}{}_JetPt_Truth_R{}_{}'.format(jetcone_R, observable, ipoint, pair_type_label, jetR, obs_label)
									h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
									h.GetXaxis().SetTitle('p_{T,ch jet}')
									h.GetYaxis().SetTitle('R_{L}')
									setattr(self, name, h)

							if self.thermal_model:
								for R_max in self.max_distance:
									name = 'h_{}{}{}_JetPt_R{}_{}_Rmax{}'.format(observable, ipoint, pair_type_label, jetR, obs_label, R_max)
									h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
									h.GetXaxis().SetTitle('p_{T,ch jet}')
									h.GetYaxis().SetTitle('R_{L}')
									setattr(self, name, h)
					if 'EEC_noweight' in observable or 'EEC_weight2' in observable:
							name = 'h_{}{}_JetPt_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
							h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
							h.GetXaxis().SetTitle('p_{T,ch jet}')
							h.GetYaxis().SetTitle('R_{L}')
							setattr(self, name, h)

							# Truth histograms
							name = 'h_{}{}_JetPt_Truth_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
							h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
							h.GetXaxis().SetTitle('p_{T,ch jet}')
							h.GetYaxis().SetTitle('R_{L}')
							setattr(self, name, h)

							# Matched det histograms
							name = 'h_matched_{}{}_JetPt_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
							h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
							h.GetXaxis().SetTitle('p_{T,ch jet}')
							h.GetYaxis().SetTitle('R_{L}')
							setattr(self, name, h)

							# Matched det histograms (with matched truth jet pT filled to the other axis)
							name = 'h_matched_extra_{}{}_JetPt_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
							h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
							h.GetXaxis().SetTitle('p_{T,ch jet}^{truth}')
							h.GetYaxis().SetTitle('R_{L}')
							setattr(self, name, h)

							# Matched truth histograms
							name = 'h_matched_{}{}_JetPt_Truth_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
							h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
							h.GetXaxis().SetTitle('p_{T,ch jet}')
							h.GetYaxis().SetTitle('R_{L}')
							setattr(self, name, h)

							if self.do_jetcone:
								for jetcone_R in self.jetcone_R_list:
									# Matched det histograms
									name = 'h_jetcone{}_matched_{}{}_JetPt_R{}_{}'.format(jetcone_R, observable, pair_type_label, jetR, obs_label)
									h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
									h.GetXaxis().SetTitle('p_{T,ch jet}')
									h.GetYaxis().SetTitle('R_{L}')
									setattr(self, name, h)

									# Matched det histograms (with matched truth jet pT filled to the other axis)
									name = 'h_jetcone{}_matched_extra_{}{}_JetPt_R{}_{}'.format(jetcone_R, observable, pair_type_label, jetR, obs_label)
									h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
									h.GetXaxis().SetTitle('p_{T,ch jet}^{truth}')
									h.GetYaxis().SetTitle('R_{L}')
									setattr(self, name, h)

									# Matched truth histograms
									name = 'h_jetcone{}_matched_{}{}_JetPt_Truth_R{}_{}'.format(jetcone_R, observable, pair_type_label, jetR, obs_label)
									h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
									h.GetXaxis().SetTitle('p_{T,ch jet}')
									h.GetYaxis().SetTitle('R_{L}')
									setattr(self, name, h)

							if self.thermal_model:
								for R_max in self.max_distance:
									name = 'h_{}{}_JetPt_R{}_{}_Rmax{}'.format(observable, pair_type_label, jetR, obs_label, R_max)
									h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.RL_nbins, self.RL_bins)
									h.GetXaxis().SetTitle('p_{T,ch jet}')
									h.GetYaxis().SetTitle('R_{L}')
									setattr(self, name, h)
					if 'track_creco' in observable:
						name = f'h_{observable}'
						h = ROOT.TH1D(name, name, self.trk_pt_nbins, self.trk_pt_bins)
						h.GetXaxis().SetTitle('p_{T}')
						setattr(self, name, h)
						name = f'h_{observable}_Truth'
						h = ROOT.TH1D(name, name, self.trk_pt_nbins, self.trk_pt_bins)
						h.GetXaxis().SetTitle('p_{T}')
						setattr(self, name, h)
					if 'jet_pair_creco' in observable:
						name = f'h_matched_{observable}'
						h = ROOT.TH2D(name, name, self.kT_nbins, self.kT_bins, self.pT_nbins, self.pT_bins)
						h.GetXaxis().SetTitle('pair k_{T}')
						h.GetYaxis().SetTitle('jet p_{T}')
						setattr(self, name, h)
						name = f'h_matched_{observable}_Truth'
						h = ROOT.TH2D(name, name, self.kT_nbins, self.kT_bins, self.pT_nbins, self.pT_bins)
						h.GetXaxis().SetTitle('pair k_{T}')
						h.GetYaxis().SetTitle('jet p_{T}')
						setattr(self, name, h)
				if 'jet_pt' in observable:
					name = 'h_{}_JetPt_R{}_{}'.format(observable, jetR, obs_label)
					h = ROOT.TH1D(name, name, self.pT_nbins, self.pT_bins)
					h.GetXaxis().SetTitle('p_{T,ch jet}')
					h.GetYaxis().SetTitle('Counts')
					setattr(self, name, h)

					name = 'h_{}_JetPt_Truth_R{}_{}'.format(observable, jetR, obs_label)
					h = ROOT.TH1D(name, name, self.pT_nbins, self.pT_bins)
					h.GetXaxis().SetTitle('p_{T,ch jet}')
					h.GetYaxis().SetTitle('Counts')
					setattr(self, name, h)

					# Matched det histograms
					# name = 'h_matched_{}_JetPt_R{}_{}'.format(observable, jetR, obs_label)
					# h = ROOT.TH1D(name, name, self.pT_nbins, self.pT_bins)
					# h.GetXaxis().SetTitle('p_{T,ch jet}')
					# h.GetYaxis().SetTitle('Counts')
					# setattr(self, name, h)

					# Matched truth histograms
					# name = 'h_matched_{}_JetPt_Truth_R{}_{}'.format(observable, jetR, obs_label)
					# h = ROOT.TH1D(name, name, self.pT_nbins, self.pT_bins)
					# h.GetXaxis().SetTitle('p_{T,ch jet}')
					# h.GetYaxis().SetTitle('Counts')
					# setattr(self, name, h)

					# Correlation between matched det and truth
					# name = 'h_matched_{}_JetPt_Truth_vs_Det_R{}_{}'.format(observable, jetR, obs_label)
					# h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.pT_nbins, self.pT_bins)
					# h.GetXaxis().SetTitle('p_{T,ch jet}^{det}')
					# h.GetYaxis().SetTitle('p_{T,ch jet}^{truth}')
					# setattr(self, name, h)
				# # Diagnostic
				# if 'jet_diag' in observable:
				#   name = 'h_{}_JetEta_R{}_{}'.format(observable, jetR, obs_label)
				#
				#   eta_bins = linbins(-10,10,200)
				#   h = ROOT.TH2D(name, name, 200, pt_bins, 200, eta_bins)
				#   h.GetXaxis().SetTitle('p_{T,ch jet}')
				#   h.GetYaxis().SetTitle('#eta_{ch jet}')
				#   setattr(self, name, h)

				#   name = 'h_{}_JetEta_Truth_R{}_{}'.format(observable, jetR, obs_label)
				#
				#   eta_bins = linbins(-10,10,200)
				#   h = ROOT.TH2D(name, name, 200, pt_bins, 200, eta_bins)
				#   h.GetXaxis().SetTitle('p_{T,ch jet}')
				#   h.GetYaxis().SetTitle('#eta_{ch jet}')
				#   setattr(self, name, h)

				# Init pair distance histograms (both det and truth level)
				# average track pt bins
				self.trk_pt_lo = [0, 1, 2, 3, 5, 7, 10]
				self.trk_pt_hi = [1, 2, 3, 5, 7, 10, 100]
				# track pt asymmetry bins: (pt_trk1-pt_trk2)/(pt_trk1+pt_trk2)
				self.trk_alpha_lo = [0, 0.2, 0.4, 0.6, 0.8]
				self.trk_alpha_hi = [0.2, 0.4, 0.6, 0.8, 1]
				# if 'EEC_detail' in observable:
				# 	# inclusive
				# 	name = 'h_{}_JetPt_R{}_{}'.format(observable, jetR, obs_label)
				# 	pt_bins = linbins(0,200,200)
				# 	RL_bins = logbins(1E-4,1,50)
				# 	h = ROOT.TH2D(name, name, 50, pt_bins, 50, RL_bins)
				# 	h.GetXaxis().SetTitle('p_{T,ch jet}')
				# 	h.GetYaxis().SetTitle('R_{L}')
				# 	setattr(self, name, h)

				# 	name = 'h_{}_JetPt_Truth_R{}_{}'.format(observable, jetR, obs_label)
				# 	pt_bins = linbins(0,200,200)
				# 	RL_bins = logbins(1E-4,1,50)
				# 	h = ROOT.TH2D(name, name, 50, pt_bins, 50, RL_bins)
				# 	h.GetXaxis().SetTitle('p_{T,ch jet}')
				# 	h.GetYaxis().SetTitle('R_{L}')
				# 	setattr(self, name, h)

				# 	# fine bins
				# 	for ipt in range( len(self.trk_pt_lo) ):
				# 		for ialpha in range( len(self.trk_alpha_lo) ):
				# 			name = 'h_{}{}{}_{:.1f}{:.1f}_JetPt_R{}_{}'.format(observable, self.trk_pt_lo[ipt], self.trk_pt_hi[ipt], self.trk_alpha_lo[ialpha], self.trk_alpha_hi[ialpha], jetR, obs_label)
				# 			pt_bins = linbins(0,200,200)
				# 			RL_bins = logbins(1E-4,1,50)
				# 			h = ROOT.TH2D(name, name, 50, pt_bins, 50, RL_bins)
				# 			h.GetXaxis().SetTitle('p_{T,ch jet}')
				# 			h.GetYaxis().SetTitle('R_{L}')
				# 			setattr(self, name, h)

				# 			name = 'h_{}{}{}_{:.1f}{:.1f}_JetPt_Truth_R{}_{}'.format(observable, self.trk_pt_lo[ipt], self.trk_pt_hi[ipt], self.trk_alpha_lo[ialpha], self.trk_alpha_hi[ialpha], jetR, obs_label)
				# 			pt_bins = linbins(0,200,200)
				# 			RL_bins = logbins(1E-4,1,50)
				# 			h = ROOT.TH2D(name, name, 50, pt_bins, 50, RL_bins)
				# 			h.GetXaxis().SetTitle('p_{T,ch jet}')
				# 			h.GetYaxis().SetTitle('R_{L}')
				# 			setattr(self, name, h)

				# Residuals and responses (currently not filled or used)
				for trk_thrd in self.obs_settings[observable]:
					for ipoint in range(2, 3):
						if not self.is_pp:
							for R_max in self.max_distance:
								self.create_response_histograms(observable, ipoint, jetR, trk_thrd, R_max)
						# else:
						# 	self.create_response_histograms(observable, ipoint, jetR, trk_thrd)

	#---------------------------------------------------------------
	# This function is called once for each jet subconfiguration
	# Fill 2D histogram of (pt, obs)
	#---------------------------------------------------------------
	def create_response_histograms(self, observable, ipoint, jetR, trk_thrd, R_max = None):

		if R_max:
			suffix = '_Rmax{}'.format(R_max)
		else:
			suffix = ''

		# Create THn of response for ENC
		dim = 4
		title = ['p_{T,det}', 'p_{T,truth}', 'R_{L,det}', 'R_{L,truth}']
		nbins = [30, 20, 100, 100]
		min = [0., 0., 0., 0.]
		max = [150., 200., 1., 1.]
		name = 'hResponse_JetPt_{}{}_R{}_{}{}'.format(observable, ipoint, jetR, trk_thrd, suffix)
		self.create_thn(name, title, dim, nbins, min, max)

		name = 'hResidual_JetPt_{}{}_R{}_{}{}'.format(observable, ipoint, jetR, trk_thrd, suffix)
		h = ROOT.TH3F(name, name, 20, 0, 200, 100, 0., 1., 200, -2., 2.)
		h.GetXaxis().SetTitle('p_{T,truth}')
		h.GetYaxis().SetTitle('R_{L}')
		h.GetZaxis().SetTitle('#frac{R_{L,det}-R_{L,truth}}{R_{L,truth}}')
		setattr(self, name, h)

	# def get_pair_eff_weights(self, corr_builder, ipoint, constituents):
	# 	# NB: currently applying the pair eff weight to both 2 point correlator and higher point correlators. Need to check if the same pair efficiency effect still work well for higher point correlators
	# 	weights_pair = []
	# 	for index in range(corr_builder.correlator(ipoint).rs().size()):
	# 		part1 = corr_builder.correlator(ipoint).indices1()[index]
	# 		part2 = corr_builder.correlator(ipoint).indices2()[index]
	# 		if part1!=part2: # FIX ME: not sure, but for now only apply pair efficiency for non auto-correlations
	# 			# Need to find the associated truth information for each pair (charge and momentum)
	# 			part1_truth = constituents[part1].python_info().particle_truth
	# 			part2_truth = constituents[part2].python_info().particle_truth
	# 			q1 = constituents[part1].python_info().charge
	# 			q2 = constituents[part2].python_info().charge
	# 			dist = corr_builder.correlator(ipoint).rs()[index] # NB: use reconstructed distance since it's faster and should be equivalent to true distance because there is no angular smearing on the track momentum. To switch back to the true distance, use: self.calculate_distance(part1_truth, part2_truth)
	# 			dq_over_p = q1/part1_truth.pt()-q2/part2_truth.pt()
	# 			# calculate pair efficeincy and apply it as an additional weight
	# 			weights_pair.append( self.get_pair_eff(dist, dq_over_p) )
	# 		else:
	# 			weights_pair.append( 1 )
	# 	return weights_pair
	def get_pair_eff_weights(self, corr_builder, ipoint, constituents):
		weights_qpt = []
		weights_phet = []
		for indices, RL, weight in zip(corr_builder.correlator(ipoint).indices(), corr_builder.correlator(ipoint).rs(), corr_builder.correlator(ipoint).weights()):
			idx1 = indices[0]
			idx2 = indices[1]
			if indices[0] != indices[1]:
				p1 = constituents[idx1].python_info().particle_truth
				p2 = constituents[idx2].python_info().particle_truth
				q1, q2 = p1.python_info().charge, p2.python_info().charge
				pt1, pt2 = p1.pt(), p2.pt()
				dq_over_p = np.abs(q1 / pt1 - q2 / pt2)
				kt = self.calc_kt(p1, p2)
				dphistar = self.calc_phistar(p1, p2, q1, q2)
				deta = p2.eta() - p1.eta()
				if q1 * q2 < 0:
					ptype = "PM"
				elif q1 > 0 and q2 > 0:
					ptype = "P"
				else:
					ptype = "M"
				weights_qpt.append( self.get_pair_eff_qpt(RL, dq_over_p, ptype) )
				weights_phet.append( self.get_pair_eff_phet(kt, dphistar, deta, ptype) )
			else:
				weights_qpt.append( 1 )
				weights_phet.append( 1 )
		return weights_qpt, weights_phet

	def is_same_charge(self, corr_builder, ipoint, constituents, index):
		part1 = int(corr_builder.correlator(ipoint).indices1()[index])
		part2 = int(corr_builder.correlator(ipoint).indices2()[index])
		q1 = int(constituents[part1].python_info().charge)
		q2 = int(constituents[part2].python_info().charge)

		if q1*q2 > 0:
			return True
		else:
			return False

	def check_pair_type(self, corr_builder, ipoint, constituents, index):
		part1 = int(corr_builder.correlator(ipoint).indices1()[index])
		part2 = int(corr_builder.correlator(ipoint).indices2()[index])
		type1 = constituents[part1].user_index()
		type2 = constituents[part2].user_index()

		# NB: match the strings in self.pair_type_label = ['bb','sb','ss']
		if type1*type2 >= 0:
			if type1 < 0 or type2 < 0:
				# print('bkg-bkg (',type1,type2,') pt1',constituents[part1].perp(),'pt2',constituents[part2].perp())
				return 0 # means bkg-bkg
			else:
				# print('sig-sig (',type1,type2,') pt1',constituents[part1].perp(),'pt2',constituents[part2].perp())
				return 2 # means sig-sig
		else:
			# print('sig-bkg (',type1,type2,') pt1',constituents[part1].perp(),'pt2',constituents[part2].perp())
			return 1 # means sig-bkg

	def fill_efficiency_histograms(self, parts_det, parts_truth):
		obs_list = [obs for obs in self.observable_list if "track_creco" in obs]
		if len(obs_list) != 0:
			reco_truth = [part for part in parts_truth if part.python_info().particle_det is not None]
			for part_truth in reco_truth:
				part_det = part_truth.python_info().particle_det
				ch_truth = part_truth.python_info().charge
				ch_det = part_det.python_info().charge
				if ch_truth > 0:
					label = "P"
				else:
					label = "M"
				getattr(self, f"h_track_creco_{label}_pt_Truth").Fill(part_truth.pt())
				if ch_truth * ch_det > 0:
					getattr(self, f"h_track_creco_{label}_pt").Fill(part_truth.pt())
				else:
					logger.warning("Mischarged particle found!")

		if "track_eff_pt" in self.observable_list:
			for part in parts_truth:
				getattr(self, "h_track_eff_pt_Truth").Fill(part.pt())
				if part.python_info().particle_det is not None:
					getattr(self, "h_track_eff_pt").Fill(part.pt())
				# 	getattr(self, "h_track_eff_pt_teff").Fill(True, part.pt())
				# else:
				# 	getattr(self, "h_track_eff_pt_teff").Fill(False, part.pt())

		if "track_pt_rsn" in self.observable_list:
			for part in parts_truth:
				if part.python_info().particle_det is not None:
					getattr(self, "h_track_pt_rsn").Fill(part.pt(), (part.python_info().particle_det.pt() - part.pt()) / part.pt())

		obs_list = [obs for obs in self.observable_list if "track_pairdist" in obs]
		if len(obs_list) != 0:
			hname = 'h_track_pairdist_{}_{}_PairKt{}'

			index_pairs = itertools.combinations(range(len(parts_truth)), 2)
			for i1, i2 in index_pairs:
				p1 = parts_truth[i1]
				p2 = parts_truth[i2]
				charges = np.array([p1.python_info().charge, p2.python_info().charge])
				# delta_phi = p1.delta_phi_to(p2)
				delta_phistar = self.calc_phistar(p1, p2, p1.python_info().charge, p2.python_info().charge)
				delta_eta = p2.eta() - p1.eta()
				# RL = np.sqrt(delta_phi ** 2 + delta_eta ** 2)
				RL = self.calculate_distance(p1, p2)
				pair_kt = self.calc_kt(p1, p2)
				# pair_kt = (p1.pt() + p2.pt()) / 2

				if np.all(charges > 0):
					pair_type = "P"
				elif np.all(charges < 0):
					pair_type = "M"
				else:
					pair_type = "PM"

				for pair_kind in ["T", pair_type]:
					# getattr(self, hname.format(pair_kind, "phi",         "_Truth")).Fill(pair_kt, delta_phi)
					# getattr(self, hname.format(pair_kind, "phistar",     "_Truth")).Fill(pair_kt, delta_phistar)
					# getattr(self, hname.format(pair_kind, "eta",         "_Truth")).Fill(pair_kt, delta_eta)
					# getattr(self, hname.format(pair_kind, "RL",          "_Truth")).Fill(pair_kt, RL)
					getattr(self, hname.format(pair_kind, "phistar_eta", "_Truth")).Fill(pair_kt, delta_phistar, delta_eta)
				
				if p1.python_info().particle_det is not None and p2.python_info().particle_det is not None:
					for pair_kind in ["T", pair_type]:
						# getattr(self, hname.format(pair_kind, "phi",         "")).Fill(pair_kt, delta_phi)
						# getattr(self, hname.format(pair_kind, "phistar",     "")).Fill(pair_kt, delta_phistar)
						# getattr(self, hname.format(pair_kind, "eta",         "")).Fill(pair_kt, delta_eta)
						# getattr(self, hname.format(pair_kind, "RL",          "")).Fill(pair_kt, RL)
						getattr(self, hname.format(pair_kind, "phistar_eta", "")).Fill(pair_kt, delta_phistar, delta_eta)

		obs_list = [obs for obs in self.observable_list if "pair_eff" in obs]
		if len(obs_list) != 0:
			hname = 'h_pair_eff_{}{}'

			# index_pairs = itertools.combinations(range(len(parts_truth)), 2)
			# for i1, i2 in index_pairs:
			# 	p1 = parts_truth[i1]
			# 	p2 = parts_truth[i2]
			for p1, p2 in itertools.combinations(parts_truth, 2):
				charges = np.array([p1.python_info().charge, p2.python_info().charge])
				RL = self.calculate_distance(p1, p2)
				# logRL = np.log10(np.sqrt((p2.eta() - p1.eta()) ** 2 + (p1.delta_phi_to(p2)) ** 2))
				logRL = np.log10(RL)
				dqpT = np.abs(p2.python_info().charge / p2.pt() - p1.python_info().charge / p1.pt())

				if np.all(charges > 0):
					ptype = "P"
				elif np.all(charges < 0):
					ptype = "M"
				else:
					ptype = "PM"
				
				for pair_type in ["T", ptype]:
					getattr(self, hname.format(pair_type, "_Truth")).Fill(logRL, dqpT)
				
				pdets = [p1.python_info().particle_det, p2.python_info().particle_det]
				if (pdets[0] is None) ^ (pdets[1] is None):
					for pair_type in ["T", ptype]:
						getattr(self, hname.format(pair_type, "_1miss")).Fill(logRL, dqpT)
				elif (pdets[0] is None) & (pdets[1] is None):
					for pair_type in ["T", ptype]:
						getattr(self, hname.format(pair_type, "_2miss")).Fill(logRL, dqpT)
				else:
					for pair_type in ["T", ptype]:
						getattr(self, hname.format(pair_type, "_0miss")).Fill(logRL, dqpT)

	#---------------------------------------------------------------
	# This function is called once for each jet subconfiguration
	# Fill 2D histogram of (pt, obs)
	#---------------------------------------------------------------
	def fill_observable_histograms(self, hname, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_pt_ungroomed):
		# For ENC in PbPb, jet_pt_ungroomed stores the corrected jet pT
		constituents = fj.sorted_by_pt(jet.constituents())
		c_select = fj.vectorPJ()
		trk_thrd = obs_setting

		for c in constituents:
			if c.pt() < trk_thrd:
				break
			c_select.append(c) # NB: use the break statement since constituents are already sorted

		if self.ENC_pair_cut and ('Truth' not in hname):
			dphi_cut = -9999 # means no dphi cut
			deta_cut = 0.008
		else:
			dphi_cut = -9999
			deta_cut = -9999

		# NB: use jet_pt_ungroomed instead of jet.perp() for PbPb, which include the UE subtraction
		if self.do_rho_subtraction:
			jet_pt = jet_pt_ungroomed
		else:
			jet_pt = jet.perp()

		maxpoint = 2
		# print(hname)
		new_corr = ecorrel.CorrelatorBuilder(c_select, jet_pt, maxpoint, 1, dphi_cut, deta_cut)
		for observable in self.observable_list:
			if 'ENC' in observable or 'EEC_noweight' in observable or 'EEC_weight2' in observable:
				for ipoint in range(2, 3):
					if self.ENC_fastsim and ('Truth' not in hname): # NB: only apply pair efficiency effect for fast sim and det level distributions
						weights_pair = self.get_pair_eff_weights(new_corr, ipoint, c_select)

					for index in range(new_corr.correlator(ipoint).rs().size()):
						# processing only like-sign pairs when self.ENC_pair_like is on
						if self.ENC_pair_like and (not self.is_same_charge(new_corr, ipoint, c_select, index)):
							continue

						# processing only unlike-sign pairs when self.ENC_pair_unlike is on
						if self.ENC_pair_unlike and self.is_same_charge(new_corr, ipoint, c_select, index):
							continue

						# separate out sig-sig, sig-bkg, bkg-bkg correlations for EEC pairs
						pair_type_label = ''
						if self.do_rho_subtraction or self.do_constituent_subtraction:
							pair_type = self.check_pair_type(new_corr, ipoint, c_select, index)
							pair_type_label = self.pair_type_labels[pair_type]

						if 'ENC' in observable:
							if self.ENC_fastsim and ('Truth' not in hname):
								getattr(self, hname.format(observable + str(ipoint) + pair_type_label,obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index]*weights_pair[index])
								getattr(self, hname.format(observable + str(ipoint) + pair_type_label + 'Pt',obs_label)).Fill(jet_pt, jet_pt*new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index]*weights_pair[index]) # NB: fill pt*RL

							else:
								getattr(self, hname.format(observable + str(ipoint) + pair_type_label,obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index])
								getattr(self, hname.format(observable + str(ipoint) + pair_type_label + 'Pt',obs_label)).Fill(jet_pt, jet_pt*new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index])

						if ipoint==2 and 'EEC_noweight' in observable:
							if self.ENC_fastsim and ('Truth' not in hname):
								getattr(self, hname.format(observable + pair_type_label,obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index], weights_pair[index])
							else:
								getattr(self, hname.format(observable + pair_type_label,obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index])

						if ipoint==2 and 'EEC_weight2' in observable:
							if self.ENC_fastsim and ('Truth' not in hname):
								getattr(self, hname.format(observable + pair_type_label,obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index], pow(new_corr.correlator(ipoint).weights()[index]*weights_pair[index],2))
							else:
								getattr(self, hname.format(observable + pair_type_label,obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index], pow(new_corr.correlator(ipoint).weights()[index],2))

			if 'jet_pt' in observable:
				getattr(self, hname.format(observable,obs_label)).Fill(jet_pt)

			# NB: for now, only perform this check on data and full sim
			if 'EEC_detail' in observable and not self.ENC_fastsim:
				ipoint = 2 # EEC is 2 point correlator
				for index in range(new_corr.correlator(ipoint).rs().size()):
					part1 = new_corr.correlator(ipoint).indices1()[index]
					part2 = new_corr.correlator(ipoint).indices2()[index]
					pt1 = c_select[part1].perp()
					pt2 = c_select[part2].perp()
					pt_avg = (pt1+pt2)/2
					alpha = math.fabs(pt1-pt2)

					for _, (pt_lo, pt_hi) in enumerate(zip(self.trk_pt_lo,self.trk_pt_hi)):
						for _, (alpha_lo, alpha_hi) in enumerate(zip(self.trk_alpha_lo,self.trk_alpha_hi)):
							if pt_avg >= pt_lo and pt_avg < pt_hi and alpha >= alpha_lo and alpha < alpha_hi:
								if 'noweight' in observable:
									getattr(self, hname.format(observable + str(pt_lo) + str(pt_hi) + '_' + '{:.1f}'.format(alpha_lo) + '{:.1f}'.format(alpha_hi),obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index])
								else:
									getattr(self, hname.format(observable + str(pt_lo) + str(pt_hi) + '_' + '{:.1f}'.format(alpha_lo) + '{:.1f}'.format(alpha_hi),obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index])
								break

					# fill inclusively
					if 'noweight' in observable:
						getattr(self, hname.format(observable,obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index])
					else:
						getattr(self, hname.format(observable,obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index])

			# if 'jet_pairdist' in observable:
			# 	if  '_PM' in observable:
			# 		obs_type = "PM"
			# 	elif  '_M' in observable:
			# 		obs_type = "M"
			# 	elif  '_P' in observable:
			# 		obs_type = "P"
			# 	elif '_T'in observable:
			# 		obs_type = "T"
			# 	else:
			# 		raise ValueError(f"couldnt' determine obs type from {observable}")
			# 	pairdist_skel = 'jet_pairdist_{}_{}'
			# 	ipoint = 2
			# 	for indices, RL, weight in zip(new_corr.correlator(ipoint).indices(), new_corr.correlator(ipoint).rs(), new_corr.correlator(ipoint).weights()):
			# 		idx1, idx2 = indices
			#		if idx1 <= idx2:
			#			continue
			# 		charges = np.array([c_select[index].python_info().charge for index in indices])
			# 		delta_phi = c_select[idx1].delta_phi_to(c_select[idx2])
			# 		delta_phistar = self.calc_phistar(c_select[idx1], c_select[idx2], c_select[idx1].python_info().charge, c_select[idx2].python_info().charge)

			# 		delta_eta = c_select[idx1].eta() - c_select[idx2].eta()
			# 		if np.all(charges > 0):
			# 			pair_type = "P"
			# 		elif np.all(charges < 0):
			# 			pair_type = "M"
			# 		else:
			# 			pair_type = "PM"

			# 		if obs_type == "T" or pair_type == obs_type:
			# 			getattr(self, hname.format(pairdist_skel.format(obs_type, 'phi'), obs_label)).Fill(jet_pt, delta_phi, 1)
			# 			getattr(self, hname.format(pairdist_skel.format(obs_type, 'phistar'), obs_label)).Fill(jet_pt, delta_phistar, 1)
			# 			getattr(self, hname.format(pairdist_skel.format(obs_type, 'eta'), obs_label)).Fill(jet_pt, delta_eta, 1)
			# 			getattr(self, hname.format(pairdist_skel.format(obs_type, 'RL'), obs_label)).Fill(jet_pt, RL, 1)
			# 			getattr(self, hname.format(pairdist_skel.format(obs_type, 'phistar_eta'), obs_label)).Fill(jet_pt, delta_phistar, delta_eta, 1)

			# if 'E2C' in observable:
		cE2C_observables = [obs for obs in self.observable_list if 'E2C' in obs]
  
		if len(cE2C_observables) >= 1 and not self.pair_eff_on:
			# hname = 'h_{{}}_JetPt_Truth_R{}_{{}}'.format(jetR)
			pair_type_label = ''
			observable_skel = "jet_E2C_{}_RL{}"
			ipoint = 2
			if self.ENC_fastsim and ('Truth' not in hname): # NB: only apply pair efficiency effect for fast sim and det level distributions
				weights_pair = self.get_pair_eff_weights(new_corr, ipoint, c_select)
			for indices, RL, weight in zip(new_corr.correlator(ipoint).indices(), new_corr.correlator(ipoint).rs(), new_corr.correlator(ipoint).weights()):
				# processing only like-sign pairs when self.ENC_pair_like is on
				if self.ENC_pair_like and (not self.is_same_charge(new_corr, ipoint, c_select, index)):
					continue

				# processing only unlike-sign pairs when self.ENC_pair_unlike is on
				if self.ENC_pair_unlike and self.is_same_charge(new_corr, ipoint, c_select, index):
					continue

				# observable = jet_E2C_T_RL
				charges = np.array([c_select[index].python_info().charge for index in indices])

				if np.all(charges > 0):
					getattr(self, hname.format(observable_skel.format('P', ''), obs_label)).Fill(jet_pt, RL, weight)
					# getattr(self, hname.format(observable_skel.format('P', 'Pt'), obs_label)).Fill(jet_pt, jet_pt * RL, weight)
				elif np.all(charges < 0):
					getattr(self, hname.format(observable_skel.format('M', ''), obs_label)).Fill(jet_pt, RL, weight)
					# getattr(self, hname.format(observable_skel.format('M', 'Pt'), obs_label)).Fill(jet_pt, jet_pt * RL, weight)
				else:
					getattr(self, hname.format(observable_skel.format('PM', ''), obs_label)).Fill(jet_pt, RL, weight)
					# getattr(self, hname.format(observable_skel.format('PM', 'Pt'), obs_label)).Fill(jet_pt, jet_pt * RL, weight)
				getattr(self, hname.format(observable_skel.format('Q', ''), obs_label)).Fill(jet_pt, RL, np.prod(charges) * weight)
				# getattr(self, hname.format(observable_skel.format('Q', 'Pt'), obs_label)).Fill(jet_pt, jet_pt * RL, np.prod(charges) * weight)
				getattr(self, hname.format(observable_skel.format('T', ''), obs_label)).Fill(jet_pt, RL, weight)

		elif len(cE2C_observables) >= 1 and self.ENC_fastsim and self.pair_eff_on:
		# name = 'h_{}{}_JetPt_Truth_R{}_{}'.format(observable, pair_type_label, jetR, obs_label)
				# h_{jet_E2C_M_RL}{}_JetPt_R{0.4}_{1.0}
			# getattr(self, hname.format(observable + str(ipoint) + pair_type_label,obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index])
			pair_type_label = ['_qpt', '_phet']
			# hname = 'h_{{}}_JetPt_Truth_R{}_{{}}'.format(jetR)
			observable_skel = "jet_E2C_{}_RL{}"
			# getattr(self, hname.format(observable + str(ipoint) + pair_type_label,obs_label)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index])
			ipoint = 2
			if 'Truth' not in hname: # NB: only apply pair efficiency effect for fast sim and det level distributions
				weights_qpt, weights_phet = self.get_pair_eff_weights(new_corr, ipoint, c_select)
			else:
				ln = len(new_corr.correlator(ipoint).rs())
				weights_qpt = weights_phet = [1] * ln
			for indices, RL, weight, peff_qpt, peff_phet in zip(new_corr.correlator(ipoint).indices(), new_corr.correlator(ipoint).rs(), new_corr.correlator(ipoint).weights(), weights_qpt, weights_phet, strict = True):

				# observable = jet_E2C_T_RL
				charges = np.array([c_select[index].python_info().charge for index in indices])

				if np.all(charges > 0):
					getattr(self, hname.format(observable_skel.format('P', '_qpt'), obs_label)).Fill(jet_pt, RL, weight*peff_qpt)
					getattr(self, hname.format(observable_skel.format('P', '_phet'), obs_label)).Fill(jet_pt, RL, weight*peff_phet)
					# getattr(self, hname.format(observable_skel.format('P', 'Pt'), obs_label)).Fill(jet_pt, jet_pt * RL, weight)
				elif np.all(charges < 0):
					getattr(self, hname.format(observable_skel.format('M', '_qpt'), obs_label)).Fill(jet_pt, RL, weight*peff_qpt)
					getattr(self, hname.format(observable_skel.format('M', '_phet'), obs_label)).Fill(jet_pt, RL, weight*peff_phet)
					# getattr(self, hname.format(observable_skel.format('M', 'Pt'), obs_label)).Fill(jet_pt, jet_pt * RL, weight)
				else:
					getattr(self, hname.format(observable_skel.format('PM', '_qpt'), obs_label)).Fill(jet_pt, RL, weight*peff_qpt)
					getattr(self, hname.format(observable_skel.format('PM', '_phet'), obs_label)).Fill(jet_pt, RL, weight*peff_phet)
					# getattr(self, hname.format(observable_skel.format('PM', 'Pt'), obs_label)).Fill(jet_pt, jet_pt * RL, weight)
				getattr(self, hname.format(observable_skel.format('Q', '_qpt'), obs_label)).Fill(jet_pt, RL, np.prod(charges) * weight*peff_qpt)
				getattr(self, hname.format(observable_skel.format('Q', '_phet'), obs_label)).Fill(jet_pt, RL, np.prod(charges) * weight*peff_phet)
				# getattr(self, hname.format(observable_skel.format('Q', 'Pt'), obs_label)).Fill(jet_pt, jet_pt * RL, np.prod(charges) * weight)
				getattr(self, hname.format(observable_skel.format('T', '_qpt'), obs_label)).Fill(jet_pt, RL, weight*peff_qpt)
				getattr(self, hname.format(observable_skel.format('T', '_phet'), obs_label)).Fill(jet_pt, RL, weight*peff_phet)


		# NOTE: for now ignoring E3C to reduce processing time
		# cE3C_observables = [obs for obs in self.observable_list if 'E3C' in obs]
		# for observable in cE3C_observables:
		# ipoint = 3
		# observable_skel = "jet_E3C_{}_RL{}"
		# for indices, RL, weight in zip(new_corr.correlator(ipoint).indices(), new_corr.correlator(ipoint).rs(), new_corr.correlator(ipoint).weights()):
		# 	# processing only like-sign pairs when self.ENC_pair_like is on
		# 	if self.ENC_pair_like and (not self.is_same_charge(new_corr, ipoint, c_select, index)):
		# 		continue

		# 	# processing only unlike-sign pairs when self.ENC_pair_unlike is on
		# 	if self.ENC_pair_unlike and self.is_same_charge(new_corr, ipoint, c_select, index):
		# 		continue

		# 	charges = np.array([c_select[index].python_info().charge for index in indices])
		# 	# if np.all(charges > 0):
		# 	# 	getattr(self, hname2.format('P', '')).Fill(jet_pt, RL, weight)
		# 	# 	getattr(self, hname2.format('P', 'Pt')).Fill(jet_pt, jet_pt * RL, weight)
		# 	# elif np.all(charges < 0):
		# 	# 	getattr(self, hname2.format('M', '')).Fill(jet_pt, RL, weight)
		# 	# 	getattr(self, hname2.format('M', 'Pt')).Fill(jet_pt, jet_pt * RL, weight)
		# 	# else:
		# 	# 	getattr(self, hname2.format('PM', '')).Fill(jet_pt, RL, weight)
		# 	# 	getattr(self, hname2.format('PM', 'Pt')).Fill(jet_pt, jet_pt * RL, weight)
		# 	getattr(self, hname.format(observable_skel.format('Q', ''), obs_label)).Fill(jet_pt, RL, np.prod(charges) * weight)
		# 	getattr(self, hname.format(observable_skel.format('Q', 'Pt'), obs_label)).Fill(jet_pt, jet_pt * RL, np.prod(charges) * weight)
		# 	getattr(self, hname.format(observable_skel.format('T', ''), obs_label)).Fill(jet_pt, RL, weight)
		# 	getattr(self, hname.format(observable_skel.format('T', 'Pt'), obs_label)).Fill(jet_pt, jet_pt * RL, weight)

	#---------------------------------------------------------------
	# This function is called per observable per jet subconfigration
	# used in fill_matched_jet_histograms
	# This function is created because we cannot use fill_observable_histograms
	# directly because observable list loop inside that function
	#---------------------------------------------------------------
	def fill_matched_observable_histograms(self, hname, observable, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_pt_ungroomed, jet_pt_matched, cone_parts = None):

		constituents = fj.sorted_by_pt(jet.constituents())
		if cone_parts is not None:
			constituents = fj.sorted_by_pt(cone_parts)

		# if cone_parts!=None and 'Truth' in hname:
		#   print('Nconst in cone',len(constituents))
		# if cone_parts==None and 'Truth' in hname:
		#   print('Nconst in jet',len(constituents))

		c_select = fj.vectorPJ()
		trk_thrd = obs_setting

		for c in constituents:
			if c.pt() < trk_thrd:
				break
			c_select.append(c) # NB: use the break statement since constituents are already sorted

		if self.ENC_pair_cut and ('Truth' not in hname):
			dphi_cut = -9999 # means no dphi cut
			deta_cut = 0.008
		else:
			dphi_cut = -9999
			deta_cut = -9999

		# Only need rho subtraction for det-level jets
		# if self.do_rho_subtraction and ('Truth' not in hname):
		# 	jet_pt = jet_pt_ungroomed
		# else:
			# we use jet_pt_ungroomed #1 for sel jet pt and jet_pt_matched #2 for weight jet pt
		jet_pt_sel = jet_pt_ungroomed
		jet_pt_weight = jet_pt_matched
			# if "_Truth_" in hname:
			# if "TruthJetPt" in hname or "TruthPairs" in hname:
			# 	jet_pt = jet_pt_matched
			# else:
			# 	jet_pt = jet.perp()

		new_corr = ecorrel.CorrelatorBuilder(c_select, jet_pt_weight, 2, 1, dphi_cut, deta_cut)
		if 'ENC' in observable or 'EEC_noweight' in observable or 'EEC_weight2' in observable:
			for ipoint in range(2, 3):
				if self.ENC_fastsim and ('Truth' not in hname): # NB: only apply pair efficiency effect for fast sim and det level distributions
					weights_pair = self.get_pair_eff_weights(new_corr, ipoint, c_select)

				for index in range(new_corr.correlator(ipoint).rs().size()):
					pair_type_label = ''
					if self.do_rho_subtraction or self.do_constituent_subtraction:
						pair_type = self.check_pair_type(new_corr, ipoint, c_select, index)
						pair_type_label = self.pair_type_labels[pair_type]

					if 'ENC' in observable:
						if self.ENC_fastsim and ('Truth' not in hname):
							getattr(self, hname.format(observable + str(ipoint) + pair_type_label,obs_label)).Fill(jet_pt_matched, new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index]*weights_pair[index])
						else:
							getattr(self, hname.format(observable + str(ipoint) + pair_type_label,obs_label)).Fill(jet_pt_matched, new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index]) # NB: use jet_pt_matched instead of jet_pt so if jet_pt_matched is different from jet_pt, it will be used. This is mainly for matched jets study

					if ipoint==2 and 'EEC_noweight' in observable:
						if self.ENC_fastsim and ('Truth' not in hname):
							getattr(self, hname.format(observable + pair_type_label,obs_label)).Fill(jet_pt_matched, new_corr.correlator(ipoint).rs()[index], weights_pair[index])
						else:
							getattr(self, hname.format(observable + pair_type_label,obs_label)).Fill(jet_pt_matched, new_corr.correlator(ipoint).rs()[index])

					if ipoint==2 and 'EEC_weight2' in observable:
						if self.ENC_fastsim and ('Truth' not in hname):
							getattr(self, hname.format(observable + pair_type_label,obs_label)).Fill(jet_pt_matched, new_corr.correlator(ipoint).rs()[index], pow(new_corr.correlator(ipoint).weights()[index]*weights_pair[index],2))
						else:
							getattr(self, hname.format(observable + pair_type_label,obs_label)).Fill(jet_pt_matched, new_corr.correlator(ipoint).rs()[index], pow(new_corr.correlator(ipoint).weights()[index],2))
		if 'E2C' in observable:
			for ipoint in range(2, 3):
				# if self.ENC_fastsim and ('Truth' not in hname): # NB: only apply pair efficiency effect for fast sim and det level distributions
				# 	weights_pair = self.get_pair_eff_weights(new_corr, ipoint, c_select)
				for indices, RL, weight in zip(new_corr.correlator(ipoint).indices(), new_corr.correlator(ipoint).rs(), new_corr.correlator(ipoint).weights()):
					pair_type_label = ''
					if self.do_rho_subtraction or self.do_constituent_subtraction:
						pair_type = self.check_pair_type(new_corr, ipoint, c_select, index)
						pair_type_label = self.pair_type_labels[pair_type]

					charges = np.array([c_select[index].python_info().charge for index in indices])

					if np.all(charges > 0):
						if '_P_' in observable:
							getattr(self, hname.format(observable + pair_type_label,obs_label)).Fill(jet_pt_sel, RL, weight)
					elif np.all(charges < 0):
						if '_M_' in observable:
							getattr(self, hname.format(observable + pair_type_label,obs_label)).Fill(jet_pt_sel, RL, weight)
					else:
						if '_PM_' in observable:
							getattr(self, hname.format(observable + pair_type_label,obs_label)).Fill(jet_pt_sel, RL, weight)

					if "_Q_" in observable:
						getattr(self, hname.format(observable + pair_type_label,obs_label)).Fill(jet_pt_sel, RL, weight * np.prod(charges))
					elif "_T_" in observable:
						getattr(self, hname.format(observable + pair_type_label,obs_label)).Fill(jet_pt_sel, RL, weight)
					# getattr(self, hname.format(pair_type_label,obs_label)).Fill(jet_pt, RL, weight)

		# if 'pairdist' in observable:
		if 'jet_pt' in observable:
			getattr(self, hname.format(observable,obs_label)).Fill(jet.perp())
		if 'jet_pairdist' in observable:
			if  '_PM' in observable:
				obs_type = "PM"
			elif  '_M' in observable:
				obs_type = "M"
			elif  '_P' in observable:
				obs_type = "P"
			elif '_T'in observable:
				obs_type = "T"
			else:
				raise ValueError(f"couldnt' determine obs type from {observable}")

			# print(observable)
			# if "JetPt" in hname:
			# 	xaxis = "JetPt"
			# elif "PairKt" in hname:
			# 	xaxis = "PairKt"
			# else:
			# 	raise ValueError(f"couldn't determine xaxis from {hname}")


			ipoint = 2     
			for indices, RL, weight in zip(new_corr.correlator(ipoint).indices(), new_corr.correlator(ipoint).rs(), new_corr.correlator(ipoint).weights()):
				idx1, idx2 = indices
				if idx1 <= idx2:
					continue

				charges = np.array([c_select[index].python_info().charge for index in indices])
				delta_phi = c_select[idx1].delta_phi_to(c_select[idx2])
				delta_phistar = self.calc_phistar(c_select[idx1], c_select[idx2], c_select[idx1].python_info().charge, c_select[idx2].python_info().charge)
				delta_eta = c_select[idx1].eta() - c_select[idx2].eta()
				pair_kt = (c_select[idx1].pt() + c_select[idx2].pt()) / 2
				if np.all(charges > 0):
					pair_type = "P"
				elif np.all(charges < 0):
					pair_type = "M"
				else:
					pair_type = "PM"
				# print('pair_type', pair_type)

				if obs_type == "T" or pair_type == obs_type:
					if "JetPt" in hname:
						xval = jet.pt()
					elif "PairKt" in hname:
						xval = pair_kt
					# print('filling')
					# print(hname.format(f"{observable}_phi", obs_label))
					getattr(self, hname.format(f"{observable}_phi", obs_label)).Fill(xval, delta_phi, 1)
					getattr(self, hname.format(f"{observable}_phistar", obs_label)).Fill(xval, delta_phistar, 1)
					getattr(self, hname.format(f"{observable}_eta", obs_label)).Fill(xval, delta_eta, 1)
					getattr(self, hname.format(f"{observable}_RL", obs_label)).Fill(xval, RL, 1)
					getattr(self, hname.format(f"{observable}_phistar_eta", obs_label)).Fill(xval, delta_phistar, delta_eta, 1)


			# getattr(self, hname.format(f"{observable}_RL",obs_label)).Fill(jet.perp(), 0.3, 1)

		if 'jet_pair_creco' in observable:
			histo_pair_type = observable.split("_")[-1]
			# print(observable)
			# print(histo_pair_type)
			# hname = f'h_matched_{observable}'

			reco_truth = [part for part in jet.constituents() if part.python_info().particle_det is not None]
			# name = f'h_matched_{observable}_Truth'
			# h = ROOT.TH2D(name, name, self.pT_nbins, self.pT_bins, self.kT_nbins, self.kT_bins)

			jet_truth_pt = jet.pt()
			if len(reco_truth) < 2:
				return
			# index_pairs = itertools.combinations(range(len(reco_truth)), 2)
			# for i1, i2 in index_pairs:
			# 	p1_truth = reco_truth[i1]
			# 	p2_truth = reco_truth[i2]
			for p1_truth, p2_truth in itertools.combinations(reco_truth, 2):
				charges_truth = np.array([p1_truth.python_info().charge, p2_truth.python_info().charge])
				if np.all(charges_truth > 0):
					pair_type_truth = "P"
				elif np.all(charges_truth < 0):
					pair_type_truth = "M"
				else:
					pair_type_truth = "PM"
				if pair_type_truth != histo_pair_type:
					continue

				# pair_kt_truth = (p1_truth.pt() + p2_truth.pt()) / 2
				pair_kt_truth = self.calc_kt(p1_truth, p2_truth)
				getattr(self, f"{hname}_Truth").Fill(pair_kt_truth, jet_truth_pt)

				p1_det = p1_truth.python_info().particle_det
				p2_det = p2_truth.python_info().particle_det
				charges_det = np.array([p1_det.python_info().charge, p2_det.python_info().charge])
				if np.all(charges_det > 0):
					pair_type_det = "P"
				elif np.all(charges_det < 0):
					pair_type_det = "M"
				else:
					pair_type_det = "PM"
				# pair_kt_det = (p1_det.pt() + p2_det.pt()) / 2

				# print(pair_kt_det - pair_kt_truth)

				if pair_type_det == pair_type_truth:
					getattr(self, hname).Fill(pair_kt_truth, jet_truth_pt)



	#---------------------------------------------------------------
	# This function is called per jet subconfigration
	# Fill matched jet histograms
	#---------------------------------------------------------------
	def fill_matched_jet_histograms(self, jet_det, jet_det_groomed_lund, jet_truth, jet_truth_groomed_lund, jet_pp_det, jetR, obs_setting, grooming_setting, obs_label, jet_pt_det_ungroomed, jet_pt_truth_ungroomed, R_max, suffix, **kwargs):
		# If jetscape, we will need to correct substructure observable for holes (pt is corrected in base class)
		# For ENC in PbPb, jet_pt_det_ungroomed stores the corrected jet pT
		if not type(jet_det.python_info().match):
			print("type is ", type(jet_det.python_info().match))
		# if self.jetscape:
		# 	holes_in_det_jet = kwargs['holes_in_det_jet']
		# 	holes_in_truth_jet = kwargs['holes_in_truth_jet']

		cone_parts_in_det_jet = kwargs['cone_parts_in_det_jet']
		cone_parts_in_truth_jet = kwargs['cone_parts_in_truth_jet']
		cone_R = kwargs['cone_R']

		# Todo: add additonal weight for jet pT spectrum
		# if self.rewight_pt:
		#   w_pt = 1+pow(jet_truth,0.2)
		# else:
		#   w_pt = 1

		if self.do_rho_subtraction:
			# print('evt #',self.event_number)
			jet_pt_det = jet_pt_det_ungroomed
			# print('Det: pT',jet_det.perp(),'(',jet_pt_det,')','phi',jet_det.phi(),'eta',jet_det.eta())
			# print('Truth: pT',jet_truth.perp(),'phi',jet_truth.phi(),'eta',jet_truth.eta())
			# print('Difference pT (truth-det)',jet_truth.perp()-jet_pt_det_ungroomed)
		else:
			jet_pt_det = jet_det.perp()

		for observable in self.observable_list:

			if cone_R == 0: # fill for jet constituents
				if "jet_pair_creco" in observable:
					hname = f'h_matched_{observable}'
					self.fill_matched_observable_histograms(hname, observable, jet_truth, jet_det_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_pt_det, jet_truth.pt())
				if "jet_pairdist" in observable:
					hname = 'h_matched_{{}}_JetPt_R{}_{{}}'.format(jetR)
					self.fill_matched_observable_histograms(hname, observable, jet_det, jet_det_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_pt_det, jet_pt_det)

					hname = 'h_matched_{{}}_JetPt_Truth_R{}_{{}}'.format(jetR)
					self.fill_matched_observable_histograms(hname, observable, jet_truth, jet_truth_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_truth.pt(), jet_truth.pt())

					hname = 'h_matched_{{}}_PairKt_R{}_{{}}'.format(jetR)
					self.fill_matched_observable_histograms(hname, observable, jet_det, jet_det_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_pt_det, jet_pt_det)

					hname = 'h_matched_{{}}_PairKt_Truth_R{}_{{}}'.format(jetR)
					self.fill_matched_observable_histograms(hname, observable, jet_truth, jet_truth_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_truth.pt(), jet_truth.pt())

				# if "E2C" in observable:
				# 	hname = 'h_matched_{{}}_JetPt_TruthJetPtWeight_R{}_{{}}'.format(jetR)
				# 	self.fill_matched_observable_histograms(hname, observable, jet_det, jet_truth_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_pt_det, jet_truth.pt())
				# 	hname = 'h_matched_{{}}_JetPt_TruthJetPtSel_R{}_{{}}'.format(jetR)
				# 	self.fill_matched_observable_histograms(hname, observable, jet_det, jet_truth_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_truth.pt(), jet_pt_det)
				# 	hname = 'h_matched_{{}}_JetPt_TruthJetPtWeightSel_R{}_{{}}'.format(jetR)
				# 	self.fill_matched_observable_histograms(hname, observable, jet_det, jet_truth_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_truth.pt(), jet_truth.pt())
				# 	hname = 'h_matched_{{}}_JetPt_TruthPairs_R{}_{{}}'.format(jetR)
				# 	self.fill_matched_observable_histograms(hname, observable, jet_truth, jet_truth_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_pt_det, jet_pt_det)
				# fill RL vs matched truth jet pT for det jets (only fill these extra histograms for ENC or pair distributions)
				# if 'ENC' in observable or 'EEC_noweight' in observable or 'EEC_weight2' in observable:
				# 	hname = 'h_matched_extra_{{}}_JetPt_R{}_{{}}'.format(jetR)
				# 	self.fill_matched_observable_histograms(hname, observable, jet_det, jet_det_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_pt_det, jet_truth.pt()) # NB: use the truth jet pt so the reco jets histograms are comparable to matched truth jets. However this also means that two identical histograms will be filled fot jet_pt observable

				# Fill correlation between matched det and truth jets
				if 'jet_pt' in observable:
					hname = 'h_matched_{}_JetPt_Truth_vs_Det_R{}_{}'.format(observable, jetR, obs_label)
					getattr(self, hname).Fill(jet_pt_det, jet_truth.pt())
					hname = 'h_matched_{}_JetPt_R{}_{}'.format(observable, jetR, obs_label)
					getattr(self, hname).Fill(jet_pt_det)
					hname = 'h_matched_{}_JetPt_Truth_R{}_{}'.format(observable, jetR, obs_label)
					getattr(self, hname).Fill(jet_truth.pt())

			else: # fill for cone parts around jet
				if 'ENC' in observable or 'EEC_noweight' in observable or 'EEC_weight2' in observable:
					hname = 'h_jetcone{}_matched_{{}}_JetPt_R{}_{{}}'.format(cone_R, jetR)
					self.fill_matched_observable_histograms(hname, observable, jet_det, jet_det_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_pt_det, jet_pt_det, cone_parts_in_det_jet)

					hname = 'h_jetcone{}_matched_{{}}_JetPt_Truth_R{}_{{}}'.format(cone_R, jetR)
					self.fill_matched_observable_histograms(hname, observable, jet_truth, jet_truth_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_pt_det, jet_truth.pt(), cone_parts_in_truth_jet)

					hname = 'h_jetcone{}_matched_extra_{{}}_JetPt_R{}_{{}}'.format(cone_R, jetR)
					self.fill_matched_observable_histograms(hname, observable, jet_det, jet_det_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_pt_det, jet_truth.pt(), cone_parts_in_det_jet)


		# # Find all subjets
		# trk_thrd = obs_setting
		# cs_subjet_det = fj.ClusterSequence(jet_det.constituents(), self.subjet_def[trk_thrd])
		# subjets_det = fj.sorted_by_pt(cs_subjet_det.inclusive_jets())

		# cs_subjet_truth = fj.ClusterSequence(jet_truth.constituents(), self.subjet_def[trk_thrd])
		# subjets_truth = fj.sorted_by_pt(cs_subjet_truth.inclusive_jets())

		# if not self.is_pp:
		#   cs_subjet_det_pp = fj.ClusterSequence(jet_pp_det.constituents(), self.subjet_def[trk_thrd])
		#   subjets_det_pp = fj.sorted_by_pt(cs_subjet_det_pp.inclusive_jets())

		# # Loop through subjets and set subjet matching candidates for each subjet in user_info
		# if self.is_pp:
		#     [[self.set_matching_candidates(subjet_det, subjet_truth, subjetR, 'hDeltaR_ppdet_pptrue_ENC_R{}_{}'.format(jetR, subjetR)) for subjet_truth in subjets_truth] for subjet_det in subjets_det]
		# else:
		#     # First fill the combined-to-pp matches, then the pp-to-pp matches
		#     [[self.set_matching_candidates(subjet_det_combined, subjet_det_pp, subjetR, 'hDeltaR_combined_ppdet_ENC_R{}_{}_Rmax{}'.format(jetR, subjetR, R_max), fill_jet1_matches_only=True) for subjet_det_pp in subjets_det_pp] for subjet_det_combined in subjets_det]
		#     [[self.set_matching_candidates(subjet_det_pp, subjet_truth, subjetR, 'hDeltaR_ppdet_pptrue_ENC_R{}_{}_Rmax{}'.format(jetR, subjetR, R_max)) for subjet_truth in subjets_truth] for subjet_det_pp in subjets_det_pp]

		# # Loop through subjets and set accepted matches
		# if self.is_pp:
		#     [self.set_matches_pp(subjet_det, 'hSubjetMatchingQA_R{}_{}'.format(jetR, subjetR)) for subjet_det in subjets_det]
		# else:
		#     [self.set_matches_AA(subjet_det_combined, subjetR, 'hSubjetMatchingQA_R{}_{}'.format(jetR, subjetR)) for subjet_det_combined in subjets_det]

		# # Loop through matches and fill histograms
		# for observable in self.observable_list:

		#   # Fill inclusive subjets
		#   if 'inclusive' in observable:

		#     for subjet_det in subjets_det:

		#       z_det = subjet_det.pt() / jet_det.pt()

		#       # If z=1, it will be default be placed in overflow bin -- prevent this
		#       if np.isclose(z_det, 1.):
		#         z_det = 0.999

		#       successful_match = False

		#       if subjet_det.has_user_info():
		#         subjet_truth = subjet_det.python_info().match

		#         if subjet_truth:

		#           successful_match = True

		#           # For subjet matching radius systematic, check distance between subjets
		#           if self.matching_systematic:
		#             if subjet_det.delta_R(subjet_truth) > 0.5 * self.jet_matching_distance * subjetR:
		#               continue

		#           z_truth = subjet_truth.pt() / jet_truth.pt()

		#           # If z=1, it will be default be placed in overflow bin -- prevent this
		#           if np.isclose(z_truth, 1.):
		#             z_truth = 0.999

		#           # In Pb-Pb case, fill matched pt fraction
		#           if not self.is_pp:
		#             self.fill_subjet_matched_pt_histograms(observable,
		#                                                    subjet_det, subjet_truth,
		#                                                    z_det, z_truth,
		#                                                    jet_truth.pt(), jetR, subjetR, R_max)

		#           # Fill histograms
		#           # Note that we don't fill 'matched' histograms here, since that is only
		#           # meaningful for leading subjets
		#           self.fill_response(observable, jetR, jet_pt_det_ungroomed, jet_pt_truth_ungroomed,
		#                              z_det, z_truth, obs_label, R_max, prong_match=False)

		#       # Fill number of subjets with/without unique match, as a function of zr
		#       if self.is_pp:
		#         name = 'h_match_fraction_{}_R{}_{}'.format(observable, jetR, subjetR)
		#         getattr(self, name).Fill(jet_truth.pt(), z_det, successful_match)

		#   # Get leading subjet and fill histograms
		#   if 'leading' in observable:

		#     leading_subjet_det = self.utils.leading_jet(subjets_det)
		#     leading_subjet_truth = self.utils.leading_jet(subjets_truth)

		#     # Note that we don't want to check whether they are geometrical matches
		#     # We rather want to correct the measured leading subjet to the true leading subjet
		#     if leading_subjet_det and leading_subjet_truth:

		#       z_leading_det = leading_subjet_det.pt() / jet_det.pt()
		#       z_leading_truth = leading_subjet_truth.pt() / jet_truth.pt()

		#       # If z=1, it will be default be placed in overflow bin -- prevent this
		#       if np.isclose(z_leading_det, 1.):
		#         z_leading_det = 0.999
		#       if np.isclose(z_leading_truth, 1.):
		#         z_leading_truth = 0.999

		#       # In Pb-Pb case, fill matched pt fraction
		#       if not self.is_pp:
		#         match = self.fill_subjet_matched_pt_histograms(observable,
		#                                                        leading_subjet_det, leading_subjet_truth,
		#                                                        z_leading_det, z_leading_truth,
		#                                                        jet_truth.pt(), jetR, subjetR, R_max)
		#       else:
		#         match = False

		#       # Fill histograms
		#       self.fill_response(observable, jetR, jet_pt_det_ungroomed, jet_pt_truth_ungroomed,
		#                          z_leading_det, z_leading_truth, obs_label, R_max, prong_match=match)

		#       # Plot deltaR distribution between the detector and truth leading subjets
		#       # (since they are not matched geometrically, the true leading may not be the measured leading
		#       deltaR = leading_subjet_det.delta_R(leading_subjet_truth)
		#       name = 'hDeltaR_det_truth_{}_R{}_{}'.format(observable, jetR, subjetR)
		#       if not self.is_pp:
		#         name += '_Rmax{}'.format(R_max)
		#       getattr(self, name).Fill(jet_truth.pt(), z_leading_truth, deltaR)

	#---------------------------------------------------------------
	# Do prong-matching
	#---------------------------------------------------------------
	# def fill_subjet_matched_pt_histograms(self, observable, subjet_det, subjet_truth,
	#                                       z_det, z_truth, jet_pt_truth, jetR, subjetR, R_max):

		# # Get pp det-level subjet
		# # Inclusive case: This is matched to the combined subjet (and its pp truth-level subjet)
		# # Leading case: This is matched only to the pp truth-level leading subjet
		# subjet_pp_det = None
		# if subjet_truth.has_user_info():
		#   subjet_pp_det = subjet_truth.python_info().match
		# if not subjet_pp_det:
		#   return

		# matched_pt = fjtools.matched_pt(subjet_det, subjet_pp_det)
		# name = 'h_{}_matched_pt_JetPt_R{}_{}_Rmax{}'.format(observable, jetR, subjetR, R_max)
		# getattr(self, name).Fill(jet_pt_truth, z_det, matched_pt)

		# # Plot dz between det and truth subjets
		# deltaZ = z_det - z_truth
		# name = 'h_{}_matched_pt_deltaZ_JetPt_R{}_{}_Rmax{}'.format(observable, jetR, subjetR, R_max)
		# getattr(self, name).Fill(jet_pt_truth, matched_pt, deltaZ)

		# # Plot dR between det and truth subjets
		# deltaR = subjet_det.delta_R(subjet_truth)
		# name = 'h_{}_matched_pt_deltaR_JetPt_R{}_{}_Rmax{}'.format(observable, jetR, subjetR, R_max)
		# getattr(self, name).Fill(jet_pt_truth, matched_pt, deltaR)

		# match = (matched_pt > 0.5)
		# return match

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

	handler = logging.StreamHandler()

	# Create a formatter and set it for the handler
	formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(funcName)s - %(message)s')
	handler.setFormatter(formatter)
	handler.setLevel(logging.INFO)
	logger.addHandler(handler)
	logger.setLevel(logging.DEBUG)

	logger.info('Configuring...')
	logger.info(f"Input file: \'{args.input_file}\'")
	logger.info(f"Config file: \'{args.config_file}\'")
	logger.info(f"Ouput directory: '{args.output_dir}'")

	# If invalid inputFile is given, exit
	if not os.path.exists(args.input_file):
		logger.critical(f"Input file '{args.input_file}' does not exist, exiting.")
		sys.exit(1)

	# If invalid configFile is given, exit
	if not os.path.exists(args.config_file):
		logger.critical(f"Config file '{args.config_file}' does not exist, exiting.")
		sys.exit(1)


	analysis = ProcessMC_ENC(input_file=args.input_file, config_file=args.config_file, output_dir=args.output_dir)
	analysis.process_mc()