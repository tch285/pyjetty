#!/usr/bin/env python3

"""
Base class to read a ROOT TTree of track information
and do jet-finding, and save basic histograms.
    
To use this class, the following should be done:

    - Implement a user analysis class inheriting from this one, such as in user/james/process_mc_XX.py
        You should implement the following functions:
            - initialize_user_output_objects_R()
            - fill_observable_histograms()
            - fill_matched_jet_histograms()
        
    - You should include the following histograms:
            - Response matrix: hResponse_JetPt_[obs]_R[R]_[subobs]_[grooming setting]
            - Residual distribution: hResidual_JetPt_[obs]_R[R]_[subobs]_[grooming setting]

    - You also should modify observable-specific functions at the top of common_utils.py
    
Author: James Mulligan (james.mulligan@berkeley.edu)
"""

from __future__ import print_function

# General
import time
import logging

# Data analysis and plotting
import pandas
import numpy as np
from array import array
import ROOT
import yaml
import random

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib
# import fjtools

# Analysis utilities
from pyjetty.alice_analysis.process.base import process_io
# from pyjetty.alice_analysis.process.base import process_io_pPb as process_io
from pyjetty.alice_analysis.process.base import process_io_emb
from pyjetty.alice_analysis.process.base import process_base
from pyjetty.alice_analysis.process.base import thermal_generator
from pyjetty.alice_analysis.process.base import jet_info
from pyjetty.mputils.csubtractor import CEventSubtractor

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
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s')
# handler.setFormatter(formatter)
handler.setFormatter(ColoredFormatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(funcName)s - %(message)s'))

logger.addHandler(handler)
logger.setLevel(logging.INFO)

def linbins(xmin, xmax, nbins):
    lspace = np.linspace(xmin, xmax, nbins+1)
    # arr = array.array('f', lspace)
    return lspace

def logbins(xmin, xmax, nbins):
    lspace = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
    # arr = array.array('f', lspace)
    return lspace

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)

################################################################
class ProcessMCBase(process_base.ProcessBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
    
        # Initialize base class
        super(ProcessMCBase, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)
        
        # Initialize configuration
        self.initialize_config()
        
    #---------------------------------------------------------------
    # Initialize config file into class members
    #---------------------------------------------------------------
    def initialize_config(self):
        
        # Call base class initialization
        process_base.ProcessBase.initialize_config(self)
        
        # Read config file
        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)
            
        self.fast_simulation = config['fast_simulation']
        if self.fast_simulation:
            if 'ENC_fastsim' in config:
                    self.ENC_fastsim = config['ENC_fastsim']
            else:
                    self.ENC_fastsim = False
        else: # if not fast simulation, set ENC_fastsim flag to False
            self.ENC_fastsim = False    
        if 'ENC_pair_cut' in config:
            self.ENC_pair_cut = config['ENC_pair_cut']
        else:
            self.ENC_pair_cut = False
        if 'ENC_pair_like' in config:
            self.ENC_pair_like = config['ENC_pair_like']
        else:
            self.ENC_pair_like = False
        if 'ENC_pair_unlike' in config:
            self.ENC_pair_unlike = config['ENC_pair_unlike']
        else:
            self.ENC_pair_unlike = False
        if 'jetscape' in config:
            self.jetscape = config['jetscape']
        else:
            self.jetscape = False
        if 'event_plane_angle' in config:
            self.event_plane_range = config['event_plane_angle']
        else:
            self.event_plane_range = None
        if 'matching_systematic' in config:
            self.matching_systematic = config['matching_systematic']
        else:
            self.matching_systematic = False
        self.dry_run = config['dry_run']
        self.skip_deltapt_RC_histograms = True
        self.fill_RM_histograms = True
        
        self.jet_matching_distance = config['jet_matching_distance']
        self.reject_tracks_fraction = config['reject_tracks_fraction']
        if 'mc_fraction_threshold' in config:
            self.mc_fraction_threshold = config['mc_fraction_threshold']
        if 'do_median_subtraction' in config:
            self.do_median_subtraction = config['do_median_subtraction']
        self.do_track_mc_matching = config['do_track_mc_matching'] if 'do_track_mc_matching' in config else False
        self.strict_mc_match = config['strict_mc_match'] if 'strict_mc_match' in config else False
        
        self.do_perpendicular_cone = config['do_perpendicular_cone']
        self.randomize_cone = config['randomize_cone']
        if self.do_constituent_subtraction:
            self.is_pp = False
            self.is_pA = False
            self.emb_file_list = config['emb_file_list']
            self.main_R_max = config['constituent_subtractor']['main_R_max']
            if self.do_perpendicular_cone:
                self.is_pA = True
        else:
            self.is_pp = True
            self.is_pA = False
        
        if 'thermal_model' in config:
            self.thermal_model = True
            beta = config['thermal_model']['beta']
            N_avg = config['thermal_model']['N_avg']
            sigma_N = config['thermal_model']['sigma_N']
            self.thermal_generator = thermal_generator.ThermalGenerator(N_avg, sigma_N, beta)
        else:
            self.thermal_model = False
        
        self.do_emb_rotation = config['do_emb_rotation'] if 'do_emb_rotation' in config else False
        self.do_feedin_check = config['do_feedin_check'] if 'do_feedin_check' in config else False
        self.do_pt_dep_track_rej = config['pt_dep_track_rej'] if 'pt_dep_track_rej' in config else False
        logger.info(f"pT-dependent track rejection: {self.do_pt_dep_track_rej}")
        if self.do_pt_dep_track_rej:
            self.do_pt_dep_track_rej_file = config['pt_dep_track_rej_file']
            if 'pt_dep_track_rej_seed' in config and config['pt_dep_track_rej_seed']:
                self.rng_seed = config['pt_dep_track_rej_seed']
            else:
                self.rng_seed = None
            logger.info(f"pT-dependent track rejection file: {self.do_pt_dep_track_rej_file}")
            logger.info(f"pT-dependent track rejection seed: {self.rng_seed}")
        else:
            self.do_pt_dep_track_rej_file = ""
        
        if 'pT_binning' in config.keys():
            self.pT_min, self.pT_max, self.pT_nbins = config["pT_binning"]
            self.pT_bins = linbins(self.pT_min,self.pT_max,self.pT_nbins)
        if 'RL_binning' in config.keys():
            self.RL_min, self.RL_max, self.RL_nbins = config["RL_binning"]
            self.RL_bins = logbins(self.RL_min,self.RL_max,self.RL_nbins)
        if 'pTRL_binning' in config.keys():
            self.pTRL_min, self.pTRL_max, self.pTRL_nbins = config["pTRL_binning"]
            self.pTRL_bins = logbins(self.pTRL_min,self.pTRL_max,self.pTRL_nbins)
        
        if "kT_binning" in config.keys():
            self.kT_min, self.kT_max, self.kT_nbins = config["kT_binning"]
            self.kT_bins = linbins(self.kT_min,self.kT_max,self.kT_nbins)
        if "pairdist_binning" in config.keys():
            self.pairdist_min, self.pairdist_max, self.pairdist_nbins = config["pairdist_binning"]
            self.pairdist_bins = linbins(self.pairdist_min,self.pairdist_max,self.pairdist_nbins)

        # Create dictionaries to store grooming settings and observable settings for each observable
        # Each dictionary entry stores a list of subconfiguration parameters
        #   The observable list stores the observable setting, e.g. subjetR
        #   The grooming list stores a list of grooming settings {'sd': [zcut, beta]} or {'dg': [a]}
        self.observable_list = config['process_observables'] if 'process_observables' in config else []
        self.obs_settings = {}
        self.obs_grooming_settings = {}
        for observable in self.observable_list:
            obs_config_dict = config[observable]
            # obs_config_list = [name for name in list(obs_config_dict.keys()) if 'config' in name ]
            
            obs_subconfig_list = [name for name in list(obs_config_dict.keys()) if 'config' in name]
            self.obs_settings[observable] = self.utils.obs_settings(observable, obs_config_dict, obs_subconfig_list)
            self.obs_grooming_settings[observable] = self.utils.grooming_settings(obs_config_dict)
            
        # Construct set of unique grooming settings
        self.grooming_settings = []
        lists_grooming = [self.obs_grooming_settings[obs] for obs in self.observable_list]
        for observable in lists_grooming:
            for setting in observable:
                if setting not in self.grooming_settings and setting is not None:
                    self.grooming_settings.append(setting)
    
    #---------------------------------------------------------------
    # Main processing function
    #---------------------------------------------------------------
    def process_mc(self):
        
        self.start_time = time.time()
        
        # ------------------------------------------------------------------------
        
        # Use IO helper class to convert detector-level ROOT TTree into
        # a SeriesGroupBy object of fastjet particles per event
        if self.fast_simulation:
            tree_dir = ''
        else:
            tree_dir = 'PWGHF_TreeCreator'
        io_det = process_io.ProcessIO(input_file=self.input_file, tree_dir=tree_dir,
                                      track_tree_name='tree_Particle', use_ev_id_ext=False,
                                      is_jetscape=self.jetscape, event_plane_range=self.event_plane_range,
                                      is_ENC=self.ENC_fastsim, is_det_level=True, load_mult = True)
                                    #   is_ENC=self.ENC_fastsim, is_det_level=True, load_mult = False)
        # df_fjparticles_det, self.df_events_det = io_det.load_data(m=self.m, reject_tracks_fraction=self.reject_tracks_fraction)
        df_fjparticles_det = io_det.load_data(m=self.m, reject_tracks_fraction=self.reject_tracks_fraction)
        self.df_events_det = io_det.event_df
        self.nEvents_det = len(df_fjparticles_det.index)
        self.nTracks_det = len(io_det.track_df.index)
        logger.info(f'Det particles loaded: --- {time.time() - self.start_time:.3f} seconds ---')
        
        # If jetscape, store also the negative status particles (holes)
        if self.jetscape:
            io_det_holes = process_io.ProcessIO(input_file=self.input_file, tree_dir=tree_dir,
                                                track_tree_name='tree_Particle', use_ev_id_ext=False,
                                                is_jetscape=self.jetscape, holes=True,
                                                event_plane_range=self.event_plane_range)
            df_fjparticles_det_holes = io_det_holes.load_data(m=self.m, reject_tracks_fraction=self.reject_tracks_fraction)
            self.nEvents_det_holes = len(df_fjparticles_det_holes.index)
            self.nTracks_det_holes = len(io_det_holes.track_df.index)
            logger.info(f'Det holes loaded: --- {time.time() - self.start_time:.3f} seconds ---')
        
        # ------------------------------------------------------------------------

        # Use IO helper class to convert truth-level ROOT TTree into
        # a SeriesGroupBy object of fastjet particles per event
        io_truth = process_io.ProcessIO(input_file=self.input_file, tree_dir=tree_dir,
                                        track_tree_name='tree_Particle_gen', use_ev_id_ext=False,
                                        is_jetscape=self.jetscape, event_plane_range=self.event_plane_range,
                                        is_ENC=self.ENC_fastsim, is_det_level=False, load_mult = True)
                                        # is_ENC=self.ENC_fastsim, is_det_level=False, load_mult = False)
        # df_fjparticles_truth, self.df_events_truth = io_truth.load_data(m=self.m) # no dropping of tracks at truth level (important for the det-truth association because the index of the truth particle is used)
        df_fjparticles_truth = io_truth.load_data(m=self.m) # no dropping of tracks at truth level (important for the det-truth association because the index of the truth particle is used)
        # self.df_events_truth = io_truth.event_df
        self.nEvents_truth = len(df_fjparticles_truth.index)
        self.nTracks_truth = len(io_truth.track_df.index)
        logger.info(f'Truth particles loaded: --- {time.time() - self.start_time:.3f} seconds ---')

        # If jetscape, store also the negative status particles (holes)
        if self.jetscape:
            io_truth_holes = process_io.ProcessIO(input_file=self.input_file, tree_dir=tree_dir,
                                                  track_tree_name='tree_Particle_gen', use_ev_id_ext=False,
                                                  is_jetscape=self.jetscape, holes=True,
                                                  event_plane_range=self.event_plane_range)
            df_fjparticles_truth_holes = io_truth_holes.load_data(m=self.m, reject_tracks_fraction=self.reject_tracks_fraction)
            self.nEvents_truth_holes = len(df_fjparticles_truth_holes.index)
            self.nTracks_truth_holes = len(io_truth_holes.track_df.index)
            print(f'Truth holes loaded: --- {time.time() - self.start_time:.3f} seconds ---')
        
        # ------------------------------------------------------------------------

        # Now merge the two SeriesGroupBy to create a groupby df with [ev_id, run_number, fj_1, fj_2]
        # (Need a structure such that we can iterate event-by-event through both fj_1, fj_2 simultaneously)
        # In the case of jetscape, we merge also the hole collections fj_3, fj_4
        if self.jetscape:
            self.df_fjparticles = pandas.concat([df_fjparticles_det, df_fjparticles_truth, df_fjparticles_det_holes, df_fjparticles_truth_holes], axis=1)
            self.df_fjparticles.columns = ['fj_particles_det', 'fj_particles_truth', 'fj_particles_det_holes', 'fj_particles_truth_holes']
        elif self.ENC_fastsim:
            self.df_fjparticles = pandas.concat([df_fjparticles_det, df_fjparticles_truth], axis=1)
            self.df_fjparticles.columns = ['fj_particles_det', 'ParticleMCIndex', 'fj_particles_truth', 'ParticlePID']
        else:
            df_fjparticles_det.columns = ['fj_particles_det', 'mult_det']
            df_fjparticles_truth.columns = ['fj_particles_truth', 'mult_truth']
            self.df_fjparticles = pandas.concat([df_fjparticles_det, df_fjparticles_truth], axis=1)
            self.df_fjparticles['mult'] = self.df_fjparticles['mult_det'].combine_first(self.df_fjparticles['mult_truth'])
            self.df_fjparticles['mult'] = self.df_fjparticles['mult'].astype('float64')
            self.df_fjparticles.drop(['mult_det', 'mult_truth'], axis = 1, inplace = True)
            self.df_fjparticles = self.df_fjparticles.droplevel(2)

            # # NOTE: In cases where no generated particles are reconstructed,
            # fill NaN values with an empty vectorPJ
            # stolen from https://stackoverflow.com/questions/31567218/replace-nan-with-empty-list-in-a-pandas-dataframe/61944174#61944174
            isnull_det = self.df_fjparticles.fj_particles_det.isnull()
            self.df_fjparticles.loc[isnull_det, 'fj_particles_det'] = pandas.Series([fj.vectorPJ()] * isnull_det.sum()).values
            isnull_truth = self.df_fjparticles.fj_particles_truth.isnull()
            self.df_fjparticles.loc[isnull_truth, 'fj_particles_truth'] = pandas.Series([fj.vectorPJ()] * isnull_truth.sum()).values
        logger.info(f'Det-truth matched: --- {time.time() - self.start_time:.3f} seconds ---')

        # ------------------------------------------------------------------------

        # Setup median subtraction machinery

        if self.do_median_subtraction:
            self.jet_def_medsub = {jetR: fj.JetDefinition(fj.kt_algorithm, jetR) for jetR in self.jetR_list}
            # NOTE: may also not need separate truth subtractor
            self.jet_selector_medsub = {jetR: fj.SelectorAbsEtaMax(0.9 - jetR) & (~fj.SelectorNHardest(2)) & (~fj.SelectorIsPureGhost()) for jetR in self.jetR_list}
            self.median_subtractor = {jetR: fj.JetMedianBackgroundEstimator(self.jet_selector_medsub[jetR], self.jet_def_medsub[jetR], fj.AreaDefinition(fj.active_area_explicit_ghosts)) for jetR in self.jetR_list}
            self.median_subtractor_truth = {jetR: fj.JetMedianBackgroundEstimator(self.jet_selector_medsub[jetR], self.jet_def_medsub[jetR], fj.AreaDefinition(fj.active_area_explicit_ghosts)) for jetR in self.jetR_list}
        # self.jet_selectors_det = {jetR: fj.SelectorPtMin(5.0) & fj.SelectorAbsRapMax(0.9 - jetR) for jetR in self.jetR_list}
        # self.jet_selector_truth_matched = fj.SelectorPtMin(5.0) & fj.SelectorAbsRapMax(0.9)
        # self.Cjet_selectors = {jetR: fj.SelectorAbsRapMax(0.9 - jetR) & (~fj.SelectorIsPureGhost()) for jetR in self.jetR_list}
        self.jet_selectors_det = {jetR: fj.SelectorPtMin(5.0) & fj.SelectorAbsEtaMax(0.9 - jetR) for jetR in self.jetR_list}
        self.jet_selector_truth_matched = fj.SelectorPtMin(5.0) & fj.SelectorAbsEtaMax(0.9)
        self.Cjet_selectors = {jetR: fj.SelectorAbsEtaMax(0.9 - jetR) & (~fj.SelectorIsPureGhost()) for jetR in self.jetR_list}
        self.jet_defs = {jetR: fj.JetDefinition(fj.antikt_algorithm, jetR) for jetR in self.jetR_list}

        # -----------------------------------------------------------------------
        
        # Set up the Pb-Pb (p-Pb) embedding object
        if not self.is_pp and not self.thermal_model:
            self.process_io_emb = process_io_emb.ProcessIO_Emb(self.emb_file_list, track_tree_name='tree_Particle', m=self.m, is_pp=(self.is_pp or self.is_pA))

        # Initialize histograms
        if not self.dry_run:
            self.initialize_output_objects()

        # Create constituent subtractor, if configured
        if self.do_constituent_subtraction and not self.is_pp and not self.is_pA:
            self.constituent_subtractor = [CEventSubtractor(max_distance=R_max, alpha=self.alpha, max_eta=self.max_eta, bge_rho_grid_size=self.bge_rho_grid_size, max_pt_correct=self.max_pt_correct, ghost_area=self.ghost_area, distance_type=fjcontrib.ConstituentSubtractor.deltaR) for R_max in self.max_distance]

        if self.do_pt_dep_track_rej:
            self.reject_tracks_pt()
        # Find jets and fill histograms
        logger.info('Analyzing events...')
        self.analyze_events()

        # Plot histograms
        logger.info('Saving histograms...')
        process_base.ProcessBase.save_output_objects(self)

        logger.info(f'Analysis complete: --- {time.time() - self.start_time:.3f} seconds ---')
    
    def reject_tracks_pt(self):
        rng = np.random.default_rng(seed = self.rng_seed)
        with np.load(self.do_pt_dep_track_rej_file) as file:
            edges = file['ptedges']
            unc = file['unc']
        tot = np.sum([len(parts) for parts in self.df_fjparticles['fj_particles_det']])

        for (run_no, ev_id), parts in zip(self.df_fjparticles.index, self.df_fjparticles['fj_particles_det']):
            new_parts = fj.vectorPJ()
            pts = [part.pt() for part in parts]
            idxs = np.searchsorted(edges, pts, side='left') - 1
            probs = [unc[idx] for idx in idxs]
            results = rng.binomial(n = 1, p = probs) == 1
            [new_parts.push_back(part) for part, res in zip(parts, results, strict = True) if res]
            self.df_fjparticles.at[(run_no, ev_id), 'fj_particles_det'] = new_parts
        tot_new = np.sum([len(parts) for parts in self.df_fjparticles['fj_particles_det']])
        logger.info(f"Out of {tot} tracks, {tot - tot_new} tracks were rejected, overall rejection rate {(tot - tot_new) / tot * 100:.4f}%.")
        logger.info(f"pT-dependent rejection complete:  --- {time.time() - self.start_time:.3f} seconds ---")

    #---------------------------------------------------------------
    # Initialize histograms
    #---------------------------------------------------------------
    def initialize_output_objects(self):
        self.hNevents = ROOT.TH1F('hNevents', 'hNevents', 2, -0.5, 1.5)
        self.hNevents.Fill(1, self.nEvents_det)
        
        for observable in self.observable_list:
            if "track_pairdist" in observable:
                xaxis = "PairKt"
                for data_class in ['', "_Truth"]:
                    # for yaxis in ['phi', 'phistar', 'eta']:
                    #     name = f'h_{observable}_{yaxis}_{xaxis}{data_class}'
                    #     h = ROOT.TH2D(name, name, self.kT_nbins, self.kT_bins, self.pairdist_nbins, self.pairdist_bins)
                    #     h.GetXaxis().SetTitle('pair k_{T}')
                    #     h.GetYaxis().SetTitle(f'{yaxis}')
                    #     setattr(self, name, h)
                    
                    # name = f'h_{observable}_RL_{xaxis}{data_class}'
                    # h = ROOT.TH2D(name, name, self.kT_nbins, self.kT_bins, self.RL_nbins, self.RL_bins)
                    # h.GetXaxis().SetTitle('pair k_{T}')
                    # h.GetYaxis().SetTitle('RL')
                    # setattr(self, name, h)

                    name = f'h_{observable}_phistar_eta_{xaxis}{data_class}'
                    h = ROOT.TH3F(name, name, self.kT_nbins, self.kT_bins, self.pairdist_nbins, self.pairdist_bins, self.pairdist_nbins, self.pairdist_bins)
                    h.GetXaxis().SetTitle('pair k_{T}')
                    h.GetYaxis().SetTitle('phistar')
                    h.GetZaxis().SetTitle('#Delta #eta')
                    setattr(self, name, h)
        
        # self.hTrackEtaPhi = ROOT.TH2F('hTrackEtaPhi', 'hTrackEtaPhi', 200, -1., 1., 628, 0., 6.28)
        # self.hTrackPt = ROOT.TH1F('hTrackPt', 'hTrackPt', 300, 0., 300.)
        
        # if not self.is_pp:
        #     self.hRho =  ROOT.TH1F('hRho', 'hRho', 1000, 0., 1000.)
            
        # if not self.skip_deltapt_RC_histograms:
        #     name = 'hN_MeanPt'
        #     h = ROOT.TH2F(name, name, 200, 0, 5000, 200, 0., 2.)
        #     setattr(self, name, h)

    #---------------------------------------------------------------
    # Initialize histograms
    #---------------------------------------------------------------
    def initialize_output_objects_R(self, jetR):
    
        # Call user-specific initialization
        self.initialize_user_output_objects_R(jetR)
        
        # Base histograms
        # if self.is_pp:
        
        #     name = 'hJES_R{}'.format(jetR)
        #     h = ROOT.TH2F(name, name, 300, 0, 300, 200, -1., 1.)
        #     setattr(self, name, h)
    
        #     name = 'hDeltaR_All_R{}'.format(jetR)
        #     h = ROOT.TH2F(name, name, 300, 0, 300, 100, 0., 2.)
        #     setattr(self, name, h)

        #     name = 'hJetPtCorrRes_R{}'.format(jetR)
        #     h = ROOT.TH2F(name, name, 300, 0, 300, 400, -20., 20.)
        #     setattr(self, name, h)

        # elif self.is_pA:

        #     for R_max in self.max_distance:
            
        #         name = 'hJES_R{}_Rmax{}'.format(jetR, R_max)
        #         h = ROOT.TH2F(name, name, 300, 0, 300, 200, -1., 1.)
        #         setattr(self, name, h)
            
        #         name = 'hDeltaPt_emb_R{}_Rmax{}'.format(jetR, R_max)
        #         h = ROOT.TH2F(name, name, 300, 0, 300, 400, -200., 200.)
        #         setattr(self, name, h)
    
        #     name = 'hJetPtCorrRes_R{}'.format(jetR)
        #     h = ROOT.TH2F(name, name, 300, 0, 300, 400, -20., 20.)
        #     setattr(self, name, h)
            
        #     name = 'hDeltaR_All_R{}'.format(jetR)
        #     h = ROOT.TH2F(name, name, 300, 0, 300, 100, 0., 2.)
        #     setattr(self, name, h)

        #     name = 'h_matched_jetconstituents_R{}'.format(jetR)
        #     h = ROOT.TH2F(name, name, 200, 0, 200, 30, 0., 30.)
        #     setattr(self, name, h)

        #     name = 'h_matched_jetconstituent_pT_R{}'.format(jetR)
        #     h = ROOT.TH2F(name, name, 200, 0, 200, 100, 0., 100.)
        #     setattr(self, name, h)

        #     name = 'hMedRho_R{}'.format(jetR)
        #     h = ROOT.TH1F(name, name, 200, 0, 20.)
        #     setattr(self, name, h)

        #     name = 'hMedRhoCArea_R{}'.format(jetR)
        #     h = ROOT.TH1F(name, name, 200, 0, 20.)
        #     setattr(self, name, h)

        #     name = 'hCArea_R{}'.format(jetR)
        #     h = ROOT.TH1F(name, name, 100, 0, 1.)
        #     setattr(self, name, h)
            
        #     name = 'hJetArea_R{}'.format(jetR)
        #     h = ROOT.TH1F(name, name, 100, 0, 5.)
        #     setattr(self, name, h)

        #     name = 'hPerpRho_R{}'.format(jetR)
        #     h = ROOT.TH1F(name, name, 200, 0, 20.)
        #     setattr(self, name, h)

        #     name = 'hSigJetpt_MedRho_R{}'.format(jetR)
        #     h = ROOT.TH2F(name, name, 100, 0, 100., 200, 0, 20.)
        #     setattr(self, name, h)

        #     name = 'h_matched_SigJetpt_MedRho_R{}'.format(jetR)
        #     h = ROOT.TH2F(name, name, 100, 0, 100., 200, 0, 20.)
        #     setattr(self, name, h)

        #     name = 'hPerpConeMult_Rho_R{}'.format(jetR)
        #     h = ROOT.TH2F(name, name, 20, 0, 20., 200, 0, 20.)
        #     setattr(self, name, h)

        #     for ptlo, pthi in [(20,40),(40,60),(60,80)]:
        #         name = 'hPerpConeMult_JetMult_{}{}_R{}'.format(ptlo,pthi,jetR)
        #         h = ROOT.TH2F(name, name, 20, 0, 20., 100, 0, 100.)
        #         setattr(self, name, h)

        #         name = 'hPerpConeMult_JetMult_150_{}{}_R{}'.format(ptlo,pthi,jetR)
        #         h = ROOT.TH2F(name, name, 20, 0, 20., 100, 0, 100.)
        #         setattr(self, name, h)

        #         name = 'hPerpConeMult_JetMult_1GeV_{}{}_R{}'.format(ptlo,pthi,jetR)
        #         h = ROOT.TH2F(name, name, 20, 0, 20., 100, 0, 100.)
        #         setattr(self, name, h)

        #         name = 'hPerpConeMult_JetMult_MC_{}{}_R{}'.format(ptlo,pthi,jetR)
        #         h = ROOT.TH2F(name, name, 20, 0, 20., 100, 0, 100.)
        #         setattr(self, name, h)

        #         name = 'hPerpConeMult_JetMult_emb_{}{}_R{}'.format(ptlo,pthi,jetR)
        #         h = ROOT.TH2F(name, name, 20, 0, 20., 100, 0, 100.)
        #         setattr(self, name, h)

        #     name = 'hDeltaR_ppdet_pptrue_R{}'.format(jetR)
        #     h = ROOT.TH2F(name, name, 300, 0, 300, 100, 0., 2.)
        #     setattr(self, name, h)
            
        #     name = 'hDeltaR_combined_ppdet_R{}'.format(jetR)
        #     h = ROOT.TH2F(name, name, 300, 0, 300, 100, 0., 2.)
        #     setattr(self, name, h)

        #     name = 'hPerpConepT_Raw_R{}'.format(jetR)
        #     h = ROOT.TH1F(name, name, 400, 0., 20.)
        #     setattr(self, name, h)

        #     name = 'hPerpConepT_MedCorrected_R{}'.format(jetR)
        #     h = ROOT.TH2F(name, name, 200, 0, 200, 400, -5., 15.)
        #     setattr(self, name, h)

        #     name = 'hPerpConepT_CSCorrected_R{}'.format(jetR)
        #     h = ROOT.TH1F(name, name, 400, -5., 15.)
        #     setattr(self, name, h)

        #     name = 'hPerpConepTRes_MedCorrected_R{}'.format(jetR)
        #     h = ROOT.TH1F(name, name, 200, -1., 1.)
        #     setattr(self, name, h)

        #     name = 'hPerpConepTComp_MedCorrected_R{}'.format(jetR)
        #     h = ROOT.TH2F(name, name, 200, -5., 15., 200, 0, 20.)
        #     setattr(self, name, h)

        #     name = 'hPerpConepTComp_CSCorrected_R{}'.format(jetR)
        #     h = ROOT.TH2F(name, name, 200, -5., 15., 200, 0, 20.)
        #     setattr(self, name, h)

        #     name = 'hSigJetpt_PerpRho_R{}'.format(jetR)
        #     h = ROOT.TH2F(name, name, 100, 0, 100., 200, 0, 20.)
        #     setattr(self, name, h)
                
        # else:
        
        #     for R_max in self.max_distance:
            
        #         name = 'hJES_R{}_Rmax{}'.format(jetR, R_max)
        #         h = ROOT.TH2F(name, name, 300, 0, 300, 200, -1., 1.)
        #         setattr(self, name, h)
            
        #         name = 'hDeltaPt_emb_R{}_Rmax{}'.format(jetR, R_max)
        #         h = ROOT.TH2F(name, name, 300, 0, 300, 400, -200., 200.)
        #         setattr(self, name, h)
                
        #         if not self.skip_deltapt_RC_histograms:
        #             name = 'hDeltaPt_RC_beforeCS_R{}_Rmax{}'.format(jetR, R_max)
        #             h = ROOT.TH1F(name, name, 400, -200., 200.)
        #             setattr(self, name, h)
                    
        #             name = 'hDeltaPt_RC_afterCS_R{}_Rmax{}'.format(jetR, R_max)
        #             h = ROOT.TH1F(name, name, 400, -200., 200.)
        #             setattr(self, name, h)
    
        #         name = 'hDeltaR_ppdet_pptrue_R{}_Rmax{}'.format(jetR, R_max)
        #         h = ROOT.TH2F(name, name, 300, 0, 300, 100, 0., 2.)
        #         setattr(self, name, h)
                
        #         name = 'hDeltaR_combined_ppdet_R{}_Rmax{}'.format(jetR, R_max)
        #         h = ROOT.TH2F(name, name, 300, 0, 300, 100, 0., 2.)
        #         setattr(self, name, h)
                        
        # name = 'hZ_Truth_R{}'.format(jetR)
        # h = ROOT.TH2F(name, name, 300, 0, 300, 100, 0., 1.)
        # setattr(self, name, h)
        
        # name = 'hZ_Det_R{}'.format(jetR)
        # h = ROOT.TH2F(name, name, 300, 0, 300, 100, 0., 1.)
        # setattr(self, name, h)

        # name = 'hMult_det'
        # h = ROOT.TH1F(name, name, 500, 0, 500)
        # setattr(self, name, h)

        # name = 'hMult_truth'
        # h = ROOT.TH1F(name, name, 500, 0, 500)
        # setattr(self, name, h)

    #---------------------------------------------------------------
    # Main function to loop through and analyze events
    #---------------------------------------------------------------
    def analyze_events(self):
        # Fill track histograms
        # if not self.dry_run:
        #     [self.fill_track_histograms(fj_particles_det) for fj_particles_det in self.df_fjparticles['fj_particles_det']]

        fj.ClusterSequence.print_banner()

        self.event_number = 0

        if not self.dry_run:
            for jetR in self.jetR_list:
                self.initialize_output_objects_R(jetR)

        if self.jetscape:
            [self.analyze_event(fj_particles_det, fj_particles_truth, fj_particles_det_holes, fj_particles_truth_holes) for fj_particles_det, fj_particles_truth, fj_particles_det_holes, fj_particles_truth_holes in zip(self.df_fjparticles['fj_particles_det'], self.df_fjparticles['fj_particles_truth'], self.df_fjparticles['fj_particles_det_holes'], self.df_fjparticles['fj_particles_truth_holes'])]
        elif self.ENC_fastsim:
            [self.analyze_event(fj_particles_det=fj_particles_det, fj_particles_truth=fj_particles_truth, particles_mcid_det=particles_mcid_det, particles_pid_truth=particles_pid_truth) for fj_particles_det, fj_particles_truth, particles_mcid_det, particles_pid_truth in zip(self.df_fjparticles['fj_particles_det'], self.df_fjparticles['fj_particles_truth'], self.df_fjparticles['ParticleMCIndex'], self.df_fjparticles['ParticlePID'])]
        else:
            [self.analyze_event(fj_particles_det, fj_particles_truth, mult = mult) for fj_particles_det, fj_particles_truth, mult in zip(self.df_fjparticles['fj_particles_det'], self.df_fjparticles['fj_particles_truth'], self.df_fjparticles['mult'])]

        logger.info('Saving THn...')
        process_base.ProcessBase.save_thn_th3_objects(self)
        
    #---------------------------------------------------------------
    # Fill track histograms.
    #---------------------------------------------------------------
    def fill_track_histograms(self, fj_particles_det):
        # Check that the entries exist appropriately
        # (need to check how this can happen -- but it is only a tiny fraction of events)
        if len(fj_particles_det) == 0:
            logger.warning("While filling track histograms, event has no detector-level particles; skipping event.")
            return

        for track in fj_particles_det:
            self.hTrackEtaPhi.Fill(track.eta(), track.phi())
            self.hTrackPt.Fill(track.pt())

    def is_mc_match(self, id_det, id_truth):
        if self.strict_mc_matching:
            return id_det == id_truth
        else:
            return abs(id_det) == id_truth
    #---------------------------------------------------------------
    # Analyze jets of a given event.
    # fj_particles is the list of fastjet pseudojets for a single fixed event.
    #---------------------------------------------------------------
    def analyze_event(self, fj_particles_det, fj_particles_truth, fj_particles_det_holes=None, fj_particles_truth_holes=None,
                      particles_mcid_det=None, particles_pid_truth=None, mult = None):
        self.event_number += 1
        if self.event_number > self.event_number_max:
            return

        # match tracks, jet clustering, pairs, pair matching, table write out
        # # single track efficiency plots here
        # self.analyze_matched_pairs(fj_particles_det, fj_particles_truth, jetR=0.4)

        if self.ENC_fastsim:
            # make charge array from pid info, needed for pair efficiency determination
            particles_charge_truth = np.array([])
            for pid in particles_pid_truth:
                # charged hadrons
                if abs(pid)==211 or abs(pid)==321 or abs(pid)==2212 or abs(pid)==3222:
                    if pid>0:
                        particles_charge_truth = np.append(particles_charge_truth, 1)
                    else:
                        particles_charge_truth = np.append(particles_charge_truth, -1)
                # electrons and muons
                elif abs(pid)==11 or abs(pid)==13 or abs(pid)==3112 or abs(pid)==3312 or abs(pid)==3334:
                    if pid>0:
                        particles_charge_truth = np.append(particles_charge_truth, -1)
                    else:
                        particles_charge_truth = np.append(particles_charge_truth, 1)
                # long lived weak decay particles (<2% of the total number of charged particles)
                # for now mark as charge 0 and later NOT applying pair efficiency for 0-charged or 0-0 pairs
                # NB: this can be avoided by decaying these paritcles within the generation step
                else:
                    particles_charge_truth = np.append(particles_charge_truth, 0)

        if not self.ENC_fastsim:
            if len(fj_particles_truth) == 0:
                logger.warning(f"Event {self.event_number} has {len(fj_particles_det)} det tracks and {len(fj_particles_truth)} truth tracks.")
            if len(fj_particles_det) == 0:
                logger.warning(f"Event {self.event_number} has {len(fj_particles_det)} det tracks and {len(fj_particles_truth)} truth tracks.")
        else:
            if not isinstance(fj_particles_truth, fj.vectorPJ):
                fj_particles_truth = fj.vectorPJ()
                logger.warning(f"Event {self.event_number} has {len(fj_particles_det)} det tracks and {len(fj_particles_truth)} truth tracks.")

        if self.ENC_fastsim:
            for index, mcid in enumerate(particles_mcid_det):
                if fj_particles_det[index].has_user_info():
                    ecorr_user_info = fj_particles_det[index].python_info()
                else:
                    ecorr_user_info = jet_info.JetInfo()
                if mcid>=0 and mcid<len(fj_particles_truth):
                    ecorr_user_info.particle_truth = fj_particles_truth[int(mcid)]
                    ecorr_user_info.charge = particles_charge_truth[int(mcid)]
                else:
                    logger.warning("Invalid associated MC Index, filling default values (particle_truth = None, charge = 1000)")
                fj_particles_det[index].set_python_info(ecorr_user_info)
                # fj_particles_det[index].set_user_index(int(mcid))

            for index in range( len(fj_particles_truth) ):
                if fj_particles_truth[index].has_user_info():
                    ecorr_user_info = fj_particles_truth[index].python_info()
                else:
                    ecorr_user_info = jet_info.JetInfo()
                ecorr_user_info.particle_truth = fj_particles_truth[index]
                ecorr_user_info.charge = particles_charge_truth[index]
                fj_particles_truth[index].set_python_info(ecorr_user_info)
                # fj_particles_truth[index].set_user_index(int(index))
        elif self.do_track_mc_matching:
            # logger.debug("Starting MC ID matching.")
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
            # logger.debug("MC ID matching completed.")

        if self.jetscape:
            if len(fj_particles_det_holes) == 0 or len(fj_particles_truth_holes) == 0:
                logger.warning('No det or truth holes -- skipping event')
                return

        if len(fj_particles_truth) > 1:
            if np.abs(fj_particles_truth[0].pt() - fj_particles_truth[1].pt()) <  1e-10:
                logger.warning('Duplicate particles may be present'
                              f'{[p.user_index() for p in fj_particles_truth]}'
                              f'{[p.pt() for p in fj_particles_truth]}')

        #HACK: pretty sure the old bug is here still
        # but how to make sure the multiplicities match up when stitching det and truth?
        # self.mult_det = self.df_events_det['V0Amult'].values[self.event_number-1]
        # self.mult_truth = self.df_events_truth['V0Amult'].values[self.event_number-1]
        #HACK: placeholder for now
        self.mult_det = 0
        self.mult_truth = 0
        
        self.fill_efficiency_histograms(fj_particles_det, fj_particles_truth)
        
        # getattr(self, 'hMult_det').Fill(self.mult_det)
        # getattr(self, 'hMult_truth').Fill(self.mult_truth)

        # If Pb-Pb (or p-Pb), construct embedded event (do this once, for all jetR)
        if not self.is_pp:
            # If thermal model, generate a thermal event and add it to the det-level particle list
            if self.thermal_model:
                fj_particles_combined_beforeCS = self.thermal_generator.load_event()
                # Form the combined det-level event
                # The pp-det tracks are each stored with a unique user_index >= 0
                #   (same index in fj_particles_combined and fj_particles_det -- which will be used in prong-matching)
                # The thermal tracks are each stored with a unique user_index < 0
                [fj_particles_combined_beforeCS.push_back(p) for p in fj_particles_det]

        #     # Main case: Get Pb-Pb event and embed it into the det-level particle list
            else:
                fj_particles_combined_beforeCS = self.process_io_emb.load_event()
                
                if self.do_emb_rotation:
                    leading_emb_particle = fj.sorted_by_pt(fj_particles_combined_beforeCS)[0]

                    jetR_for_rot = self.jetR_list[0]
                    jet_def_for_rot = fj.JetDefinition(fj.antikt_algorithm, jetR_for_rot)
                    jet_selector_det_for_rot = fj.SelectorPtMin(5.0) & fj.SelectorAbsRapMax(0.9 - jetR_for_rot)
                    cs_det_for_rot = fj.ClusterSequence(fj_particles_det, jet_def_for_rot)
                    jets_det_pp_for_rot = fj.sorted_by_pt(cs_det_for_rot.inclusive_jets())
                    if jet_selector_det_for_rot(jets_det_pp_for_rot):
                        leading_pp_jet =  jet_selector_det_for_rot(jets_det_pp_for_rot)[0]
                        rotation_angle = leading_emb_particle.delta_phi_to(leading_pp_jet) - np.pi/2
                        for particle in fj_particles_combined_beforeCS:
                            particle.reset_momentum_PtYPhiM(particle.pt(),particle.rap(),(particle.phi()+rotation_angle)%(2*np.pi),particle.m())
                            particle.set_user_index(int(-1e6))
                            # if particle.pt() == leading_emb_particle.pt():
                                # print(particle.delta_phi_to(leading_pp_jet))
                
                # Form the combined det-level event
                # The pp-det tracks are each stored with a unique user_index >= 0
                #   (same index in fj_particles_combined and fj_particles_det -- which will be used in prong-matching)
                # The Pb-Pb tracks are each stored with a unique user_index < 0
                [fj_particles_combined_beforeCS.push_back(p) for p in fj_particles_det]

            # For Pb-Pb, perform constituent subtraction for each R_max
            if not self.is_pA:
                fj_particles_combined = [self.constituent_subtractor[i].process_event(fj_particles_combined_beforeCS) for i, R_max in enumerate(self.max_distance)]

        if self.dry_run:
            return

        # Loop through jetR, and process event for each R
        for jetR in self.jetR_list:
        
            # Keep track of whether to fill R-independent histograms
            # self.fill_R_indep_hists = (jetR == self.jetR_list[0])

            # Set jet definition and a jet selector
            jet_def = self.jet_defs[jetR]
            jet_selector_det = self.jet_selectors_det[jetR]

            if self.do_median_subtraction:
                csa_medsub = fj.ClusterSequenceArea(fj_particles_combined_beforeCS, self.jet_def_medsub[jetR], fj.AreaDefinition(fj.active_area_explicit_ghosts))
                self.median_subtractor[jetR].set_cluster_sequence(csa_medsub)
                rho = self.median_subtractor[jetR].rho()
                # getattr(self, 'hMedRho_R{}'.format(jetR)).Fill(rho)

                Cjet_selector = self.Cjet_selectors[jetR]
                # medsub_selected_jets = fj.sorted_by_pt(Cjet_selector(csa_medsub.inclusive_jets()))
                medsub_selected_jets = Cjet_selector(csa_medsub.inclusive_jets())
                # n_activejets = len(medsub_selected_jets)
                C_area = np.sum([jet.area() for jet in medsub_selected_jets]) / (2 * np.pi * 0.9 * 2)
                # C_area = 0
                # for jet in medsub_selected_jets:
                #     C_area += jet.area()
                # C_area = C_area / (2*np.pi*1.8)
                # getattr(self, 'hCArea_R{}'.format(jetR)).Fill(C_area)
                # getattr(self, 'hMedRhoCArea_R{}'.format(jetR)).Fill(rho*C_area)

                csa_medsub_truth = fj.ClusterSequenceArea(fj_particles_truth, self.jet_def_medsub[jetR], fj.AreaDefinition(fj.active_area_explicit_ghosts))
                self.median_subtractor_truth[jetR].set_cluster_sequence(csa_medsub_truth)
                rho_truth = self.median_subtractor_truth[jetR].rho()
                # medsub_selected_jets_truth = fj.sorted_by_pt(Cjet_selector(csa_medsub_truth.inclusive_jets()))
                medsub_selected_jets_truth = Cjet_selector(csa_medsub_truth.inclusive_jets())
                C_area_truth = np.sum([jet.area() for jet in medsub_selected_jets_truth]) / (2 * np.pi * 2 * 0.9)
                # C_area_truth = 0
                # for jet in medsub_selected_jets_truth:
                #     C_area_truth += jet.area()
                # C_area_truth = C_area_truth / (2*np.pi*1.8)

            # Analyze
            if self.is_pp:
                # Find pp det and truth jets
                if self.ENC_fastsim:
                    # FIX ME: should treat long lived charged particle differently (check how the existing fast herwig and pythia handles it)
                    fj_particles_det_ch = fj.vectorPJ()
                    for part in fj_particles_det:
                        if part.python_info().charge!=0:
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
                    # FIX ME: should treat long lived charged particle differently (check how the existing fast herwig and pythia handles it)
                    fj_particles_truth_ch = fj.vectorPJ()
                    for part in fj_particles_truth:
                        if part.python_info().charge!=0:
                            fj_particles_truth_ch.append(part)
                    cs_truth = fj.ClusterSequence(fj_particles_truth_ch, jet_def)
                else:
                    cs_truth = fj.ClusterSequence(fj_particles_truth, jet_def)

                jets_truth = fj.sorted_by_pt(cs_truth.inclusive_jets())
                # make sure the user info (on the jet side) for jets are all empty right after the jet-clustering  
                for jet in jets_truth:
                    if jet.has_user_info():
                        jet.python_info().clear_jet_info()
                jets_truth_selected = jet_selector_det(jets_truth)
                jets_truth_selected_matched = self.jet_selector_truth_matched(jets_truth)
            
                self.analyze_jets(jets_det_pp_selected, jets_truth_selected, jets_truth_selected_matched, jetR)

            elif self.is_pA:
                # print('Total number of combined particles: {}'.format(len([p.pt() for p in fj_particles_combined_beforeCS])))
                for i, R_max in enumerate(self.max_distance):
                    # print('After constituent subtraction: {}'.format(len([p.pt() for p in fj_particles_combined[i]])))
                    self.fill_Rmax_indep_hists = (i == 0)
                    # rhocs = self.constituent_subtractor[i].bge_rho.rho() 

                    # Do jet finding (re-do each time, to make sure matching info gets reset)
                    
                    #HACK: why no area defined?
                    cs_det = fj.ClusterSequence(fj_particles_det, jet_def)
                    jets_det = fj.sorted_by_pt(cs_det.inclusive_jets())
                    jets_det_selected = jet_selector_det(jets_det)

                    #HACK: why i s it active area explicit ghosts not voronoi area?
                    # cs_truth = fj.ClusterSequenceArea(fj_particles_truth, jet_def, fj.AreaDefinition(fj.active_area_explicit_ghosts))
                    # cs_truth = fj.ClusterSequenceArea(fj_particles_truth, jet_def, fj.AreaDefinition(fj.active_area))
                    cs_truth = fj.ClusterSequenceArea(fj_particles_truth, jet_def, fj.AreaDefinition(fj.VoronoiAreaSpec()))
                    jets_truth = fj.sorted_by_pt(cs_truth.inclusive_jets())
                    jets_truth_selected = jet_selector_det(jets_truth)
                    jets_truth_selected_matched = self.jet_selector_truth_matched(jets_truth)

                    # cs_combined_cs = fj.ClusterSequence(fj_particles_combined[i], jet_def)
                    # jets_combined_cs = fj.sorted_by_pt(cs_combined_cs.inclusive_jets())
                    # jets_combined_selected_cs = jet_selector_det(jets_combined_cs)

                    # cs_combined = fj.ClusterSequenceArea(fj_particles_combined_beforeCS, jet_def, fj.AreaDefinition(fj.active_area_explicit_ghosts))
                    # cs_combined = fj.ClusterSequenceArea(fj_particles_combined_beforeCS, jet_def, fj.AreaDefinition(fj.active_area))
                    cs_combined = fj.ClusterSequenceArea(fj_particles_combined_beforeCS, jet_def, fj.AreaDefinition(fj.VoronoiAreaSpec()))
                    jets_combined = fj.sorted_by_pt(cs_combined.inclusive_jets())
                    jets_combined_selected = jet_selector_det(jets_combined)

                    if self.do_median_subtraction:
                        # rho = rho * C_area
                        # rho_truth = rho_truth * C_area_truth

                        jets_combined_reselected = []
                        if rho > 0:
                            for jet in jets_combined_selected:
                                if jet.pt()-rho*jet.area()*C_area > 5:
                                    # getattr(self, 'hSigJetpt_MedRho_R{}'.format(jetR)).Fill(jet.pt()-rho*jet.area(),rho)
                                    jets_combined_reselected.append(jet)
                        # jets_det_reselected = [jet for jet in jets_combined_selected if jet.pt() - rho * C_area * jet.area() > 5]
                        # jets_det_reselected = [jet for jet in jets_det_selected if jet.pt() - rho * C_area * jet.area() > 5]
                        # jets_truth_reselected = [jet for jet in jets_truth_selected if jet.pt() - rho_truth * C_area_truth * jet.area() > 5]

                        if self.do_perpendicular_cone:
                            jets_combined_reselected_wpcone = [self.attach_perp_cones(fj_particles_det, jet, coneR = jetR) for jet in jets_combined_reselected]
                            # jets_combined_reselected_wpcone = [self.attach_perp_cones(fj_particles_det, jet, coneR = jetR) for jet in jets_combined_selected]
                            jets_truth_selected_wpcone = [self.attach_perp_cones(fj_particles_truth, jet, coneR = jetR) for jet in jets_truth_selected]
                            self.analyze_jets(jets_combined_reselected_wpcone, jets_truth_selected_wpcone, jets_truth_selected_matched, jetR,
                                              jets_det_pp_selected = jets_det_selected, R_max = R_max, fj_particles_det_holes = fj_particles_det_holes,
                                              fj_particles_truth_holes = fj_particles_truth_holes, rho_bge = (rho * C_area,rho_truth * C_area_truth),
                                              fj_particles_det_cones = fj_particles_combined_beforeCS, fj_particles_truth_cones = fj_particles_truth)
                        else:
                            self.analyze_jets(jets_combined_reselected, jets_truth_selected, jets_truth_selected_matched, jetR,
                                              jets_det_pp_selected = jets_det_selected, R_max = R_max, fj_particles_det_holes = fj_particles_det_holes,
                                              fj_particles_truth_holes = fj_particles_truth_holes, rho_bge = (rho * C_area,rho_truth * C_area_truth),
                                              fj_particles_det_cones = fj_particles_combined_beforeCS, fj_particles_truth_cones = fj_particles_truth)
                        
                        # self.analyze_perpendicular_cone(jets_combined_reselected, jets_det_selected, jets_truth_selected, jets_truth_selected_matched, jetR, fj_particles_combined_beforeCS, rho*C_area)

                    else:
                        self.analyze_jets(jets_combined_selected, jets_truth_selected, jets_truth_selected_matched, jetR,
                                          jets_det_pp_selected = jets_det_pp_selected, R_max = R_max,
                                          fj_particles_det_holes = fj_particles_det_holes,
                                          fj_particles_truth_holes = fj_particles_truth_holes, rho_bge = 0, fj_particles_det_cones = fj_particles_combined_beforeCS, fj_particles_truth_cones = fj_particles_truth)

                        self.analyze_perpendicular_cone(jets_combined_selected, jets_det_pp_selected, jets_truth_selected, jets_truth_selected_matched, jetR, fj_particles_combined_beforeCS)

            else:
                for i, R_max in enumerate(self.max_distance):
                    if self.debug_level > 1:
                        print('')
                        print('R_max: {}'.format(R_max))
                        print('Total number of combined particles: {}'.format(len([p.pt() for p in fj_particles_combined_beforeCS])))
                        print('After constituent subtraction {}: {}'.format(i, len([p.pt() for p in fj_particles_combined[i]])))
                        
                    # Keep track of whether to fill R_max-independent histograms
                    self.fill_Rmax_indep_hists = (i == 0)
                    
                    # Perform constituent subtraction on det-level, if applicable
                    self.fill_background_histograms(fj_particles_combined_beforeCS, fj_particles_combined[i], jetR, i)
            
                    # Do jet finding (re-do each time, to make sure matching info gets reset)
                    cs_det = fj.ClusterSequence(fj_particles_det, jet_def)
                    jets_det_pp = fj.sorted_by_pt(cs_det.inclusive_jets())
                    jets_det_pp_selected = jet_selector_det(jets_det_pp)
                    
                    cs_truth = fj.ClusterSequence(fj_particles_truth, jet_def)
                    jets_truth = fj.sorted_by_pt(cs_truth.inclusive_jets())
                    jets_truth_selected = jet_selector_det(jets_truth)
                    jets_truth_selected_matched = self.jet_selector_truth_matched(jets_truth)
                    
                    cs_combined = fj.ClusterSequence(fj_particles_combined[i], jet_def)
                    jets_combined = fj.sorted_by_pt(cs_combined.inclusive_jets())
                    jets_combined_selected = jet_selector_det(jets_combined)

                    self.analyze_jets(jets_combined_selected, jets_truth_selected, jets_truth_selected_matched, jetR,
                                                        jets_det_pp_selected = jets_det_pp_selected, R_max = R_max,
                                                        fj_particles_det_holes = fj_particles_det_holes,
                                                        fj_particles_truth_holes = fj_particles_truth_holes)

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

    #---------------------------------------------------------------
    # Analyze jets of a given event.
    #---------------------------------------------------------------
    def analyze_jets(self, jets_det_selected, jets_truth_selected, jets_truth_selected_matched, jetR,
                     jets_det_pp_selected = None, R_max = None, fj_particles_det_holes = None,
                     fj_particles_truth_holes = None, rho_bge = 0, fj_particles_det_cones = None,
                     fj_particles_truth_cones = None):
        if isinstance(rho_bge, tuple):
            rho_bge_det = rho_bge[0]
            rho_bge_truth = rho_bge[1]
        else:
            rho_bge_det = rho_bge
            rho_bge_truth = rho_bge

        # Fill det-level jet histograms (before matching)
        for jet_det in jets_det_selected:
            # Check additional acceptance criteria
            # skip event if not satisfied -- since first jet in event is highest pt
            if not self.utils.is_det_jet_accepted(jet_det):
                logger.warning('Jet rejected due to jet acceptance')
                self.hNevents.Fill(0)
            self.fill_det_before_matching(jet_det, jetR, rho_bge_det)
    
        # Fill truth-level jet histograms (before matching)
        for jet_truth in jets_truth_selected:
            self.fill_truth_before_matching(jet_truth, jetR, rho_bge_truth)
            # if self.is_pp or self.fill_Rmax_indep_hists:
    
        # Loop through jets and set jet matching candidates for each jet in user_info
        # if self.is_pp:
        #         [[self.set_matching_candidates(jet_det, jet_truth, jetR, 'hDeltaR_All_R{}'.format(jetR)) for jet_truth in jets_truth_selected_matched] for jet_det in jets_det_selected]
        # elif self.is_pA:
        #     # First fill the combined-to-pp matches, then the pp-to-pp matches
        #     # [[self.set_matching_candidates(jet_det_combined, jet_det_pp, jetR, 'hDeltaR_combined_ppdet_R{}', fill_jet1_matches_only=True) for jet_det_pp in jets_det_pp_selected] for jet_det_combined in jets_det_selected]
        #     # [[self.set_matching_candidates(jet_det_pp, jet_truth, jetR, 'hDeltaR_ppdet_pptrue_R{}') for jet_truth in jets_truth_selected_matched] for jet_det_pp in jets_det_pp_selected]
        #     [[self.set_matching_candidates(jet_det_combined, jet_det_pp, jetR, '', fill_jet1_matches_only=True) for jet_det_pp in jets_det_pp_selected] for jet_det_combined in jets_det_selected]
        #     [[self.set_matching_candidates(jet_det_pp, jet_truth, jetR, '') for jet_truth in jets_truth_selected_matched] for jet_det_pp in jets_det_pp_selected]
        # else:
        #     # First fill the combined-to-pp matches, then the pp-to-pp matches
        #     [[self.set_matching_candidates(jet_det_combined, jet_det_pp, jetR, 'hDeltaR_combined_ppdet_R{{}}_Rmax{}'.format(R_max), fill_jet1_matches_only=True) for jet_det_pp in jets_det_pp_selected] for jet_det_combined in jets_det_selected]
        #     [[self.set_matching_candidates(jet_det_pp, jet_truth, jetR, 'hDeltaR_ppdet_pptrue_R{{}}_Rmax{}'.format(R_max)) for jet_truth in jets_truth_selected_matched] for jet_det_pp in jets_det_pp_selected]

        # debug
        # for jet_det_combined in jets_det_selected:
        #   print('debug7.1--jet_det',jet_det_combined.pt(),'user_info',jet_det_combined.has_user_info())
        #   if jet_det_combined.has_user_info() and jet_det_combined.python_info().closest_jet:
        #     print('matches to',jet_det_combined.python_info().closest_jet.pt())
        #     print('debug7.1--jet_det',len(jet_det_combined.constituents()))
        #     print('matches to',len(jet_det_combined.python_info().closest_jet.constituents()))
                
        # Loop through jets and set accepted matches
        # if self.is_pp:
        #     hname = 'hJetMatchingQA_R{}'.format(jetR)
        #     [self.set_matches_pp(jet_det, hname) for jet_det in jets_det_selected]
        # elif self.is_pA:
        #     hname = 'hJetMatchingQA_R{}'.format(jetR)
        #     [self.set_matches_AA(jet_det_combined, jetR, hname) for jet_det_combined in jets_det_selected]
        # else:
        #     hname = 'hJetMatchingQA_R{}_Rmax{}'.format(jetR, R_max)
        #     [self.set_matches_AA(jet_det_combined, jetR, hname) for jet_det_combined in jets_det_selected]

        # # Loop through jets and fill response histograms if both det and truth jets are unique match
        # [self.fill_jet_matches(jet_det, jetR, R_max, fj_particles_det_holes, fj_particles_truth_holes, rho_bge_det, fj_particles_det_cones, fj_particles_truth_cones) for jet_det in jets_det_selected]

    #---------------------------------------------------------------
    # Fill some background histograms
    #---------------------------------------------------------------
    def fill_background_histograms(self, fj_particles_combined_beforeCS, fj_particles_combined, jetR, i):

        # Fill rho
        rho = self.constituent_subtractor[i].bge_rho.rho()
        if self.fill_R_indep_hists and self.fill_Rmax_indep_hists:
            getattr(self, 'hRho').Fill(rho)
        
        # Fill random cone delta-pt before constituent subtraction
        if not self.skip_deltapt_RC_histograms:
            R_max = self.max_distance[i]
            self.fill_deltapt_RC_histogram(fj_particles_combined_beforeCS, rho, jetR, R_max, before_CS=True)
                    
            # Fill random cone delta-pt after constituent subtraction
            self.fill_deltapt_RC_histogram(fj_particles_combined, rho, jetR, R_max, before_CS=False)
        
    #---------------------------------------------------------------
    # Fill delta-pt histogram
    #---------------------------------------------------------------
    def fill_deltapt_RC_histogram(self, fj_particles, rho, jetR, R_max, before_CS=False):
    
        # Choose a random eta-phi in the fiducial acceptance
        phi = random.uniform(0., 2*np.pi)
        eta = random.uniform(-0.9+jetR, 0.9-jetR)
        
        # Loop through tracks and sum pt inside the cone
        pt_sum = 0.
        pt_sum_global = 0.
        for track in fj_particles:
                if self.utils.delta_R(track, eta, phi) < jetR:
                        pt_sum += track.pt()
                pt_sum_global += track.pt()
                        
        if before_CS:
                delta_pt = pt_sum - rho * np.pi * jetR * jetR
                getattr(self, 'hDeltaPt_RC_beforeCS_R{}_Rmax{}'.format(jetR, R_max)).Fill(delta_pt)
        else:
                delta_pt = pt_sum
                getattr(self, 'hDeltaPt_RC_afterCS_R{}_Rmax{}'.format(jetR, R_max)).Fill(delta_pt)
                
        # Fill mean pt
        if before_CS and self.fill_R_indep_hists and self.fill_Rmax_indep_hists:
            N_tracks = len(fj_particles)
            mean_pt = pt_sum_global/N_tracks
            getattr(self, 'hN_MeanPt').Fill(N_tracks, mean_pt)

    #---------------------------------------------------------------
    # Fill truth jet histograms
    #---------------------------------------------------------------
    def fill_truth_before_matching(self, jet, jetR, rho_bge = 0):
        
        # jet_pt = jet.pt()
        # for constituent in jet.constituents():
        #     z = constituent.pt() / jet.pt()
        #     getattr(self, 'hZ_Truth_R{}'.format(jetR)).Fill(jet.pt(), z)
                    
        # Fill 2D histogram of truth (pt, obs)
        hname = 'h_{{}}_JetPt_Truth_R{}_{{}}'.format(jetR)
        self.fill_unmatched_jet_histograms(jet, jetR, hname, rho_bge)

    #---------------------------------------------------------------
    # Fill det jet histograms
    #---------------------------------------------------------------
    def fill_det_before_matching(self, jet, jetR, rho_bge = 0):
        
        # if self.is_pp or self.fill_Rmax_indep_hists:
        #     jet_pt = jet.pt()
        #     for constituent in jet.constituents():
        #         z = constituent.pt() / jet_pt
        #         getattr(self, 'hZ_Det_R{}'.format(jetR)).Fill(jet_pt, z)
            
        # Fill groomed histograms
        # if self.thermal_model:
        #     hname = 'h_{{}}_JetPt_R{}_{{}}'.format(jetR)
        #     self.fill_unmatched_jet_histograms(jet, jetR, hname)

        if self.is_pp:
            hname = 'h_{{}}_JetPt_R{}_{{}}'.format(jetR)
            self.fill_unmatched_jet_histograms(jet, jetR, hname)

        if self.do_median_subtraction:
            hname = 'h_{{}}_JetPt_R{}_{{}}'.format(jetR)
            self.fill_unmatched_jet_histograms(jet, jetR, hname, rho_bge)
    
    #---------------------------------------------------------------
    # This function is called once for each jet
    #---------------------------------------------------------------
    def fill_unmatched_jet_histograms(self, jet, jetR, hname, rho_bge = 0):

        # Loop through each jet subconfiguration (i.e. subobservable / grooming setting)
        observable = self.observable_list[0]
        for i in range(len(self.obs_settings[observable])):
            obs_setting = self.obs_settings[observable][i]
            grooming_setting = self.obs_grooming_settings[observable][i]
            obs_label = self.utils.obs_label(obs_setting, grooming_setting)

            # Groom jet, if applicable
            if grooming_setting:
                gshop = fjcontrib.GroomerShop(jet, jetR, self.reclustering_algorithm)
                jet_groomed_lund = self.utils.groom(gshop, grooming_setting, jetR)
                if not jet_groomed_lund:
                    continue
            else:
                jet_groomed_lund = None
            
            if self.do_median_subtraction and rho_bge > 0:
                jet_pt = jet.perp()-rho_bge*jet.area() # use subtracted jet pt for energy weight calculation and pt selection for there is a non-zero UE energy density
            else:
                jet_pt = jet.perp()
            # Call user function to fill histograms
            self.fill_observable_histograms(hname, jet, jet_groomed_lund, jetR, obs_setting,
                                            grooming_setting, obs_label, jet_pt)
    
    #---------------------------------------------------------------
    # Loop through jets and call user function to fill matched
    # histos if both det and truth jets are unique match.
    #---------------------------------------------------------------
    def fill_jet_matches(self, jet_det, jetR, R_max, fj_particles_det_holes, fj_particles_truth_holes, rho_bge=0, fj_particles_det_cones = None, fj_particles_truth_cones = None):
    
        # Set suffix for filling histograms
        if R_max:
            suffix = '_Rmax{}'.format(R_max)
        else:
            suffix = ''

        if isinstance(rho_bge, tuple):
            rho_bg = rho_bge[0]
            rho_bg_truth = rho_bge[1]
        else:
            rho_bg = rho_bge
            rho_bg_truth = rho_bge
        
        # Get matched truth jet
        if jet_det.has_user_info():
            jet_truth = jet_det.python_info().match
            if self.do_median_subtraction and rho_bg > 0:
                jet_det_pt = jet_det.perp()-rho_bg*jet_det.area() # use subtracted jet pt for energy weight calculation and pt selection for there is a non-zero UE energy density
            else:
                jet_det_pt = jet_det.perp()

            if jet_truth:

                # # debug
                # print('debug8--jet det', jet_det_pt, 'size', len(jet_det.constituents()))
                # print('debug8--jet_truth', jet_truth.pt(), 'size', len(jet_truth.constituents()))

                if self.do_median_subtraction and rho_bg_truth > 0:
                    jet_truth_pt = jet_truth.perp()-rho_bg_truth*jet_truth.area() # use subtracted jet pt for energy weight calculation and pt selection for there is a non-zero UE energy density
                else:
                    jet_truth_pt = jet_truth.perp()
                
                jet_pt_det_ungroomed = jet_det_pt
                jet_pt_truth_ungroomed = jet_truth_pt
                # JES = (jet_pt_det_ungroomed - jet_pt_truth_ungroomed) / jet_pt_truth_ungroomed
                # jetpt_correction_resolution = jet_pt_det_ungroomed - jet_pt_truth_ungroomed
                # getattr(self, 'hJES_R{}{}'.format(jetR, suffix)).Fill(jet_pt_truth_ungroomed, JES)
                # getattr(self, 'hJetPtCorrRes_R{}'.format(jetR)).Fill(jet_pt_det_ungroomed, jetpt_correction_resolution)

                if self.is_pA:
                    matchedconstituents = fj.sorted_by_pt(jet_det.constituents())
                    matchedconstituents_1gev = 0
                    for c in matchedconstituents:
                        if c.pt() < 1: 
                            break
                        # getattr(self, 'h_matched_jetconstituent_pT_R{}'.format(jetR)).Fill(jet_det_pt, c.pt())
                        matchedconstituents_1gev += 1
                    # getattr(self, 'h_matched_jetconstituents_R{}'.format(jetR)).Fill(jet_det_pt, matchedconstituents_1gev)
                    # getattr(self, 'h_matched_SigJetpt_MedRho_R{}'.format(jetR)).Fill(jet_det_pt,rho_bg)

                # If Pb-Pb case, we need to keep jet_det, jet_truth, jet_pp_det
                jet_pp_det = None
                if not self.is_pp:
                    # Get pp-det jet
                    jet_pp_det = jet_truth.python_info().match

                    # Fill delta-pt histogram
                    # if jet_pp_det:
                    #     jet_pp_det_pt = jet_pp_det.pt()
                    #     delta_pt = (jet_pt_det_ungroomed - jet_pp_det_pt)
                    #     getattr(self, 'hDeltaPt_emb_R{}_Rmax{}'.format(jetR, R_max)).Fill(jet_pt_truth_ungroomed, delta_pt)

                # Loop through each jet subconfiguration (i.e. subobservable / grooming setting)
                observable = self.observable_list[0]

                if self.do_perpendicular_cone:
                    angle = np.pi/2
                    if self.randomize_cone:
                        angle = np.random.uniform(low=np.pi/3, high=2*np.pi/3)

                    perp_jet1 = fj.PseudoJet()
                    perp_jet1.reset_PtYPhiM(jet_det.pt(), jet_det.rapidity(), jet_det.phi() + angle, jet_det.m())
                    perp_jet2 = fj.PseudoJet()
                    perp_jet2.reset_PtYPhiM(jet_det.pt(), jet_det.rapidity(), jet_det.phi() - angle, jet_det.m())

                    # truth-level perp cone
                    perp_jet1_truth = fj.PseudoJet()
                    perp_jet1_truth.reset_PtYPhiM(jet_truth.pt(), jet_truth.rapidity(), jet_truth.phi() + angle, jet_truth.m())
                    perp_jet2_truth = fj.PseudoJet()
                    perp_jet2_truth.reset_PtYPhiM(jet_truth.pt(), jet_truth.rapidity(), jet_truth.phi() - angle, jet_truth.m())

                    perpcone_R = jetR
                    constituents = fj.vectorPJ()
                    for c in jet_det.constituents():
                        constituents.push_back(c)
                    constituents_truth = fj.vectorPJ()
                    for c in jet_truth.constituents():
                        constituents_truth.push_back(c)
                        
                    parts_in_perpcone1 = self.find_parts_around_jet(fj_particles_det_cones, perp_jet1, perpcone_R)
                    parts_in_perpcone1 = self.rotate_parts(parts_in_perpcone1, -angle)
                        
                    parts_in_perpcone2 = self.find_parts_around_jet(fj_particles_det_cones, perp_jet2, perpcone_R)
                    parts_in_perpcone2 = self.rotate_parts(parts_in_perpcone2, +angle)

                    parts_in_perpcone1_truth = self.find_parts_around_jet(fj_particles_truth_cones, perp_jet1_truth, perpcone_R)
                    parts_in_perpcone1_truth = self.rotate_parts(parts_in_perpcone1_truth, -angle)
                        
                    parts_in_perpcone2_truth = self.find_parts_around_jet(fj_particles_truth_cones, perp_jet2_truth, perpcone_R)
                    parts_in_perpcone2_truth = self.rotate_parts(parts_in_perpcone2_truth, +angle)

                    parts_in_cone1 = fj.vectorPJ()
                    for part in constituents:
                        part.set_user_index(999)
                        parts_in_cone1.append(part)
                    for part in parts_in_perpcone1:
                        part.set_user_index(-999)
                        parts_in_cone1.append(part)							

                    parts_in_cone2 = fj.vectorPJ()
                    for part in constituents:
                        part.set_user_index(999)
                        parts_in_cone2.append(part)
                    for part in parts_in_perpcone2:
                        part.set_user_index(-999)
                        parts_in_cone2.append(part)
                
                    parts_in_cone1_truth = fj.vectorPJ()
                    for part in constituents_truth:
                        part.set_user_index(999)
                        parts_in_cone1_truth.append(part)
                    for part in parts_in_perpcone1_truth:
                        part.set_user_index(-999)
                        parts_in_cone1_truth.append(part)							

                    parts_in_cone2_truth = fj.vectorPJ()
                    for part in constituents_truth:
                        part.set_user_index(999)
                        parts_in_cone2_truth.append(part)
                    for part in parts_in_perpcone2_truth:
                        part.set_user_index(-999)
                        parts_in_cone2_truth.append(part)
                
                
                for i in range(len(self.obs_settings[observable])):
                
                    obs_setting = self.obs_settings[observable][i]
                    grooming_setting = self.obs_grooming_settings[observable][i]
                    obs_label = self.utils.obs_label(obs_setting, grooming_setting)
                    
                    if self.debug_level > 3:
                        print('obs_label: {}'.format(obs_label))
                    
                    # Groom jets, if applicable
                    if grooming_setting:
                                        
                        # Groom det jet
                        gshop_det = fjcontrib.GroomerShop(jet_det, jetR, self.reclustering_algorithm)
                        jet_det_groomed_lund = self.utils.groom(gshop_det, grooming_setting, jetR)
                        if not jet_det_groomed_lund:
                            continue

                        # Groom truth jet
                        gshop_truth = fjcontrib.GroomerShop(jet_truth, jetR, self.reclustering_algorithm)
                        jet_truth_groomed_lund = self.utils.groom(gshop_truth, grooming_setting, jetR)
                        if not jet_truth_groomed_lund:
                            continue
                            
                    else:
                    
                        jet_det_groomed_lund = None
                        jet_truth_groomed_lund = None
                        
                    # If jetscape, pass the list of holes within R of the jet to the user
                    holes_in_det_jet = None
                    holes_in_truth_jet = None
                    if self.jetscape:
                        holes_in_det_jet = [hadron for hadron in fj_particles_det_holes if jet_det.delta_R(hadron) < jetR]
                        holes_in_truth_jet = [hadron for hadron in fj_particles_truth_holes if jet_truth.delta_R(hadron) < jetR]
                        
                        # Get the corrected jet pt by subtracting the negative recoils within R
                        for hadron in holes_in_det_jet:
                                jet_pt_det_ungroomed -= hadron.pt()
                                
                        for hadron in holes_in_truth_jet:
                                jet_pt_truth_ungroomed -= hadron.pt()

                    # # debug
                    # constituents = fj.sorted_by_pt(jet_truth.constituents())
                                                
                    # Call user function to fill histos
                    self.fill_matched_jet_histograms(jet_det, jet_det_groomed_lund, jet_truth,
                                                                 jet_truth_groomed_lund, jet_pp_det, jetR,
                                                                 obs_setting, grooming_setting, obs_label,
                                                                 jet_pt_det_ungroomed, jet_pt_truth_ungroomed,
                                                                 R_max, suffix, holes_in_det_jet=holes_in_det_jet,
                                                                 holes_in_truth_jet=holes_in_truth_jet, cone_parts_in_det_jet=None, rho=rho_bge)
                
                    if self.do_perpendicular_cone:	
                        self.fill_matched_jet_histograms(jet_det, jet_det_groomed_lund, jet_truth,
                                 jet_truth_groomed_lund, jet_pp_det, jetR,
                                 obs_setting, grooming_setting, obs_label,
                                 jet_pt_det_ungroomed, jet_pt_truth_ungroomed,
                                 R_max, suffix, holes_in_det_jet=holes_in_det_jet,
                                 holes_in_truth_jet=holes_in_truth_jet, cone_parts_in_det_jet=[parts_in_cone1, parts_in_cone2, parts_in_cone1_truth, parts_in_cone2_truth], rho=rho_bge)				

    #---------------------------------------------------------------
    # Fill response histograms -- common utility function
    #---------------------------------------------------------------
    def fill_response(self, observable, jetR, jet_pt_det_ungroomed, jet_pt_truth_ungroomed,
                                        obs_det, obs_truth, obs_label, R_max, prong_match = False):

        if self.fill_RM_histograms:
            x = ([jet_pt_det_ungroomed, jet_pt_truth_ungroomed, obs_det, obs_truth])
            x_array = array('d', x)
            name = 'hResponse_JetPt_{}_R{}_{}'.format(observable, jetR, obs_label)
            if not self.is_pp and not self.is_pA:
                name += '_Rmax{}'.format(R_max)
            getattr(self, name).Fill(x_array)
            
        if obs_truth > 1e-5:
            obs_resolution = (obs_det - obs_truth) / obs_truth
            name = 'hResidual_JetPt_{}_R{}_{}'.format(observable, jetR, obs_label)
            if not self.is_pp and not self.is_pA:
                name += '_Rmax{}'.format(R_max)
            getattr(self, name).Fill(jet_pt_truth_ungroomed, obs_truth, obs_resolution)
        
        # Fill prong-matched response
        if not self.is_pp and not self.is_pA and R_max == self.main_R_max:
            if prong_match:
            
                name = 'hResponse_JetPt_{}_R{}_{}_Rmax{}_matched'.format(observable, jetR, obs_label, R_max)
                getattr(self, name).Fill(x_array)
                
                if obs_truth > 1e-5:
                    name = 'hResidual_JetPt_{}_R{}_{}_Rmax{}_matched'.format(observable, jetR, obs_label, R_max)
                    getattr(self, name).Fill(jet_pt_truth_ungroomed, obs_truth, obs_resolution)

    #---------------------------------------------------------------
    # Calculate rho from perpendicular cone 
    # and call user histograms
    #---------------------------------------------------------------
    def analyze_perpendicular_cone(self, jets_combined_selected, jets_det_pp_selected, jets_truth_selected, jets_truth_selected_matched, jetR, fj_particles, rho=0, rhocs = 0, R_max = None):
        # Set suffix for filling histograms
        if R_max:
            suffix = '_Rmax{}'.format(R_max)
        else:
            suffix = ''
        
        # Loop through jets and call user function on particles in perp cone
        for jet in jets_combined_selected:
            
            if not self.utils.is_det_jet_accepted(jet):
                continue
            
            # jet_area = jet.area()
            # getattr(self, 'hJetArea_R{}'.format(jetR)).Fill(jet_area)
            
            # jet_Rsq = jet_area/np.pi
            # perp_area = np.pi*0.4*0.4
            perp_cone_phis = [(jet.phi()+np.pi/2)%(2*np.pi),(jet.phi()-np.pi/2)%(2*np.pi)]
            perp_cone_eta = jet.eta()
            
            for perp_cone_phi in perp_cone_phis:
                
                perp_cone_particles = fj.vectorPJ()
                perp_cone_pj = fj.PseudoJet()
                for particle in fj_particles:
                    if (np.square(particle.eta()-perp_cone_eta)+np.square(particle.phi()-perp_cone_phi) < 0.4*0.4): #jet_Rsq):
                        perp_cone_particles.append(particle)
                        perp_cone_pj += particle
            
                # perp_cone_pt = perp_cone_pj.pt()
                # perp_cone_rho = perp_cone_pt / perp_area #jet_area
                # getattr(self, 'hPerpRho_R{}'.format(jetR)).Fill(perp_cone_rho)
                # getattr(self, 'hPerpConepT_Raw_R{}'.format(jetR)).Fill(perp_cone_pt)

                # nperp_cone_particles = len(perp_cone_particles)
                # if nperp_cone_particles != 0:
                    # perp_cone_ptcorr = perp_cone_pt - rho*perp_area #jet_area
                    # perp_cone_ptcorr_woC = perp_cone_pt - rho*perp_area #jet_area
                    # perp_cone_ptcorr_CS = perp_cone_pt - rhocs*perp_area #jet_area
                    # getattr(self, 'hPerpConepT_MedCorrected_R{}'.format(jetR)).Fill(jet.pt()-rho*jet_area, perp_cone_ptcorr)
                    # getattr(self, 'hPerpConepT_CSCorrected_R{}'.format(jetR)).Fill(perp_cone_ptcorr_CS)
                    # getattr(self, 'hPerpConepTRes_MedCorrected_R{}'.format(jetR)).Fill(perp_cone_ptcorr/perp_cone_pt)
                    # getattr(self, 'hPerpConepTComp_MedCorrected_R{}'.format(jetR)).Fill(perp_cone_ptcorr,perp_cone_pt)
                    # getattr(self, 'hPerpConepTComp_CSCorrected_R{}'.format(jetR)).Fill(perp_cone_ptcorr_CS,perp_cone_pt)
             
                # getattr(self, 'hSigJetpt_PerpRho_R{}'.format(jetR)).Fill(jet.pt()-rho*jet_area, perp_cone_rho)
                # getattr(self, 'hPerpConeMult_Rho_R{}'.format(jetR)).Fill(nperp_cone_particles, perp_cone_rho)
                
                # nperp_150 = [p for p in perp_cone_particles if p.pt()>0.15]
                # nperp_1gev = [p for p in perp_cone_particles if p.pt()>1]
                # nperp_1gev_MC = [p for p in nperp_1gev if p.user_index()>=0]
                # nperp_1gev_emb = [p for p in nperp_1gev if p.user_index()<0]
                # njet_150 = [p for p in jet.constituents() if p.pt()>0.15]
                # njet_1gev = [p for p in jet.constituents() if p.pt()>1]
                # njet_1gev_MC = [p for p in njet_1gev if p.user_index()>=0]
                # njet_1gev_emb = [p for p in njet_1gev if p.user_index()<0]
                
                # for ptlo, pthi in [(20,40),(40,60),(60,80)]:
                #     if jet.pt()-rho*jet_area < pthi and jet.pt()-rho*jet_area > ptlo:
                #         getattr(self, 'hPerpConeMult_JetMult_{}{}_R{}'.format(ptlo,pthi,jetR)).Fill(nperp_cone_particles, len(jet.constituents()))
                #         getattr(self, 'hPerpConeMult_JetMult_150_{}{}_R{}'.format(ptlo,pthi,jetR)).Fill(len(nperp_150), len(njet_150))
                #         getattr(self, 'hPerpConeMult_JetMult_1GeV_{}{}_R{}'.format(ptlo,pthi,jetR)).Fill(len(nperp_1gev), len(njet_1gev))
                #         getattr(self, 'hPerpConeMult_JetMult_MC_{}{}_R{}'.format(ptlo,pthi,jetR)).Fill(len(nperp_1gev_MC), len(njet_1gev_MC))
                #         getattr(self, 'hPerpConeMult_JetMult_emb_{}{}_R{}'.format(ptlo,pthi,jetR)).Fill(len(nperp_1gev_emb), len(njet_1gev_emb))

                
                observable = self.observable_list[0]
                for i in range(len(self.obs_settings[observable])):
                    obs_setting = self.obs_settings[observable][i]
                    grooming_setting = self.obs_grooming_settings[observable][i]
                    obs_label = self.utils.obs_label(obs_setting, grooming_setting)
                    self.fill_perpcone_histograms(jet, jetR, perp_cone_particles, obs_setting, obs_label, suffix, rho_bge=rho)

    #---------------------------------------------------------------
    # Helper functions for perpendicular cone 
    # background pair subtraction
    #---------------------------------------------------------------
    def find_parts_around_jet(self, parts, jet, cone_R):
        # select particles around jet axis
        cone_parts = fj.vectorPJ()
        for part in parts:
            if jet.delta_R(part) <= cone_R:
                cone_parts.push_back(part)
        
        return cone_parts

    def rotate_parts(self, parts, rotate_phi):
        # rotate parts in azimuthal direction
        parts_rotated = fj.vectorPJ()
        for part in parts:
            pt_new = part.pt()
            y_new = part.rapidity()
            phi_new = part.phi() + rotate_phi
            m_new = part.m()
            index_new = part.user_index()
            # print('before',part.phi())
            part.reset_PtYPhiM(pt_new, y_new, phi_new, m_new)
            part.set_user_index(index_new)
            # print('after',part.phi())
            parts_rotated.push_back(part)
        
        return parts_rotated

    #---------------------------------------------------------------
    # This function is called once for each jetR
    # You must implement this
    #---------------------------------------------------------------
    def initialize_user_output_objects_R(self, jetR):
            
        raise NotImplementedError('You must implement initialize_user_output_objects_R()!')

    #---------------------------------------------------------------
    # This function is called once for each jet subconfiguration
    # You must implement this
    #---------------------------------------------------------------
    def fill_observable_histograms(self, hname, jet, jet_groomed_lund, jetR, obs_setting,
                                   grooming_setting, obs_label, jet_pt_ungroomed):

        raise NotImplementedError('You must implement fill_observable_histograms()!')

    #---------------------------------------------------------------
    # This function is called once for each matched jet subconfiguration
    # You must implement this
    #---------------------------------------------------------------
    def fill_matched_jet_histograms(self, jet_det, jet_det_groomed_lund, jet_truth,
                                    jet_truth_groomed_lund, jet_pp_det, jetR,
                                    obs_setting, grooming_setting, obs_label,
                                    jet_pt_det_ungroomed, jet_pt_truth_ungroomed,
                                    R_max, suffix,
                                    **kwargs):

        raise NotImplementedError('You must implement fill_matched_jet_histograms()!')