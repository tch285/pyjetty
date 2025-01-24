#!/usr/bin/env python3

"""
Analysis class to read an intermediary ROOT TTree of data
and produce analysis objects via RDataFrame.

Author: Tucker Hwang (tucker_hwang@berkeley.edu)
"""

import argparse
import logging

# General
import os
import sys

# Data analysis and plotting
import numpy as np
import ROOT
import yaml
from time import perf_counter

import pyjetty.alice_analysis.process.user.thwang.tree_utils as utils

logger = logging.getLogger(__name__)

ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()

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

################################################################
class ProcessTree_ENC_Data:
    # ---------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------
    def __init__(
        self,
        input_file,
        config_file,
        output_dir,
        output_filename = None,
    ):
        # Initialize base class
        self.input_file = input_file
        self.config_file = config_file
        self.output_dir = output_dir
        self.output_filename =  output_filename if output_filename is not None else "AnalysisResults.root"

        # Create output dir
        # if not self.output_dir.endswith("/"):
        #     self.output_dir = self.output_dir + "/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Create output file
        self.output_filepath = os.path.join(self.output_dir, self.output_filename)

        with open(self.config_file, "r") as stream:
            config = yaml.safe_load(stream)

        self.targets = config['targets']
        self.jetR = config['jetR']
        self.bin_info = config['bins']
        self.rules = config['rules']

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

    def calc_kt(self, p1, p2):
        # original formula below
        # return 0.5*np.sqrt( pow(p1.pt(),2)+pow(p1.pt(),2)+2*p1.pt()*p2.pt()*np.cos(p1.phi()-p2.phi()) )
        # fixed formula below
        return 0.5 * np.sqrt(
            pow(p1.pt(), 2)
            + pow(p2.pt(), 2)
            + 2 * p1.pt() * p2.pt() * np.cos(p1.phi() - p2.phi())
        )

    def note_time(self, msg): logger.info(f"{msg}: ---------- {perf_counter() - self.start_time:.3f} sec. ----------")

    def process_data_trees(self):
        self.start_time = perf_counter()
        logger.info("Beginning analysis.")
        self.parse_observables()
        self.initialize_trees()
        self.setup_histo_bins()

        self.set_rules()

        self.save_histos()
        self.note_time("Completed analysis")

    def parse_observables(self):
        self.valid_targets = []
        self.sources = set()
        self.named_recipes = {}
        self.histname2src = {}
        for obs in self.targets:
            logger.debug(f"Parsing observable {obs}:")
            try:
                path = utils.find_key_path(self.rules, obs)
                logger.debug(f"  Found keypath: {path}:")
                source = utils.find_source_tree(self.rules, path)
                logger.debug(f"  Found {source=}:")
                recipe = utils.build_recipe(self.rules, path)
                logger.debug(f"  Found {recipe=}")
                params = utils.find_parameters(self.rules, path)
                logger.debug(f"  Found {params=}")
                relparams = utils.find_relevant_parameters(params, recipe)
                logger.debug(f"  Found {relparams=}")
                pdirs = utils.parse_directives(relparams, recipe)
                logger.debug(f"  Found {pdirs=}")
                for tag, recipe in pdirs.items():
                    histname = f"{obs}_{tag}" if tag else obs
                    self.named_recipes[histname] = recipe
                    self.histname2src[histname] = source
                self.valid_targets.append(obs)
                self.sources.add(source)
            except utils.InvalidTargetError as e:
                logger.warning(f"Invalid target '{obs}': {e.msg}")
        self.note_time("Completed observable parsing")

    def initialize_trees(self):
        src2br = {'pairs':  'pairs',
                  'parts':  'parts',
                  'jets':   'jets',
                  'events': 'events',
        }
        valid_sources = set(src2br.keys())
        if self.sources - valid_sources:
            raise ValueError(f"Invalid sources found: {self.sources - valid_sources}")
        # extract branches needed (may be fewer than the number of sources)
        branches = {src2br[source] for source in self.sources}

        # create RDF for each branch
        dfs = {br: ROOT.RDataFrame(br, self.input_file) for br in branches}

        self.trees = {}
        # for each source name, assign a RDF with possible filtering
        if 'pairs' in self.sources:
            self.trees['pairs'] = dfs[src2br['pairs']]

        if 'parts' in self.sources:
            self.trees['parts'] = dfs[src2br['parts']]

        if 'jets' in self.sources:
            self.trees['jets'] = dfs[src2br['jets']]

        if 'events' in self.sources:
            self.trees['events'] = dfs[src2br['events']]

        self.note_time("Completed tree initialization")
    
    def setup_histo_bins(self):
        self.bins = {}
        self.nbins = {}
        for name, params in self.bin_info.items():
            self.bins[name] = utils.calc_bins(params)
            self.nbins[name] = params[-1]
        self.note_time("Completed bin setup")

    def set_rules(self):
        self.hists = {}
        for histname, recipe in self.named_recipes.items():
            hist = self.trees[self.histname2src[histname]]
            for directive in recipe:
                hist = self.apply_directive(histname, hist, directive)
            self.hists[histname] = hist

        self.note_time("Completed rule setting")

    def apply_directive(self, histname, source, directive):
        dirclass, *args = directive
        if dirclass == 'f':
            return source.Filter(*args)
        elif dirclass == 'd':
            return source.Define(*args)
        elif dirclass == 'h':
            if args[0] == 1:
                title, xbins, *remargs = args[1:]
                return source.Histo1D((histname, title, self.nbins[xbins], self.bins[xbins]), *remargs)
            elif args[0] == 2:
                title, xbins, ybins, *remargs = args[1:]
                return source.Histo2D((histname, title, self.nbins[xbins], self.bins[xbins], self.nbins[ybins], self.bins[ybins]), *remargs)
            else:
                raise NotImplementedError(f"Histograms of dimension {args[0]} not implemented.")

    def save_histos(self):
        with ROOT.TFile(self.input_file, "READ") as infile:
            hist_nev = infile.Get("hNevents")
            hist_nevc = hist_nev.Clone()
            hist_nevc.SetDirectory(0)

        outfile = ROOT.TFile(self.output_filepath, "RECREATE")
        for hist in self.hists.values():
            hist.Write()
        outfile.WriteTObject(hist_nevc)
        outfile.Close()
        self.note_time("Completed histogram saving")


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
        default="AnalysisResults.root",
        help="Output filename for output to be written to",
    )
    parser.add_argument(
        "-n",
        "--nthreads",
        type=int,
        metavar="number-of-threads",
        default=0,
        help="Number of threads",
    )

    # Parse the arguments
    args = parser.parse_args()

    handler = logging.StreamHandler()

    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(funcName)s - %(message)s"
    )
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(funcName)s - %(message)s'))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.info("Configuring...")
    logger.info(f"Input file: '{args.input_file}'")
    logger.info(f"Config file: '{args.config_file}'")
    logger.info(f"Ouput directory: '{args.output_dir}'")
    logger.info(f"Ouput filename: '{args.output_filename}'")

    # If invalid inputFile is given, exit
    if not os.path.exists(args.input_file):
        logger.critical(f"Input file '{args.input_file}' does not exist, exiting.")
        sys.exit(1)

    # If invalid configFile is given, exit
    if not os.path.exists(args.config_file):
        logger.critical(f"Config file '{args.config_file}' does not exist, exiting.")
        sys.exit(1)

    if args.nthreads <= 0:
        ROOT.EnableImplicitMT()
        logger.info(f"Using maximum number of threads: {ROOT.ROOT.GetThreadPoolSize()}")
    else:
        ROOT.EnableImplicitMT(args.nthreads)
        # ROOT.DisableImplicitMT()
        logger.info(f"Using {args.nthreads} threads with ROOT.")
    
    analysis = ProcessTree_ENC_Data(
        input_file=args.input_file,
        config_file=args.config_file,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
    )
    analysis.process_data_trees()