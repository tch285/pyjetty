#!/usr/bin/env python3

"""
  Plotting utilities for jet substructure analysis with track dataframe.
  
  Author: James Mulligan (james.mulligan@berkeley.edu)
"""

from __future__ import print_function

# General
import os
import sys
import math
import yaml
import argparse

# Data analysis and plotting
import numpy as np
from array import *
import ROOT

# Base class
from pyjetty.alice_analysis.analysis.base import common_base
from pyjetty.alice_analysis.analysis.user.substructure import analysis_utils_obs
from pyjetty.alice_analysis.analysis.user.james import plotting_utils

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)

# Suppress a lot of standard output
ROOT.gErrorIgnoreLevel = ROOT.kWarning

################################################################
class PlotGroomers(common_base.CommonBase):
  
  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, output_dir = '.', config_file = '', **kwargs):
    super(PlotGroomers, self).__init__(**kwargs)
    
    # Initialize utils class
    self.utils = analysis_utils_obs.AnalysisUtils_Obs()
    
    self.utils.set_plotting_options()
    ROOT.gROOT.ForceStyle()
    
    # Read config file
    self.config_file = config_file
    with open(config_file, 'r') as stream:
      config = yaml.safe_load(stream)
    
    self.observables = config['process_observables']
    thermal_data = config['main_response']
    self.fMC = ROOT.TFile(thermal_data, 'READ')
    
    self.max_distance = config['constituent_subtractor']['max_distance']
    self.R_max = config['constituent_subtractor']['main_R_max']
    
    self.file_format = config['file_format']
    self.output_dir = config['output_dir']

    self.ColorArray = [ROOT.kBlue-4, ROOT.kAzure+7, ROOT.kCyan-2, ROOT.kViolet-8,
                       ROOT.kBlue-6, ROOT.kGreen+3, ROOT.kPink-4, ROOT.kRed-4,
                       ROOT.kOrange-3, ROOT.kGray]
    self.MarkerArray = [20, 21, 22, 23, 33, 34, 24, 25, 26, 32]
    self.OpenMarkerArray = [24, 25, 26, 32, 27, 28]
    
    print(self)

  #---------------------------------------------------------------
  def init_observable(self, observable):
    
    with open(self.config_file, 'r') as stream:
      config = yaml.safe_load(stream)
    
    # Get the sub-configs
    self.jetR_list = config['jetR']
    self.obs_config_dict = config[observable]
    self.obs_subconfig_list = [name for name in list(self.obs_config_dict.keys()) if 'config' in name ]
    self.grooming_settings = self.utils.grooming_settings(self.obs_config_dict)
    self.obs_settings = self.utils.obs_settings(observable, self.obs_config_dict, self.obs_subconfig_list)
    self.obs_labels = [self.utils.obs_label(self.obs_settings[i], self.grooming_settings[i])
                       for i in range(len(self.obs_subconfig_list))]
    self.xtitle = self.obs_config_dict['common_settings']['xtitle']
    self.ytitle = self.obs_config_dict['common_settings']['ytitle']
    self.pt_bins_reported = self.obs_config_dict['common_settings']['pt_bins_reported']
    self.plot_overlay_list = self.obs_config_dict['common_settings']['plot_overlay_list']

    # Create output dirs
    output_dir = os.path.join(self.output_dir, observable)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  #---------------------------------------------------------------
  def plot_groomers(self):
  
    # Loop through all observables
    for observable in self.observables:
      self.init_observable(observable)
  
      # Plot for each R_max
      for R_max in self.max_distance:
      
        # Plot for each R
        for jetR in self.jetR_list:
        
          # Create output dir
          output_dir = os.path.join(self.output_dir, observable)
          output_dir = os.path.join(output_dir, 'jetR{}'.format(jetR))
          output_dir = os.path.join(output_dir, 'Rmax{}'.format(R_max))
        
          # Plot money plot for all observables
          for i, _ in enumerate(self.obs_subconfig_list):

            obs_setting = self.obs_settings[i]
            grooming_setting = self.grooming_settings[i]
            obs_label = self.utils.obs_label(obs_setting, grooming_setting)
            self.create_output_subdir(output_dir, self.utils.grooming_label(grooming_setting))
            


            self.plot_money_plot(observable, jetR, R_max, obs_label, obs_setting, grooming_setting, output_dir)

          # Plot performance plots only once
          if observable == 'theta_g':

            # Create output subdirectories
            output_dir = os.path.join(self.output_dir, 'performance')
            self.create_output_subdir(output_dir, 'delta_pt')
            self.create_output_subdir(output_dir, 'prong_matching_fraction_pt')
            self.create_output_subdir(output_dir, 'prong_matching_deltaR')
            self.create_output_subdir(output_dir, 'prong_matching_deltaZ')
            self.create_output_subdir(output_dir, 'prong_matching_correlation')
            
            self.plotting_utils = plotting_utils.PlottingUtils(output_dir, self.config_file, R_max=R_max,
                                                               thermal = False, groomer_studies = True)
        
            # Plot some subobservable-independent performance plots
            self.plotting_utils.plot_delta_pt(jetR, self.pt_bins_reported)

            # Plot prong matching histograms
            self.prong_match_threshold = 0.5
            min_pt = 80.
            max_pt = 100.
            prong_list = ['leading', 'subleading']
            match_list = ['leading', 'subleading', 'ungroomed', 'outside']
            for i, overlay_list in enumerate(self.plot_overlay_list):
              for prong in prong_list:
                for match in match_list:

                  hname = 'hProngMatching_{}_{}_JetPt_R{}'.format(prong, match, jetR)
                  self.plotting_utils.plot_prong_matching(i, jetR, hname, self.obs_subconfig_list, self.obs_settings, self.grooming_settings, overlay_list, self.prong_match_threshold)
                  self.plotting_utils.plot_prong_matching_delta(i, jetR, hname, self.obs_subconfig_list, self.obs_settings, self.grooming_settings, overlay_list, self.prong_match_threshold, min_pt, max_pt, plot_deltaz=False)

                  if 'subleading' in prong or 'leading' in prong:
                    hname = 'hProngMatching_{}_{}_JetPtZ_R{}'.format(prong, match, jetR)
                    self.plotting_utils.plot_prong_matching_delta(i, jetR, hname, self.obs_subconfig_list, self.obs_settings, self.grooming_settings, overlay_list, self.prong_match_threshold, min_pt, max_pt, plot_deltaz=True)

              hname = 'hProngMatching_subleading-leading_correlation_JetPt_R{}'.format(jetR)
              self.plotting_utils.plot_prong_matching_correlation(i, jetR, hname, self.obs_subconfig_list, self.obs_settings, self.grooming_settings, overlay_list, self.prong_match_threshold)

  #---------------------------------------------------------------
  def plot_money_plot(self, observable, jetR, R_max, obs_label, obs_setting, grooming_setting, output_dir):
      
    if observable in ['zg', 'theta_g']:
      # (pt, zg, theta_g, flag)
      #  Flag based on where >50% of subleading matched pt resides:
      #    1: subleading
      #    2: leading
      #    3: ungroomed
      #    4: outside
      #    5: other (i.e. 50% is not in any of the above)
      #    6: pp-truth passed grooming, but combined jet failed grooming
      #    7: combined jet passed grooming, but pp-truth failed grooming
      #    8: both pp-truth and combined jet failed SoftDrop
      name = 'h_theta_g_zg_JetPt_R{}_{}_Rmax{}'.format(jetR, obs_label, R_max)
      h4D = self.fMC.Get(name)
      if h4D.GetSumw2() is 0:
        h4D.Sumw2()
        
      # Loop through pt slices, and plot 1D projection onto observable
      for i in range(0, len(self.pt_bins_reported) - 1):
        min_pt_truth = self.pt_bins_reported[i]
        max_pt_truth = self.pt_bins_reported[i+1]
        
        self.plot_obs_projection(observable, h4D.Clone(), jetR, obs_label, obs_setting,
                                 grooming_setting, min_pt_truth, max_pt_truth, output_dir)

  #---------------------------------------------------------------
  def plot_obs_projection(self, observable, h4D, jetR, obs_label, obs_setting,
                          grooming_setting, min_pt, max_pt, output_dir):
  
    # Set pt range
    h4D.GetAxis(0).SetRangeUser(min_pt, max_pt)

    # Draw histogram
    c = ROOT.TCanvas('c','c: hist',600,450)
    c.cd()
    ROOT.gPad.SetLeftMargin(0.2)
    ROOT.gPad.SetBottomMargin(0.15)

    myBlankHisto = ROOT.TH1F('myBlankHisto','Blank Histogram', 20, 0, 1.)
    myBlankHisto.SetNdivisions(505)
    myBlankHisto.GetXaxis().SetTitleOffset(1.4)
    myBlankHisto.GetYaxis().SetTitleOffset(1.6)
    myBlankHisto.SetMinimum(0.)
    myBlankHisto.GetXaxis().SetTitle(self.xtitle)
    ytitle = '#frac{{d#it{{N}}}}{{d{}}}'.format(self.xtitle)
    myBlankHisto.GetYaxis().SetTitle(ytitle)
    myBlankHisto.Draw()
    
    #h_stack = ROOT.THStack('h_stack', 'stacked')
    
    leg = ROOT.TLegend(0.6,0.55,0.75,0.8)
    self.utils.setup_legend(leg, 0.032)
    
    h_list = [] # Store hists in a list, since otherwise it seems I lose the marker information
                # (removed from memory?)
    
    legend_list = ['subleading', 'leading', 'ungroomed', 'outside', 'other',
                   'combined fail', 'truth fail', 'both fail']
    
    # Loop over each flag
    for i in range(8):
      flag = i+1
      h4D.GetAxis(3).SetRange(flag, flag)

      # Project onto 1D
      if observable == 'theta_g':
        h1D = h4D.Projection(2)
        h1D.SetName('h1D_{}'.format(i))
      elif observable == 'zg':
        h1D = h4D.Projection(1)
        h1D.SetName('h1D_{}'.format(i))

      h1D.SetLineColor(self.ColorArray[i])
      h1D.SetLineWidth(2)
      h1D.GetYaxis().SetTitleOffset(1.5)
      if i == 0:
        myBlankHisto.SetMaximum(2*h1D.GetMaximum())

      h1D.Draw('hist same')
      #h_stack.Add(h1D);
      leg.AddEntry(h1D, legend_list[i], 'L')
      leg.Draw('same')
    
    #h_stack.Draw()
    
    text_latex = ROOT.TLatex()
    text_latex.SetNDC()
    
    x = 0.23
    y = 0.85
    text_latex.SetTextSize(0.04)
    text = 'PYTHIA8 embedded in thermal background'
    text_latex.DrawLatex(x, y, text)
        
    text = '#sqrt{#it{s_{#it{NN}}}} = 5.02 TeV'
    text_latex.DrawLatex(x, y-0.05, text)

    text = 'Charged jets   anti-#it{k}_{T}'
    text_latex.DrawLatex(x, y-0.1, text)
    
    text = '#it{R} = ' + str(jetR) + '   | #it{#eta}_{jet}| < 0.5'
    text_latex.DrawLatex(x, y-0.15, text)
    
    text = self.utils.formatted_grooming_label(grooming_setting)
    text_latex.DrawLatex(x, y-0.2, text)
    
    text = str(int(min_pt)) + ' < #it{p}_{T, ch jet}^{pp-det} < ' + str(int(max_pt)) + ' GeV/#it{c}'
    text_latex.DrawLatex(x, y-0.25, text)

    output_filename = os.path.join(output_dir, '{}/money_plot_{}_{}-{}.pdf'.format(self.utils.grooming_label(grooming_setting),
                                          observable, obs_label, min_pt, max_pt))
    c.SaveAs(output_filename)
    c.Close()

  #---------------------------------------------------------------
  # Create a single output subdirectory
  #---------------------------------------------------------------
  def create_output_subdir(self, output_dir, name):

    output_subdir = os.path.join(output_dir, name)
    setattr(self, 'output_dir_{}'.format(name), output_subdir)
    if not os.path.isdir(output_subdir):
      os.makedirs(output_subdir)

    return output_subdir

#----------------------------------------------------------------------
if __name__ == '__main__':

  # Define arguments
  parser = argparse.ArgumentParser(description='Jet substructure analysis')
  parser.add_argument('-c', '--configFile', action='store',
                      type=str, metavar='configFile',
                      default='analysis_config.yaml',
                      help='Path of config file for analysis')

  # Parse the arguments
  args = parser.parse_args()
  
  print('Configuring...')
  print('configFile: \'{0}\''.format(args.configFile))
  
  # If invalid configFile is given, exit
  if not os.path.exists(args.configFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
    sys.exit(0)

  analysis = PlotGroomers(config_file = args.configFile)
  analysis.plot_groomers()
