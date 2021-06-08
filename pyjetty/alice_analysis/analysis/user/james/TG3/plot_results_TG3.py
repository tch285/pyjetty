"""
macro for constructing TG3 plots

The script has to support a few things:
 - Plot either ratio alone, or distribution+ratio
 - Plot either data from hepdata, or custom data file
 - Plot a variety of theory predictions, either for ratio or distribution
 
To do this, we write a separate plotting function for each observable
We include generic functions for:
  - Plotting ratio
  - Plotting distribution+ratio
  - Initialize common config items

Author: James Mulligan (james.mulligan@berkeley.edu)
"""

# General
import os
import sys
import yaml
import argparse

# Data analysis and plotting
import ROOT
import numpy as np

# Base class
sys.path.append('.')
from pyjetty.alice_analysis.analysis.base import common_base
from pyjetty.alice_analysis.analysis.user.substructure import analysis_utils_obs

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)

################################################################
class PlotResults(common_base.CommonBase):

    # ---------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------
    def __init__(self, config_file='', output_dir='', **kwargs):
        super(PlotResults, self).__init__(**kwargs)
        self.config_file = config_file
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.utils = analysis_utils_obs.AnalysisUtils_Obs()
        self.utils.set_plotting_options()
        ROOT.gROOT.ForceStyle()
            
        self.data_color = ROOT.kGray+3
        self.data_markers = [25, 21]
        self.alpha = 0.7
        self.marker_size = 1.5
        self.line_width = 2
        self.line_style = 1
        self.theory_colors = [ROOT.kViolet-8, ROOT.kAzure-4, ROOT.kTeal-8, ROOT.kRed-7, ROOT.kPink+1, ROOT.kBlue-10]
                            # ROOT.kBlue-10, ROOT.kOrange+6, ROOT.kCyan-2
        
        print(self)
        
    #-------------------------------------------------------------------------------------------
    #
    #-------------------------------------------------------------------------------------------
    def plot_results(self):
        
        # Charged particle RAA
        #self.plot_hadron_raa()

        # Inclusive full jet RAA
        #self.plot_jet_raa()
        
        # Charged jet g
        self.plot_girth()
        
        # Charged jet mass
        self.plot_mass()
        
        # Charged Soft Drop zg
        self.plot_zg()
        
        # Charged Soft Drop theta_g
        self.plot_tg()
        
        # h-jet
        #self.plot_semi_inclusive_chjet_IAA()
        #self.plot_semi_inclusive_chjet_dphi()
        
    #-------------------------------------------------------------------------------------------
    #
    #-------------------------------------------------------------------------------------------
    def plot_girth(self):
    
        # Initialize data and theory info
        self.observable = 'girth'
        self.init_result(self_normalize=True)
             
        # Plot distribution + ratio
        self.y_max = 37
        self.y_ratio_min = 0.
        self.y_ratio_max = 2.49
        pp_label = 'PYTHIA6'
        self.ytitle = f'#frac{{1}}{{#it{{#sigma}}}} #frac{{d#it{{#sigma}}}}{{ d{self.xtitle} }}'
        self.plot_distribution_and_ratio(pp_label=pp_label)

    #-------------------------------------------------------------------------------------------
    def plot_mass(self):
    
        # Initialize data and theory info
        self.observable = 'mass'
        self.init_result(self_normalize=True)
             
        # Plot distribution + ratio
        self.y_max = 0.3
        self.y_ratio_min = 0.
        self.y_ratio_max = 2.49
        pp_label = 'PYTHIA6'
        self.ytitle = f'#frac{{1}}{{#it{{#sigma}}}} #frac{{d#it{{#sigma}}}}{{ d{self.xtitle} }}'
        self.plot_distribution_and_ratio(pp_label=pp_label, plot_ratio=False)
        
        
    #-------------------------------------------------------------------------------------------
    def plot_zg(self):
        
        # Initialize data and theory info
        self.observable = 'zg'
        self.init_result()
             
        # Plot distribution + ratio
        self.y_max = 10.
        self.y_ratio_min = 0.6
        self.y_ratio_max = 1.59
        self.ytitle = f'#frac{{1}}{{#it{{#sigma}}_{{jet, inc}}}} #frac{{d#it{{#sigma}}}}{{ d{self.xtitle} }}'
        self.plot_distribution_and_ratio()
        #hname = f'h_chjet_zg_alice_R{R}{self.suffix}Scaled')

    #-------------------------------------------------------------------------------------------
    def plot_tg(self):
        
        # Initialize data and theory info
        self.observable = 'theta_g'
        self.init_result()
             
        # Plot distribution + ratio
        self.y_max = 4.1
        self.y_ratio_min = 0.
        self.y_ratio_max = 2.69
        self.ytitle = f'#frac{{1}}{{#it{{#sigma}}_{{jet, inc}}}} #frac{{d#it{{#sigma}}}}{{ d{self.xtitle} }}'
        self.plot_distribution_and_ratio()
        #hname = f'h_chjet_tg_alice_R{R}{self.suffix}Scaled')

    #---------------------------------------------------------------
    # Initialize a given observable
    #---------------------------------------------------------------
    def init_result(self, self_normalize=False):
    
        # Initialize an empty dict containing relevant info
        self.observable_settings = {}
        
        # Store theoretical predictions and labels in list
        self.observable_settings['prediction_distribution_list'] = []
        self.observable_settings['prediction_distribution_labels'] = []
        self.observable_settings['prediction_ratio_list'] = []
        self.observable_settings['prediction_ratio_labels'] = []
    
        # Load config file
        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)
            result = config[self.observable]
            
        # Common settings
        self.xtitle = result['xtitle']
        self.sqrts = result['sqrts']
        self.centrality = result['centrality']
        self.jetR = result['jetR']
        self.pt = result['pt']
        if 'eta_R' in result:
            self.eta_R = result['eta_R']
            self.eta_cut = np.round(self.eta_R - self.jetR, decimals=1)
        if 'obs_label' in result:
            self.obs_label = result['obs_label']
        if 'zcut' in result:
            self.zcut = result['zcut']
        if 'theory_config' in result:
            self.theory_config = result['theory_config']

        # Data -- distribution
        if self.observable in ['girth', 'mass']:
            file = ROOT.TFile(result['file'], 'READ')
            self.observable_settings['g_pp'] = file.Get(result['name_pp'])
            self.observable_settings['h_AA_stat'] = file.Get(result['name_AA_stat'])
            self.observable_settings['g_AA_sys'] = file.Get(result['name_AA_sys'])
            self.observable_settings['h_AA_stat'].SetDirectory(0)
            self.bins = np.array(self.observable_settings['h_AA_stat'].GetXaxis().GetXbins())
        elif self.observable in ['zg', 'theta_g']:
            self.file_pp_name = result['file_pp']
            self.file_AA_name = result['file_AA']
            self.file_pp = ROOT.TFile(self.file_pp_name, 'READ')
            self.file_AA = ROOT.TFile(self.file_AA_name, 'READ')
            main_result_name = f'hmain_{self.observable}_R{self.jetR}_{self.obs_label}_{self.pt[0]}-{self.pt[1]}'
            sys_total_name = f'hResult_{self.observable}_systotal_R{self.jetR}_{self.obs_label}_{self.pt[0]}-{self.pt[1]}'
            self.observable_settings['h_pp'] = self.file_pp.Get(main_result_name)
            self.observable_settings['h_pp_sys'] = self.file_pp.Get(sys_total_name)
            self.observable_settings['h_AA_stat'] = self.file_AA.Get(main_result_name)
            self.observable_settings['h_AA_sys'] = self.file_AA.Get(sys_total_name)
            self.observable_settings['h_pp'].SetDirectory(0)
            self.observable_settings['h_pp_sys'].SetDirectory(0)
            self.observable_settings['h_AA_stat'].SetDirectory(0)
            self.observable_settings['h_AA_sys'].SetDirectory(0)
            self.bins = np.array(self.observable_settings['h_AA_stat'].GetXaxis().GetXbins())[1:]

            h_tagging_fraction_name = f'h_tagging_fraction_R{self.jetR}_{self.obs_label}'
            self.h_tagging_fraction_pp = self.file_pp.Get(h_tagging_fraction_name)
            self.h_tagging_fraction_AA = self.file_AA.Get(h_tagging_fraction_name)
            bin = self.h_tagging_fraction_pp.GetXaxis().FindBin((self.pt[0] + self.pt[1])/2)
            self.f_tagging_pp = self.h_tagging_fraction_pp.GetBinContent(bin)
            bin = self.h_tagging_fraction_AA.GetXaxis().FindBin((self.pt[0] + self.pt[1])/2)
            self.f_tagging_AA = self.h_tagging_fraction_AA.GetBinContent(bin)
            
        # Data -- ratio
        if self.observable == 'girth':
            #self.observable_settings['hepdata_ratio'] = result['hepdata_ratio']
            #f = ROOT.TFile(self.inclusive_chjet_observables['g_alice']['hepdata'], 'READ')
            #dir = f.Get('Table 11')
            #h_data = dir.Get('Graph1D_y1')
            #h_data_list.append([h_data, '0-10%'])
            #f.Close()
            self.observable_settings['h_ratio_stat'] = file.Get(result['name_ratio_stat'])
            self.observable_settings['g_ratio_sys'] = file.Get(result['name_ratio_sys'])
            self.observable_settings['h_ratio_stat'].SetDirectory(0)
        elif self.observable in ['zg', 'theta_g']:
            self.observable_settings['h_ratio_stat'] = self.observable_settings['h_AA_stat'].Clone()
            self.observable_settings['h_ratio_stat'].Divide(self.observable_settings['h_pp'])
            self.observable_settings['h_ratio_sys'] = self.observable_settings['h_AA_sys'].Clone()
            self.observable_settings['h_ratio_sys'].Divide(self.observable_settings['h_pp_sys'])
        
        # Theory -- JETSCAPE
        if 'file_jetscape_pp' in result:
            
            # pp
            f_jetscape_pp = ROOT.TFile(result['file_jetscape_pp'], 'READ')
            h_jetscape_pp = f_jetscape_pp.Get(result['hname_jetscape'])
            h_jetscape_pp.SetDirectory(0)
            f_jetscape_pp.Close()
            
            # AA
            f_jetscape_AA = ROOT.TFile(result['file_jetscape_AA'], 'READ')
            h_jetscape_AA = f_jetscape_AA.Get(result['hname_jetscape'])
            h_jetscape_AA.SetDirectory(0)
            #if self.observable == 'jet':
            #    h_jetscape_AA.Scale(0.5) # Since we hadd [0,5] and [5,10]
            f_jetscape_AA.Close()

            ## For hadrons, impose a 1 GeV minimum, and subtract the recoil hadrons
            #if raa_type == 'hadron':
            #    # Impose 1 Gev minimum
            #    h_pp_xbins = np.array(h_pp.GetXaxis().GetXbins())
            #    h_pp_xbins = h_pp_xbins[(h_pp_xbins>=8.)]
            #    h_pp = h_pp.Rebin(h_pp_xbins.size-1, f'{hname}_pp_rebinned', h_pp_xbins)

            #    h_AA_xbins = np.array(h_AA.GetXaxis().GetXbins())
            #    h_AA_xbins = h_AA_xbins[(h_AA_xbins>=8.)]
            #    h_AA = h_AA.Rebin(h_AA_xbins.size-1, f'{hname}_{self.dir_AA}rebinned', h_AA_xbins)
            #
            #    # Subtract holes
            #    filename = os.path.join(self.output_dir, f'{self.dir_AA}/AnalysisResultsFinal.root')
            #    f_AA = ROOT.TFile(filename, 'READ')
            #    h_recoil_name = 'h_hadron_pt_alice_holesScaled'
            #    h_recoil = f_AA.Get(h_recoil_name)
            #    h_recoil.SetDirectory(0)
            #    h_recoil_rebinned = h_recoil.Rebin(h_AA_xbins.size-1, f'{h_recoil_name}_{self.dir_AA}', h_AA_xbins)
            #    h_AA.Add(h_recoil_rebinned, -1)
            #    h_AA.SetDirectory(0)
            #    f_AA.Close()
            
            # Normalization
            h_jetscape_pp.Scale(1., 'width')
            h_jetscape_AA.Scale(1., 'width')
            if self_normalize:
                if self.observable in ['zg', 'theta_g']:
                    min_bin = 0
                else:
                    min_bin = 1
                h_jetscape_AA.Scale(1./h_jetscape_AA.Integral(min_bin, h_jetscape_AA.GetNbinsX()))
                h_jetscape_pp.Scale(1./h_jetscape_pp.Integral(min_bin, h_jetscape_pp.GetNbinsX()))
                
            # Form ratio
            h_ratio_jetscape = h_jetscape_AA
            h_ratio_jetscape.Divide(h_jetscape_pp)
            self.observable_settings['prediction_ratio_list'].append(h_ratio_jetscape)
            self.observable_settings['prediction_ratio_labels'].append('JETSCAPE')
            
            if self.observable == 'mass':
                self.observable_settings['prediction_distribution_list'].append(h_jetscape_AA)
                self.observable_settings['prediction_distribution_labels'].append('JETSCAPE')
        
        # Theory -- Hybrid
        if self.observable == 'girth':
            self.observable_settings['prediction_ratio_list'].append(file.Get(result['name_ratio_hybrid_wake0_lres0']))
            self.observable_settings['prediction_ratio_labels'].append('Hybrid Model, #it{L}_{res} = 0, wake off')
            
            self.observable_settings['prediction_ratio_list'].append(file.Get(result['name_ratio_hybrid_wake1_lres0']))
            self.observable_settings['prediction_ratio_labels'].append('Hybrid Model, #it{L}_{res} = 0, wake on')

            #self.observable_settings['prediction_ratio_list'].append(file.Get(result['name_ratio_hybrid_wake0_lres2']))
            #self.observable_settings['prediction_ratio_labels'].append('Hybrid Model, #it{L}_{res} = 2#pi/T, wake off')

            self.observable_settings['prediction_ratio_list'].append(file.Get(result['name_ratio_hybrid_wake0_lresinf']))
            self.observable_settings['prediction_ratio_labels'].append('Hybrid Model, #it{L}_{res} = #infty, wake off')
            file.Close()
            
        if self.observable == 'mass':
            self.observable_settings['prediction_distribution_list'].append(file.Get(result['name_hybrid_wake0_lres0']))
            self.observable_settings['prediction_distribution_labels'].append('Hybrid Model, #it{L}_{res} = 0, wake off')
            
            self.observable_settings['prediction_distribution_list'].append(file.Get(result['name_hybrid_wake1_lres0']))
            self.observable_settings['prediction_distribution_labels'].append('Hybrid Model, #it{L}_{res} = 0, wake on')

        # Theory -- JEWEL
        if self.observable == 'mass':
            self.observable_settings['prediction_distribution_list'].append(file.Get(result['name_jewel_rec']))
            self.observable_settings['prediction_distribution_labels'].append('JEWEL, recoils on')
            
            self.observable_settings['prediction_distribution_list'].append(file.Get(result['name_jewel_norec']))
            self.observable_settings['prediction_distribution_labels'].append('JEWEL, recoils off')
            
        # Theory -- zg/tg
        if self.observable in ['zg', 'theta_g']:
            self.bin_array = np.array(self.observable_settings['h_pp'].GetXaxis().GetXbins())[1:]

            with open(self.theory_config, 'r') as stream:
                config = yaml.safe_load(stream)
                  
            self.label_list = []
            self.sublabel_list = []
            self.prediction_g_list = []
          
            self.prediction_types = [name for name in list(config.keys())]
            for type in self.prediction_types:
          
                theory = config[type]
                configs = [name for name in list(theory.keys()) if 'config' in name ]
                for prediction in configs:
                    theory_prediction = theory[prediction]
                    config_jetR = theory_prediction['jetR']
                    if config_jetR == self.jetR:
                    
                        plot_list = theory['plot_list']
                        
                        if type in ['yang_ting', 'guangyou']:
                            continue
                    
                        if type == 'lbnl':
                        
                            if not 'med q/g' in theory_prediction['sublabel']:
                                continue
                        
                            x = np.array(theory_prediction['x'])
                            y_pp = np.array(theory_prediction['y_pp'])

                            n = len(x)
                            xerr = np.zeros(n)

                            if 'f_q' in theory_prediction:
                                f_q = theory_prediction['f_q']
                                f_q_min = theory_prediction['f_q_min']
                                f_q_max = theory_prediction['f_q_max']
                                y_AA_quark = np.array(theory_prediction['y_AA_quark'])
                                y_AA_gluon = np.array(theory_prediction['y_AA_gluon'])

                                y_AA = f_q*y_AA_quark + (1-f_q)*y_AA_gluon
                                y_AA_min = f_q_min*y_AA_quark + (1-f_q_min)*y_AA_gluon
                                y_AA_max = f_q_max*y_AA_quark + (1-f_q_max)*y_AA_gluon

                                ratio = np.divide(y_AA, y_pp)
                                ratio_lower = np.divide(y_AA_min, y_pp)
                                ratio_upper = np.divide(y_AA_max, y_pp)
                                g = ROOT.TGraphAsymmErrors(n, x, ratio, xerr, xerr, ratio-ratio_lower, ratio_upper-ratio)

                            else:
                                y_AA = np.array(theory_prediction['y_AA'])
                                ratio = np.divide(y_AA, y_pp)
                                g = ROOT.TGraph(n, x, ratio)

                        elif type == 'caucal':
                                    
                            x = np.array(theory_prediction['x'])
                            ratio = np.array(theory_prediction['ratio'])
                            ratio_neg_unc_tot = np.array(theory_prediction['ratio_neg_unc_tot'])
                            ratio_pos_unc_tot = np.array(theory_prediction['ratio_pos_unc_tot'])

                            n = len(x)
                            xerr = np.zeros(n)
                            g = ROOT.TGraphAsymmErrors(n, x, ratio, xerr, xerr, ratio_neg_unc_tot, ratio_pos_unc_tot)
                        
                        elif type == 'guangyou':
                        
                            x = np.array(theory_prediction['x'])
                            ratio = np.array(theory_prediction['ratio'])

                            n = len(x)
                            g = ROOT.TGraph(n, x, ratio)
                        
                        elif type in ['hybrid_model', 'yang_ting']:
                        
                            x = np.array(theory_prediction['x'])
                            ratio_lower = np.array(theory_prediction['ratio_lower'])
                            ratio_upper = np.array(theory_prediction['ratio_upper'])
                            ratio = (ratio_lower + ratio_upper) / 2.

                            n = len(x)
                            xerr = np.zeros(n)
                            g = ROOT.TGraphAsymmErrors(n, x, ratio, xerr, xerr, ratio-ratio_lower, ratio_upper-ratio)
                        
                        elif type == 'jetscape':

                            plot_ratio_directly = True
                            if plot_ratio_directly:
                                x = np.array(theory_prediction['x_ratio'])
                                ratio = np.array(theory_prediction['ratio'])
                                ratio_err = np.array(theory_prediction['ratio_err'])
                            else:
                                # Get distributions
                                xbins = np.array(theory_prediction['xbins'])
                                y_pp = np.array(theory_prediction['y_pp'])
                                y_pp_err = np.array(theory_prediction['y_pp_err'])
                                y_AA = np.array(theory_prediction['y_AA'])
                                y_AA_err = np.array(theory_prediction['y_AA_err'])
                                
                                # Rebin distributions
                                x, y_pp, y_pp_err = self.rebin_arrays(xbins, y_pp, y_pp_err)
                                _, y_AA, y_AA_err = self.rebin_arrays(xbins, y_AA, y_AA_err)

                                # Form ratio and propagate uncertainty
                                y_pp_err_fraction = np.divide(y_pp_err, y_pp)
                                y_AA_err_fraction = np.divide(y_AA_err, y_AA)
                                
                                ratio = np.divide(y_AA, y_pp)
                                ratio_err_fraction = np.sqrt(np.square(y_AA_err_fraction) + np.square(y_pp_err_fraction))
                                ratio_err = np.multiply(ratio, ratio_err_fraction)
                  
                            n = len(x)
                            xerr = np.zeros(n)
                            g = ROOT.TGraphErrors(n, x, ratio, xerr, ratio_err)
        
                        if prediction in plot_list:
                            self.observable_settings['prediction_ratio_list'].append(g)
                            label = theory['label']
                            sublabel = theory_prediction['sublabel']
                            self.observable_settings['prediction_ratio_labels'].append(f'{label}{sublabel}')

    #-------------------------------------------------------------------------------------------
    # Plot distributions in upper panel, and ratio in lower panel
    #-------------------------------------------------------------------------------------------
    def plot_distribution_and_ratio(self, plot_ratio=True, pp_label=None, logy = False):
    
        # Output filename
        output_filename = os.path.join(self.output_dir, f'h_{self.observable}.pdf')
    
        c = ROOT.TCanvas('c', 'c', 600, 650)
        c.Draw()
               
        c.cd()
        pad2_dy = 0.45
        pad1 = ROOT.TPad('myPad', 'The pad',0,pad2_dy,1,1)
        pad1.SetLeftMargin(0.2)
        pad1.SetTopMargin(0.08)
        pad1.SetRightMargin(0.04)
        pad1.SetBottomMargin(0.)
        pad1.SetTicks(0,1)
        pad1.Draw()
        if logy:
            pad1.SetLogy()
        pad1.cd()
        
        if self.observable in ['theta_g']:
            myLegend = ROOT.TLegend(0.24,0.65,0.4,0.88)
        else:
            myLegend = ROOT.TLegend(0.68,0.65,0.85,0.88)
        self.utils.setup_legend(myLegend, 0.055, sep=-0.1)
        
        # pp distribution
        if 'h_pp_sys' in self.observable_settings:
            self.observable_settings['h_pp_sys'].SetLineColor(0)
            self.observable_settings['h_pp_sys'].SetFillColor(self.data_color)
            #self.observable_settings['h_pp_sys'].SetFillColorAlpha(self.data_color, 0.3)
            self.observable_settings['h_pp_sys'].SetFillStyle(3004)
            self.observable_settings['h_pp_sys'].SetLineWidth(0)
            self.observable_settings['h_pp_sys'].SetMarkerStyle(0)
        
        if 'g_pp' in self.observable_settings:
            self.observable_settings['g_pp'].SetMarkerSize(self.marker_size)
            self.observable_settings['g_pp'].SetMarkerStyle(self.data_markers[0])
            self.observable_settings['g_pp'].SetMarkerColor(self.data_color)
            self.observable_settings['g_pp'].SetLineStyle(self.line_style)
            self.observable_settings['g_pp'].SetLineWidth(self.line_width)
            self.observable_settings['g_pp'].SetLineColor(self.data_color)
        elif 'h_pp' in self.observable_settings:
            self.observable_settings['h_pp'].SetMarkerSize(self.marker_size)
            self.observable_settings['h_pp'].SetMarkerStyle(self.data_markers[0])
            self.observable_settings['h_pp'].SetMarkerColor(self.data_color)
            self.observable_settings['h_pp'].SetLineStyle(self.line_style)
            self.observable_settings['h_pp'].SetLineWidth(self.line_width)
            self.observable_settings['h_pp'].SetLineColor(self.data_color)
        
        # AA distribution
        self.observable_settings['h_AA_stat'].SetMarkerSize(self.marker_size)
        self.observable_settings['h_AA_stat'].SetMarkerStyle(self.data_markers[1])
        self.observable_settings['h_AA_stat'].SetMarkerColor(self.data_color)
        self.observable_settings['h_AA_stat'].SetLineStyle(self.line_style)
        self.observable_settings['h_AA_stat'].SetLineWidth(self.line_width)
        self.observable_settings['h_AA_stat'].SetLineColor(self.data_color)
          
        if 'g_AA_sys' in self.observable_settings:
            self.observable_settings['g_AA_sys'].SetLineColor(0)
            self.observable_settings['g_AA_sys'].SetFillColor(self.data_color)
            self.observable_settings['g_AA_sys'].SetFillColorAlpha(self.data_color, 0.3)
            self.observable_settings['g_AA_sys'].SetFillStyle(1001)
            self.observable_settings['g_AA_sys'].SetLineWidth(0)
        if 'h_AA_sys' in self.observable_settings:
            self.observable_settings['h_AA_sys'].SetLineColor(0)
            self.observable_settings['h_AA_sys'].SetFillColor(self.data_color)
            self.observable_settings['h_AA_sys'].SetFillColorAlpha(self.data_color, 0.3)
            self.observable_settings['h_AA_sys'].SetFillStyle(1001)
            self.observable_settings['h_AA_sys'].SetLineWidth(0)
             
        myBlankHisto = ROOT.TH1F('myBlankHisto','Blank Histogram', 1, self.bins[0], self.bins[-1])
        myBlankHisto.SetNdivisions(505)
        myBlankHisto.SetXTitle(self.xtitle)
        myBlankHisto.SetYTitle(self.ytitle)
        myBlankHisto.SetMaximum(self.y_max)
        myBlankHisto.SetMinimum(2e-4) # Don't draw 0 on top panel
        myBlankHisto.GetYaxis().SetTitleSize(0.09)
        myBlankHisto.GetYaxis().SetTitleOffset(1.)
        myBlankHisto.GetYaxis().SetLabelSize(0.07)
        myBlankHisto.Draw('E')

        # Ratio
        c.cd()
        pad2 = ROOT.TPad('pad2', 'pad2', 0, 0.02, 1, pad2_dy)
        pad2.SetTopMargin(0)
        pad2.SetBottomMargin(0.21)
        pad2.SetLeftMargin(0.2)
        pad2.SetRightMargin(0.04)
        pad2.SetTicks(0,1)
        pad2.Draw()
        pad2.cd()
              
        myBlankHisto2 = myBlankHisto.Clone('myBlankHisto_C')
        myBlankHisto2.SetYTitle('#frac{Pb#font[122]{-}Pb}{pp}')
        myBlankHisto2.SetXTitle(self.xtitle)
        myBlankHisto2.GetXaxis().SetTitleSize(30)
        myBlankHisto2.GetXaxis().SetTitleFont(43)
        myBlankHisto2.GetXaxis().SetTitleOffset(1.95)
        myBlankHisto2.GetXaxis().SetLabelFont(43)
        myBlankHisto2.GetXaxis().SetLabelSize(25)
        myBlankHisto2.GetYaxis().SetTitleSize(28)
        myBlankHisto2.GetYaxis().SetTitleFont(43)
        myBlankHisto2.GetYaxis().SetTitleOffset(2.)
        myBlankHisto2.GetYaxis().SetLabelFont(43)
        myBlankHisto2.GetYaxis().SetLabelSize(25)
        myBlankHisto2.GetYaxis().SetNdivisions(505)
        
        myBlankHisto2.GetYaxis().SetRangeUser(self.y_ratio_min, self.y_ratio_max)

        if self.observable == 'girth':
            ratio_legend = ROOT.TLegend(0.45,0.65,0.6,0.95)
        elif self.observable == 'mass':
            ratio_legend = ROOT.TLegend(0.25,0.65,0.4,0.95)
        if self.observable == 'zg':
            ratio_legend = ROOT.TLegend(0.29,0.75,0.4,0.95)
        if self.observable == 'theta_g':
            ratio_legend = ROOT.TLegend(0.34,0.7,0.52,0.95)
        self.utils.setup_legend(ratio_legend, 0.07, sep=0.2)
        
        if self.observable == 'zg':
            ratio_legend2 = ROOT.TLegend(0.5,0.72,0.7,0.975)
            self.utils.setup_legend(ratio_legend2, 0.07, sep=0.2)
        elif self.observable == 'theta_g':
            ratio_legend2 = ROOT.TLegend(0.6,0.7,0.8,0.95)
            self.utils.setup_legend(ratio_legend2, 0.07, sep=0.2)

        myBlankHisto2.Draw('')
        
        if plot_ratio:
            self.observable_settings['h_ratio_stat'].SetMarkerSize(1.5)
            self.observable_settings['h_ratio_stat'].SetMarkerStyle(self.data_markers[1])
            self.observable_settings['h_ratio_stat'].SetMarkerColor(self.data_color)
            self.observable_settings['h_ratio_stat'].SetLineColor(self.data_color)
            self.observable_settings['h_ratio_stat'].SetLineWidth(self.line_width)

            if 'g_ratio_sys' in self.observable_settings:
                self.observable_settings['g_ratio_sys'].SetFillColor(self.data_color)
                self.observable_settings['g_ratio_sys'].SetFillColorAlpha(self.data_color, 0.3)
            elif 'h_ratio_sys' in self.observable_settings:
                self.observable_settings['h_ratio_sys'].SetFillColor(self.data_color)
                self.observable_settings['h_ratio_sys'].SetFillColorAlpha(self.data_color, 0.3)

        pad1.cd()
        
        # Draw theory predictions for distribution
        for i, prediction in enumerate(self.observable_settings['prediction_distribution_list']):
        
            label = self.observable_settings['prediction_distribution_labels'][i]
            color = self.theory_colors[i]
            
            prediction.SetFillColor(color)
            prediction.SetFillColorAlpha(color, self.alpha)
            prediction.SetLineColor(color)
            prediction.SetLineWidth(8)
            prediction.SetMarkerSize(0)
            prediction.SetMarkerStyle(0)
        
            if type(prediction) in [ROOT.TH1F]:
            
                prediction.Draw('E3 same')
                ratio_legend.AddEntry(prediction, label, 'L')

            elif type(prediction) in [ROOT.TGraph, ROOT.TGraphErrors, ROOT.TGraphAsymmErrors]:
                 
                if type(prediction) in [ROOT.TGraphErrors, ROOT.TGraphAsymmErrors]:
                    prediction.Draw('3 same')
                elif type(prediction) == ROOT.TGraph:
                    prediction.Draw('same')

                ratio_legend.AddEntry(prediction, label, 'L')
           
        # Draw distribution
        if 'h_pp_sys' in self.observable_settings:
            self.observable_settings['h_pp_sys'].Draw('E2 same')
        if 'g_pp' in self.observable_settings:
            self.observable_settings['g_pp'].Draw('P same')
        elif 'h_pp' in self.observable_settings:
            self.observable_settings['h_pp'].Draw('PE X0 same')
            
        if 'g_AA_sys' in self.observable_settings:
            self.observable_settings['g_AA_sys'].Draw('E2 same')
        elif 'h_AA_sys' in self.observable_settings:
            self.observable_settings['h_AA_sys'].Draw('E2 same')
        self.observable_settings['h_AA_stat'].DrawCopy('PE X0 same')
        
        pad2.cd()

        # Draw theory predictions for ratio
        for i, prediction in enumerate(self.observable_settings['prediction_ratio_list']):
        
            label = self.observable_settings['prediction_ratio_labels'][i]
            color = self.theory_colors[i]
            
            prediction.SetFillColor(color)
            prediction.SetFillColorAlpha(color, self.alpha)
            prediction.SetLineColor(color)
            prediction.SetLineWidth(8)
            prediction.SetMarkerSize(0)
            prediction.SetMarkerStyle(0)
        
            if type(prediction) in [ROOT.TH1F]:
            
                prediction.Draw('E3 same')
                if plot_ratio:
                    ratio_legend.AddEntry(prediction, label, 'L')

            elif type(prediction) in [ROOT.TGraph, ROOT.TGraphErrors, ROOT.TGraphAsymmErrors]:
                 
                if type(prediction) in [ROOT.TGraphErrors, ROOT.TGraphAsymmErrors]:
                    prediction.Draw('3 same')
                elif type(prediction) == ROOT.TGraph:
                    prediction.Draw('same')

                if self.observable in ['zg', 'theta_g'] and 'Pablos' in label:
                    ratio_legend2.AddEntry(prediction, label, 'L')
                else:
                    ratio_legend.AddEntry(prediction, label, 'L')
                    
        # Re-draw some curves
        if self.observable in ['girth']:
            self.observable_settings['prediction_ratio_list'][0].Draw('E3 same')
          
        ratio_legend.Draw()
        if self.observable in ['zg', 'theta_g']:
            ratio_legend2.Draw()
        line = ROOT.TLine(self.bins[0], 1, self.bins[-1], 1)
        line.SetLineColor(920+2)
        line.SetLineStyle(2)
        line.SetLineWidth(2)
        line.Draw()
        
        if plot_ratio:
            if 'g_ratio_sys' in self.observable_settings:
                self.observable_settings['g_ratio_sys'].Draw('E2 same')
            if 'h_ratio_sys' in self.observable_settings:
                self.observable_settings['h_ratio_sys'].Draw('E2 same')
            self.observable_settings['h_ratio_stat'].Draw('PE X0 same')
       
        pad1.cd()
        if not pp_label:
            pp_label = 'pp'
        if 'g_pp' in self.observable_settings:
            myLegend.AddEntry(self.observable_settings['g_pp'], pp_label, 'P')
        else:
            myLegend.AddEntry(self.observable_settings['h_pp'], pp_label, 'PE')
        myLegend.AddEntry(self.observable_settings['h_AA_stat'], 'Pb#font[122]{{-}}Pb {}#font[122]{{-}}{}%'.format(int(self.centrality[0]), int(self.centrality[1])), 'PE')
        if 'g_AA_sys' in self.observable_settings:
            myLegend.AddEntry(self.observable_settings['g_AA_sys'], 'Sys. uncertainty', 'f')
        elif 'h_AA_sys' in self.observable_settings:
            myLegend.AddEntry(self.observable_settings['h_AA_sys'], 'Sys. uncertainty', 'f')
       
        text_latex = ROOT.TLatex()
        text_latex.SetNDC()
        
        if self.observable in ['theta_g']:
            x = 0.52
        else:
            x = 0.25
        text_latex.SetTextSize(0.065)
        text = f'#bf{{ALICE}} #sqrt{{#it{{s_{{#it{{NN}}}}}}}} = {self.sqrts/1000.} TeV'
        text_latex.DrawLatex(x, 0.83, text)

        text = 'Charged-particle jets'
        text_latex.DrawLatex(x, 0.75, text)
        
        text = f'#it{{R}} = {self.jetR}, anti-#it{{k}}_{{T}}, |#it{{#eta}}_{{jet}}| < {0.9-self.jetR}'
        text_latex.DrawLatex(x, 0.67, text)
        
        # pt bin label
        pt_label = None
        if self.observable in ['girth', 'mass', 'zg', 'theta_g']:
            pt_label = f'{self.pt[0]} < #it{{p}}_{{T, ch jet}} < {self.pt[1]} GeV/#it{{c}}'
            text_latex.DrawLatex(x, 0.57, pt_label)
            
        if self.observable in ['zg', 'theta_g']:
            grooming_label = f'Soft Drop #it{{z}}_{{cut}}={self.zcut}, #it{{#beta}}=0'
            text_latex.DrawLatex(x, 0.47, grooming_label)
        
        myLegend.Draw()

        c.SaveAs(output_filename)
        c.Close()

    #-------------------------------------------------------------------------------------------
    def plot_ratio(self):
          
        # Draw RAA
        cname = f'c_{outputfilename}_{self.constituent_threshold}'
        c = ROOT.TCanvas(cname, cname, 600, 450)
        c.SetRightMargin(0.05);
        c.SetLeftMargin(0.15);
        c.SetTopMargin(0.05);
        c.SetBottomMargin(0.17);
        c.cd()

        leg = ROOT.TLegend(0.6,0.75,0.8,0.9)
        self.setupLegend(leg,0.03)

        # Draw experimental data
        if h_data_list:
            for i,h_data_entry in enumerate(h_data_list):
                h_data_entry[0].SetMarkerColor(self.data_color)
                h_data_entry[0].SetLineColor(self.data_color)
                h_data_entry[0].SetMarkerStyle(self.data_markers[i])
                h_data_entry[0].SetMarkerSize(2)
                leg.AddEntry(h_data_entry[0], f'ALICE {h_data_entry[1]}','Pe')

        # Draw JETSCAPE predictions
        h_RAA.SetNdivisions(505)
        h_RAA.GetXaxis().SetTitleSize(0.07)
        h_RAA.GetXaxis().SetTitleOffset(1.1)
        h_RAA.SetXTitle(xtitle)
        h_RAA.GetXaxis().SetLabelSize(0.05)
        h_RAA.GetYaxis().SetLabelSize(0.05)
        if raa_type in ['hadron', 'jet']:
            h_RAA.SetYTitle("#it{R}_{AA}")
            h_RAA.GetYaxis().SetTitleSize(0.07)
            h_RAA.GetYaxis().SetTitleOffset(1.)
        else:
            h_RAA.SetYTitle("#frac{Pb-Pb}{pp}")
            h_RAA.GetYaxis().SetTitleSize(0.06)
            h_RAA.GetYaxis().SetTitleOffset(1.1)
        h_RAA.GetYaxis().SetRangeUser(0,ymax)
        h_RAA.Draw('PE same')

        h_RAA.SetMarkerColorAlpha(self.theory_colors[0], self.alpha)
        h_RAA.SetLineColorAlpha(self.theory_colors[0], self.alpha)
        h_RAA.SetLineWidth(self.line_width)
        h_RAA.SetMarkerStyle(self.data_markers[0])
        leg.AddEntry(h_RAA,f'{mc_centralities[0]}%','P')

        if h_data_list:
            for h_data_entry in h_data_list:
                h_data_entry[0].Draw('PE same')
        h_RAA.Draw('PE same')

        leg.Draw('same')

        line = ROOT.TLine(h_RAA.GetXaxis().GetXmin(),1,h_RAA.GetXaxis().GetXmax(),1)
        line.SetLineColor(1)
        line.SetLineStyle(2)
        line.Draw('same')

        # # # # # # # # # # # # # # # # # # # # # # # #
        # text
        # # # # # # # # # # # # # # # # # # # # # # # #
        ymax = 0.9
        dy = 0.05
        x = 0.2
        size = 0.04
        system0 = ROOT.TLatex(x,ymax,'#bf{JETSCAPE}')
        system0.SetNDC()
        system0.SetTextSize(size)
        system0.Draw()

        system1 = ROOT.TLatex(x,ymax-dy,'MATTER+LBT')
        system1.SetNDC()
        system1.SetTextSize(size)
        system1.Draw()

        system2 = ROOT.TLatex(x,ymax-2*dy,f'Pb-Pb  #sqrt{{#it{{s}}}} = {self.sqrts/1000.} TeV')
        system2.SetNDC()
        system2.SetTextSize(size)
        system2.Draw()

        if raa_type == 'hadron':
            system3 = ROOT.TLatex(x,ymax-3*dy, f'Charged particles  |#eta| < {eta_cut}')
        else:
            system3 = ROOT.TLatex(x,ymax-3*dy, f'AKT  #it{{R}} = {R}  |#eta_{{jet}}| < {eta_cut}')
        system3.SetNDC()
        system3.SetTextSize(size)
        system3.Draw()
        
        if pt_label:
            system4 = ROOT.TLatex(x,ymax-4.4*dy, pt_label)
            system4.SetNDC()
            system4.SetTextSize(size)
            system4.Draw()
        
        outdir = os.path.join(self.output_dir, self.output_dir_suffix)
        output_filename = os.path.join(outdir, outputfilename)
        c.SaveAs(output_filename)

    #---------------------------------------------------------------
    # Initialize theory predictions
    #---------------------------------------------------------------
    def init_theory(self, result):
    
        #----------------------------------------------------------
        # Factorization (Ringer, Sato)
        if 'medium_jet_functions' in result:
            zr = np.array(result['medium_jet_functions']['zr'])
            y_vac = np.array(result['medium_jet_functions']['y_vac'])
            y_med = np.array(result['medium_jet_functions']['y_med'])
            
            # Compute integral (excluding the last measurement bin)
            # Note: already normalized by bin widths
            integral_medium_jet_functions_vac = np.sum(y_vac)
            integral_medium_jet_functions_med = np.sum(y_med)
            
            # Normalize to integral of data (excluding the last measurement bin)
            scale_factor_pp = self.integral_pp_truncated / integral_medium_jet_functions_vac
            scale_factor_AA = self.integral_AA_truncated / integral_medium_jet_functions_med
            y_med_normalized = scale_factor_AA*y_med
            y_vac_normalized = scale_factor_pp*y_vac

            # Form ratio
            ratio = np.divide(y_med_normalized, y_vac_normalized)
                  
            # Draw as a line, since the uncertainties are too small
            n = len(zr)
            g = ROOT.TGraph(n, zr, ratio)
            
            # Add to theory list
            self.prediction_list.append(g)
            self.label_list.append('Medium jet functions')
            self.sublabel_list.append('')

    #-------------------------------------------------------------------------------------------
    def plot_hadron_raa(self):
    
        # Get experimental data
        h_data_list = []
        f = ROOT.TFile(self.hadron_observables['pt_alice']['hepdata'], 'READ')
        dir = f.Get('Table 8')
        h_data_0_5 = dir.Get('Graph1D_y1')
        h_data_5_10 = dir.Get('Graph1D_y2')
        h_data_list.append([h_data_0_5, '0-5%'])
        #h_data_list.append([h_data_5_10, '5-10%'])
        f.Close()

        # Plot
        self.plot_raa(raa_type='hadron',
                      hname = 'h_hadron_pt_aliceScaled',
                      h_data_list=h_data_list,
                      eta_cut=np.round(self.hadron_observables['pt_alice']['eta_cut'], decimals=1),
                      data_centralities=['0-5%'],
                      mc_centralities=[f'{self.min_cent}-{self.max_cent}'],
                      xtitle="#it{p}_{T} (GeV/#it{c})",
                      ytitle = '#frac{d^{2}N}{d#it{p}_{T}d#it{#eta}} #left[(GeV/c)^{-1}#right]',
                      ymax=1.8,
                      outputfilename=f'h_hadron_RAA_alice_{self.sqrts}_{self.min_cent}-{self.max_cent}{self.file_format}')

    #-------------------------------------------------------------------------------------------
    def plot_jet_raa(self):

        for R in [0.2, 0.4]:
        
            # Get experimental data
            h_data_list = []
            if R==0.2:
                f = ROOT.TFile(self.inclusive_jet_observables['pt_alice']['hepdata_0_10_R02'], 'READ')
                dir = f.Get('Table 30')
            elif R==0.4:
                f = ROOT.TFile(self.inclusive_jet_observables['pt_alice']['hepdata_0_10_R04'], 'READ')
                dir = f.Get('Table 31')
            h_data = dir.Get('Graph1D_y1')
            h_data_list.append([h_data, '0-10%'])
            f.Close()
            
            # Plot
            self.plot_raa(raa_type='jet',
                          hname = f'h_jet_pt_alice_R{R}{self.suffix}Scaled',
                          h_data_list=h_data_list,
                          eta_cut=np.round(self.inclusive_jet_observables['pt_alice']['eta_cut_R']-R, decimals=1),
                          data_centralities=['0-10'],
                          mc_centralities=[f'{self.min_cent}-{self.max_cent}'],
                          xtitle="#it{p}_{T,jet} (GeV/#it{c})",
                          ytitle = '#frac{d^{2}N}{d#it{p}_{T}d#it{#eta}} #left[(GeV/c)^{-1}#right]',
                          ymax=1.8,
                          outputfilename=f'h_jet_RAA_alice_R{R}_{self.sqrts}_{self.min_cent}-{self.max_cent}{self.file_format}',
                          R=R)
                      
    #-------------------------------------------------------------------------------------------
    def plot_semi_inclusive_chjet_IAA(self):
            
        # Get experimental data
        R=0.4
        c_ref = 0.96 # R02: 0.99, R04: 0.96, R05: 0.93
        h_data_list = []
        if R == 0.2:
            f = ROOT.TFile(self.semi_inclusive_chjet_observables['hjet_alice']['hepdata_IAA_276_R02'], 'READ')
            dir = f.Get('Table 32')
        elif R == 0.4:
            f = ROOT.TFile(self.semi_inclusive_chjet_observables['hjet_alice']['hepdata_IAA_276_R04'], 'READ')
            dir = f.Get('Table 33')
        elif R == 0.5:
            f = ROOT.TFile(self.semi_inclusive_chjet_observables['hjet_alice']['hepdata_IAA_276_R05'], 'READ')
            dir = f.Get('Table 34')
        h_data = dir.Get('Graph1D_y1')
        h_data_list.append([h_data, '0-10%'])
        f.Close()
        
        if self.sqrts == 2760:
            hname_ntrigger = f'h_semi_inclusive_chjet_hjet_ntrigger_alice_R{R}{self.suffix}Scaled'
            hname_high = f'h_semi_inclusive_chjet_IAA_highTrigger_alice_R{R}_276{self.suffix}Scaled'
            hname_low = f'h_semi_inclusive_chjet_IAA_lowTrigger_alice_R{R}_276{self.suffix}Scaled'
            
            # Get JETSCAPE pp prediction
            filename_pp = os.path.join(self.output_dir, f'{self.dir_pp}/AnalysisResultsFinal.root')
            f_pp = ROOT.TFile(filename_pp, 'READ')
            h_pp_ntrigger = f_pp.Get(hname_ntrigger)
            h_pp_ntrigger.SetDirectory(0)
            h_pp_high = f_pp.Get(hname_high)
            h_pp_high.SetDirectory(0)
            h_pp_low = f_pp.Get(hname_low)
            h_pp_low.SetDirectory(0)
            f_pp.Close()
            
            # Get JETSCAPE AA prediction
            filename = os.path.join(self.output_dir, f'{self.dir_AA}/AnalysisResultsFinal.root')
            f_AA = ROOT.TFile(filename, 'READ')
            h_AA_ntrigger = f_AA.Get(hname_ntrigger)
            h_AA_ntrigger.SetDirectory(0)
            h_AA_high = f_AA.Get(hname_high)
            h_AA_high.SetDirectory(0)
            h_AA_low = f_AA.Get(hname_low)
            h_AA_low.SetDirectory(0)
            f_AA.Close()
        elif self.sqrts == 5020:
            hname_ntrigger = f'h_semi_inclusive_chjet_hjet_ntrigger_alice_R{R}{self.suffix}Scaled'
            hname_high = f'h_semi_inclusive_chjet_IAA_dphi_highTrigger_alice_R{R}_502{self.suffix}Scaled'
            hname_low = f'h_semi_inclusive_chjet_IAA_dphi_lowTrigger_alice_R{R}_502{self.suffix}Scaled'
            
            # Get JETSCAPE pp prediction
            filename_pp = os.path.join(self.output_dir, f'{self.dir_pp}/AnalysisResultsFinal.root')
            f_pp = ROOT.TFile(filename_pp, 'READ')
            h_pp_ntrigger = f_pp.Get(hname_ntrigger)
            h_pp_ntrigger.SetDirectory(0)
            
            h_pp_high2d = f_pp.Get(hname_high)
            h_pp_high2d.SetDirectory(0)
            h_pp_high2d.GetYaxis().SetRangeUser(np.pi-0.6, np.pi)
            h_pp_high = h_pp_high2d.ProjectionX()
            h_pp_high.Rebin(20)
            h_pp_high.SetDirectory(0)
                        
            h_pp_low2d = f_pp.Get(hname_low)
            h_pp_low2d.SetDirectory(0)
            h_pp_low2d.GetYaxis().SetRangeUser(np.pi-0.6, np.pi)
            h_pp_low = h_pp_low2d.ProjectionX()
            h_pp_low.Rebin(20)
            h_pp_low.SetDirectory(0)
            f_pp.Close()

            # Get JETSCAPE AA prediction
            filename = os.path.join(self.output_dir, f'{self.dir_AA}/AnalysisResultsFinal.root')
            f_AA = ROOT.TFile(filename, 'READ')
            h_AA_ntrigger = f_AA.Get(hname_ntrigger)
            h_AA_ntrigger.SetDirectory(0)
            
            h_AA_high2d = f_AA.Get(hname_high)
            h_AA_high2d.SetDirectory(0)
            h_AA_high2d.GetYaxis().SetRangeUser(np.pi-0.6, np.pi)
            h_AA_high = h_AA_high2d.ProjectionX()
            h_AA_high.Rebin(20)
            h_AA_high.SetDirectory(0)
              
            h_AA_low2d = f_AA.Get(hname_low)
            h_AA_low2d.SetDirectory(0)
            h_AA_low2d.GetYaxis().SetRangeUser(np.pi-0.6, np.pi)
            h_AA_low = h_AA_low2d.ProjectionX()
            h_AA_low.Rebin(20)
            h_AA_low.SetDirectory(0)
            f_AA.Close()
        
        # Delta recoil
        n_trig_high_pp = h_pp_ntrigger.GetBinContent(h_pp_ntrigger.FindBin(30.))
        n_trig_low_pp = h_pp_ntrigger.GetBinContent(h_pp_ntrigger.FindBin(8.5))
        n_trig_high_AA = h_AA_ntrigger.GetBinContent(h_AA_ntrigger.FindBin(30.))
        n_trig_low_AA = h_AA_ntrigger.GetBinContent(h_AA_ntrigger.FindBin(8.5))
        print(f'n_trig_high_pp: {n_trig_high_pp}')
        print(f'n_trig_low_pp: {n_trig_low_pp}')
        print(f'n_trig_high_AA: {n_trig_high_AA}')
        print(f'n_trig_low_AA: {n_trig_low_AA}')

        h_pp_high.Scale(1./n_trig_high_pp, 'width')
        h_pp_low.Scale(1./n_trig_low_pp, 'width')
        h_AA_high.Scale(1./n_trig_high_AA, 'width')
        h_AA_low.Scale(1./n_trig_low_AA, 'width')
        
        h_delta_recoil_pp = h_pp_high.Clone('h_delta_recoil_pp')
        h_delta_recoil_pp.Add(h_pp_low, -1)
        
        h_delta_recoil_AA = h_AA_high.Clone('h_delta_recoil_AA')
        h_delta_recoil_AA.Add(h_AA_low, -1*c_ref)
 
        # Plot
        xtitle="#it{p}_{T,ch} (GeV/#it{c})"
        self.plot_raa_ratio(raa_type='hjet_IAA',
                            h_pp=h_delta_recoil_pp,
                            h_AA=h_delta_recoil_AA,
                            h_data_list=h_data_list,
                            eta_cut=np.round(self.inclusive_chjet_observables['eta_cut_alice_R']-R, decimals=1),
                            data_centralities=['0-10'],
                            mc_centralities=[f'{self.min_cent}-{self.max_cent}'],
                            xtitle=xtitle,
                            ytitle = '#Delta_{recoil}',
                            ymax=1.8,
                            outputfilename=f'h_semi_inclusive_chjet_IAA_alice_R{R}_{self.sqrts}_{self.min_cent}-{self.max_cent}{self.file_format}',
                            R=R)

    #-------------------------------------------------------------------------------------------
    def plot_semi_inclusive_chjet_dphi(self):
            
        # Get experimental data
        R=0.4
        c_ref = 0.96 # R02: 0.99, R04: 0.96, R05: 0.93
        h_data_list = []
        f = ROOT.TFile(self.semi_inclusive_chjet_observables['hjet_alice']['hepdata_dphi_276'], 'READ')
        dir = f.Get('Table 37')
        h_data = dir.Get('Graph1D_y1')
        h_data_list.append([h_data, 'pp'])
        f.Close()
        
        hname_ntrigger = f'h_semi_inclusive_chjet_hjet_ntrigger_alice_R{R}{self.suffix}Scaled'
        hname_high = f'h_semi_inclusive_chjet_dphi_highTrigger_alice_R{R}_276{self.suffix}Scaled'
        hname_low = f'h_semi_inclusive_chjet_dphi_lowTrigger_alice_R{R}_276{self.suffix}Scaled'
        
        # Get JETSCAPE pp prediction
        filename_pp = os.path.join(self.output_dir, f'{self.dir_pp}/AnalysisResultsFinal.root')
        f_pp = ROOT.TFile(filename_pp, 'READ')
        h_pp_ntrigger = f_pp.Get(hname_ntrigger)
        h_pp_ntrigger.SetDirectory(0)
        h_pp_high = f_pp.Get(hname_high)
        h_pp_high.SetDirectory(0)
        h_pp_low = f_pp.Get(hname_low)
        h_pp_low.SetDirectory(0)
        f_pp.Close()
        
        # Get JETSCAPE AA prediction
        filename = os.path.join(self.output_dir, f'{self.dir_AA}/AnalysisResultsFinal.root')
        f_AA = ROOT.TFile(filename, 'READ')
        h_AA_ntrigger = f_AA.Get(hname_ntrigger)
        h_AA_ntrigger.SetDirectory(0)
        h_AA_high = f_AA.Get(hname_high)
        h_AA_high.SetDirectory(0)
        h_AA_low = f_AA.Get(hname_low)
        h_AA_low.SetDirectory(0)
        f_AA.Close()
        
        # Delta recoil
        n_trig_high_pp = h_pp_ntrigger.GetBinContent(h_pp_ntrigger.FindBin(30.))
        n_trig_low_pp = h_pp_ntrigger.GetBinContent(h_pp_ntrigger.FindBin(8.5))
        n_trig_high_AA = h_AA_ntrigger.GetBinContent(h_AA_ntrigger.FindBin(30.))
        n_trig_low_AA = h_AA_ntrigger.GetBinContent(h_AA_ntrigger.FindBin(8.5))
        print(f'n_trig_high_pp: {n_trig_high_pp}')
        print(f'n_trig_low_pp: {n_trig_low_pp}')
        print(f'n_trig_high_AA: {n_trig_high_AA}')
        print(f'n_trig_low_AA: {n_trig_low_AA}')

        h_pp_high.Scale(1./n_trig_high_pp, 'width')
        h_pp_low.Scale(1./n_trig_low_pp, 'width')
        h_AA_high.Scale(1./n_trig_high_AA, 'width')
        h_AA_low.Scale(1./n_trig_low_AA, 'width')
        
        h_delta_Phi_pp = h_pp_high.Clone('h_delta_Phi_pp')
        h_delta_Phi_pp.Add(h_pp_low, -1)
        
        h_delta_Phi_AA = h_AA_high.Clone('h_delta_Phi_AA')
        h_delta_Phi_AA.Add(h_AA_low, -1*c_ref)
 
        # Plot
        xtitle="#Delta #it{#varphi}"
        self.plot_raa_ratio(raa_type='hjet_dphi',
                            h_pp=h_delta_Phi_pp,
                            h_AA=h_delta_Phi_AA,
                            h_data_list=h_data_list,
                            eta_cut=np.round(self.inclusive_chjet_observables['eta_cut_alice_R']-R, decimals=1),
                            data_centralities=['0-10'],
                            mc_centralities=[f'{self.min_cent}-{self.max_cent}'],
                            xtitle=xtitle,
                            ytitle = '#Phi(#Delta#it{#varphi})',
                            ymax=0.1,
                            outputfilename=f'h_semi_inclusive_chjet_dphi_alice_R{R}_{self.sqrts}_{self.min_cent}-{self.max_cent}{self.file_format}',
                            R=R)
                            
    #---------------------------------------------------------------
    # Rebin numpy arrays (xbins,y) representing a histogram,
    # where x represents the bin edges, and y the bin content
    #
    # Return (x_rebinned, y_rebinned) where x_rebinned is the bin centers
    #
    # I don't find any easy way to do this...so I
    #        construct TH1, rebin, and then extract array)
    #----------------------------------------------------------------------
    def rebin_arrays(self, x, y, yerr):

        x_array = array('d', x)
        h = ROOT.TH1F('h', 'h', len(x)-1, x_array)
        for i, content in enumerate(y):
            h.SetBinContent(i+1, y[i])
            h.SetBinError(i+1, yerr[i])
        h_rebinned = h.Rebin(len(self.bin_array)-1, 'h_rebinned', self.bin_array)
        x_rebinned = []
        y_rebinned = []
        y_err_rebinned = []
        for i in range(len(self.bin_array)-1):
            x_rebinned.append(h_rebinned.GetBinCenter(i+1))
            y_rebinned.append(h_rebinned.GetBinContent(i+1))
            y_err_rebinned.append(h_rebinned.GetBinError(i+1))
          
        return (np.array(x_rebinned), np.array(y_rebinned), np.array(y_err_rebinned))
  
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
if __name__ == '__main__':
    print('Executing plot_results_TG3.py...')
    print('')

    # Define arguments
    parser = argparse.ArgumentParser(description='Plot TG3 observables')
    parser.add_argument(
        '-c',
        '--configFile',
        action='store',
        type=str,
        metavar='configFile',
        default='TG3.yaml',
        help='Config file'
    )
    parser.add_argument(
        '-o',
        '--outputDir',
        action='store',
        type=str,
        metavar='outputDir',
        default='.',
        help='Output directory for output to be written to'
    )

    # Parse the arguments
    args = parser.parse_args()

    analysis = PlotResults(config_file=args.configFile, output_dir=args.outputDir)
    analysis.plot_results()