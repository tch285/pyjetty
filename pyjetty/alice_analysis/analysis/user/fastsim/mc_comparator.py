#!/usr/bin/env python3

'''
For comparing MC simulation results between det & tru level,
to the fast simulation, or to the data.
Ezra Lesser (elesser@berkeley.edu)
'''

from __future__ import print_function

import os
import sys
import argparse
import yaml
import numpy as np
from array import *
import ROOT

from pyjetty.alice_analysis.analysis.base import analysis_base, analysis_utils


# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)


################################################################
class mc_comparator(analysis_base.AnalysisBase):

    ################################################################
    # Analysis class constructor
    def __init__(self, inputFileMC, inputFileFS, inputFileData, configFile,
                 outputDir, imageFormat, **kwargs):
        super(mc_comparator, self).__init__(inputFileFS, inputFileMC, configFile, 
                                               outputDir, imageFormat, **kwargs)

        # Load configuration and open files
        with open(args.configFile, 'r') as stream:
            self.config = yaml.safe_load(stream)

        self.subconfig_list = [name for name in list(self.config["ang"].keys()) if 'config' in name ]

        self.jetR_list = self.config['jetR']
        self.beta_list = self.config['betas']

        self.pt_bins = self.config["ang"]["common_settings"]["pt_bins_reported"]

        self.f_mc = ROOT.TFile(args.inputFileMC, 'READ')

        self.do_fs = False
        if len(inputFileFS):
            if not os.path.exists(args.inputFileFS):
                print('File \"{0}\" does not exist! Exiting!'.format(args.inputFileFS))
                sys.exit(0)
            self.do_fs = True
            self.f_fs = ROOT.TFile(args.inputFileFS, 'READ')

        self.do_data = False
        if len(inputFileData):
            if not os.path.exists(inputFileData):
                print('File \"{0}\" does not exist! Exiting!'.format(inputFileData))
                sys.exit(0)
            self.do_data = True
            self.f_data = ROOT.TFile(args.inputFileData, 'READ')

        self.outdir = args.outputDir
        self.image_format = args.imageFormat

    ################################################################
    # Create comparison plots
    def mc_comparator(self):

        for jetR in self.jetR_list:
            for subconfig_name in self.subconfig_list:
                subconfig = self.config["ang"][subconfig_name]
                beta = subconfig["beta"]
                label = "R%s_%s" % (str(jetR), str(beta))

                # Get histograms from file for this configuration
                name = "hAng_JetPt_det_%sScaled" % label
                h_mc = self.f_mc.Get(name)
                if self.do_fs:
                    h_fs = self.f_fs.Get(name)
                h_tru = self.f_mc.Get("hAng_JetPt_tru_%sScaled" % label)
                if self.do_data:
                    h_data = self.f_data.Get("h_ang_JetPt_%s" % label)

                for i in range(0, len(self.pt_bins)-1):

                    # Get relevant info from configuration file
                    max_edge = subconfig["obs_max_reported"][i]
                    step = round(max_edge/20, 2)  # Approx 20 bins with min width 0.01
                    ang_bins = np.arange(0, max_edge, step)

                    # Rebin histograms for comparions
                    h_mc_rebinned = self.utils.rebin_data(
                        h_mc, h_mc.GetName()+'_mc', len(self.pt_bins)-1,
                        array('d', self.pt_bins), len(ang_bins)-1, array('d', ang_bins))
                    h_tru_rebinned = self.utils.rebin_data(
                        h_tru, h_tru.GetName()+'_fs', len(self.pt_bins)-1,
                        array('d', self.pt_bins), len(ang_bins)-1, array('d', ang_bins))

                    if self.do_fs:
                        h_fs_rebinned = self.utils.rebin_data(
                            h_fs, h_fs.GetName()+'_fs', len(self.pt_bins)-1,
                            array('d', self.pt_bins), len(ang_bins)-1, array('d', ang_bins))

                    if self.do_data:
                        h_data_rebinned = self.utils.rebin_data(
                            h_data, h_data.GetName()+'_fs', len(self.pt_bins)-1,
                            array('d', self.pt_bins), len(ang_bins)-1, array('d', ang_bins))

                    title = "MCcomp_%s_pT_%i-%i" % (label, self.pt_bins[i], self.pt_bins[i+1])

                    # Canvas to accompany all plots
                    c = ROOT.TCanvas('c'+title, 'c'+title, 600, 800)
                    c.cd()

                    # "Pad" to contain the top histograms
                    topPad = ROOT.TPad('topPad', 'topPad', 0, 0.3, 1, 1.0)
                    topPad.SetLeftMargin(0.15)
                    topPad.SetTopMargin(0.1)
                    topPad.SetRightMargin(0.05)
                    topPad.SetBottomMargin(0)
                    topPad.Draw()
                    topPad.cd()

                    # Draw histograms to top pad
                    h_tru_projy = h_tru_rebinned.ProjectionY(title+'_tru', i, i+1)
                    h_tru_projy.SetMarkerSize(1.5)
                    h_tru_projy.SetMarkerStyle(21)
                    h_tru_projy.SetMarkerColor(1)
                    h_tru_projy.SetStats(0)
                    h_tru_projy.Scale(1. / h_tru_projy.GetEntries(), "width")
                    if self.do_fs:
                        h_tru_projy.SetTitle("Angularity distribution for full and fast MC")
                    else:
                        h_tru_projy.SetTitle("Angularity distribution: true vs. det level")
                    h_tru_projy.SetYTitle('#frac{1}{#it{N}_{jet}} #frac{d#it{N}}{d#it{#lambda}}')
                    h_tru_projy.SetMinimum(1e-4)
                    h_tru_projy.SetMaximum(1.5*h_tru_projy.GetMaximum())
                    h_tru_projy.Draw("C same E")
                    h_mc_projy = h_mc_rebinned.ProjectionY(title+'_mc', i, i+1)
                    h_mc_projy.SetMarkerSize(1.5)
                    h_mc_projy.SetMarkerStyle(20)
                    h_mc_projy.SetMarkerColor(2)
                    h_mc_projy.SetStats(0)
                    h_mc_projy.Scale(1. / h_mc_projy.GetEntries(), "width")
                    h_mc_projy.SetMinimum(1e-4)
                    h_mc_projy.SetMaximum(1.5*h_mc_projy.GetMaximum())
                    h_mc_projy.Draw("C same E")
                    if self.do_data:
                        h_data_projy = h_data_rebinned.ProjectionY(title+'_data', i, i+1)
                        h_data_projy.SetMarkerSize(1.5)
                        h_data_projy.SetMarkerStyle(22)
                        h_data_projy.SetMarkerColor(4)
                        h_data_projy.SetStats(0)
                        h_data_projy.Scale(1. / h_data_projy.GetEntries(), "width")
                        h_data_projy.SetMinimum(1e-4)
                        h_data_projy.SetMaximum(1.5*h_data_projy.GetMaximum())
                        h_data_projy.Draw("C same E")
                    if self.do_fs:
                        h_fs_projy = h_fs_rebinned.ProjectionY(title+'_fs', i, i+1)
                        h_fs_projy.SetMarkerSize(1.5)
                        h_fs_projy.SetMarkerStyle(21)
                        h_fs_projy.SetMarkerColor(4)
                        h_fs_projy.SetStats(0)
                        h_fs_projy.Scale(1. / h_fs_projy.GetEntries(), "width")
                        h_fs_projy.SetMinimum(1e-4)
                        h_fs_projy.SetMaximum(1.5*h_fs_projy.GetMaximum())
                        h_fs_projy.Draw("C same E")

                    # Create and format the legend
                    leg = ROOT.TLegend(0.69, 0.69, 0.88, 0.88)
                    #leg.SetHeader(title, 'C')
                    self.utils.setup_legend(leg, 0.035)
                    if self.do_fs:
                        leg.AddEntry(h_tru_projy, "Full MC, truth", "lep")
                        leg.AddEntry(h_mc_projy, "Full MC, det level", "lep")
                        leg.AddEntry(h_fs_projy, "Fast simulation", "lep")
                    else:
                        leg.AddEntry(h_tru_projy, "MC truth", "lep")
                        leg.AddEntry(h_mc_projy, "MC det. level", "lep")
                    if self.do_data:
                        leg.AddEntry(h_data_projy, "Data (raw)", "lep")
                    leg.Draw()

                    # Text overlaid on the plot
                    text_latex = ROOT.TLatex()
                    text_latex.SetNDC()
                    text = "%i < #it{p}_{T}^{jet, ch} < %s GeV/#it{c}" % \
                           (self.pt_bins[i], self.pt_bins[i+1])
                    text_latex.DrawLatex(0.22, 0.82, text)
                    text = "#it{R} = " + str(jetR) + "  #it{#beta} = " + str(beta)
                    text_latex.DrawLatex(0.22, 0.72, text)

                    # Create bottom pad for ratio plot
                    c.cd()
                    botPad = ROOT.TPad("botPad", "botPad", 0, 0.05, 1, 0.3)
                    botPad.SetTopMargin(0)
                    botPad.SetBottomMargin(0.25)
                    botPad.SetLeftMargin(0.15)
                    botPad.SetRightMargin(0.05)
                    botPad.Draw()
                    botPad.cd()

                    # Create and format the ratio plot
                    h_ratio = h_mc_projy.Clone()
                    if self.do_fs:
                        h_ratio.Divide(h_fs_projy)
                    else:
                        h_ratio.Divide(h_tru_projy)
                    h_ratio.SetTitle("")#'h'+title)
                    h_ratio.GetXaxis().SetTitleSize(30)
                    h_ratio.GetXaxis().SetTitleFont(43)
                    h_ratio.GetXaxis().SetTitleOffset(4.)
                    h_ratio.GetXaxis().SetLabelFont(43)
                    h_ratio.GetXaxis().SetLabelSize(20)
                    h_ratio.SetXTitle('#lambda_{#beta=%s}' % beta)
                    h_ratio.GetYaxis().SetTitleSize(20)
                    h_ratio.GetYaxis().SetTitleFont(43)
                    h_ratio.GetYaxis().SetTitleOffset(2.2)
                    h_ratio.GetYaxis().SetLabelFont(43)
                    h_ratio.GetYaxis().SetLabelSize(20)
                    if self.do_fs:
                        h_ratio.SetYTitle("full MC det / fast sim")
                    else:
                        h_ratio.SetYTitle("MC det / truth") 
                    h_ratio.SetMarkerColor(51)
                    h_ratio.SetStats(0)
                    h_ratio.SetMinimum(0.75)
                    h_ratio.SetMaximum(1.199)
                    h_ratio.LabelsDeflate("Y")
                    h_ratio.Draw('E')

                    # Line at 1 for visual reference
                    line = ROOT.TLine(0, 1, ang_bins[-1], 1)
                    line.Draw()

                    outf_name = os.path.join(self.outdir, title + self.image_format)
                    c.SaveAs(outf_name)
                    #input("Press Enter to continue...")

        # Close files after reading / writing
        self.close_files()

    ################################################################
    def close_files(self):
        self.f_mc.Close()
        if self.do_fs:
            self.f_fs.Close()
        if self.do_data:
            self.f_data.Close()

################################################################
if __name__ == '__main__':
    # Parse and check arguments
    parser = argparse.ArgumentParser(description='Compare full and fast simulations')
    parser.add_argument('-m', '--inputFileMC', action='store',
                        type=str, metavar='inputFileMC', default='AnalysisResults.root',
                        help='Path of ROOT file containing full simulation TTrees')
    parser.add_argument('-f', '--inputFileFS', action='store',
                        type=str, metavar='inputFileFS', default='',
                        help='Path of ROOT file containing fast simulation TTrees')
    parser.add_argument('-d', '--inputFileData', action='store',
                        type=str, metavar='inputFileData', default='',
                        help='Path of ROOT file containing data TTrees')
    parser.add_argument('-c', '--configFile', action='store', type=str, metavar='configFile',
                        default='config/analysis_config.yaml',
                        help="Path of config file for analysis")
    parser.add_argument('-o', '--outputDir', action='store',
                        type=str, metavar='outputDir', default='./TestOutput',
                        help='Output directory for output to be written to')
    parser.add_argument('-i', '--imageFormat', action='store',
                        type=str, metavar='imageFormat', default='.pdf',
                        help='Image format to save plots in, e.g. \".pdf\" or \".png\"')

    # Parse the arguments
    args = parser.parse_args()

    print('Configuring...')
    print('inputFileMC: "{0}"'.format(args.inputFileMC))
    print('inputFileFS: "{0}"'.format(args.inputFileFS))
    print('configFile: "{0}"'.format(args.configFile))
    print('ouputDir: "{0}"'.format(args.outputDir))

    # If invalid inputFile is given, exit
    if not os.path.exists(args.inputFileMC):
        print('File \"{0}\" does not exist! Exiting!'.format(args.inputFileMC))
        sys.exit(0)

    # Initiate class and start analysis
    analysis = mc_comparator(args.inputFileMC, args.inputFileFS, args.inputFileData,
                                args.configFile, args.outputDir, args.imageFormat) 
    analysis.mc_comparator()
