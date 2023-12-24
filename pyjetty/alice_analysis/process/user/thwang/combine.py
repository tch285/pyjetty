#!/usr/bin/env python

import ROOT
import numpy as np
import argparse
import array
import os
import yaml
from matplotlib import pyplot as plt

ROOT.gROOT.SetBatch(True)
ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()

def linbins(xmin, xmax, nbins):
    arr = np.linspace(xmin, xmax, nbins+1)
    # arr = array.array('f', lspace)
    centers = (arr[:-1] + arr[1:]) / 2
    return arr, centers
def linbins_old(xmin, xmax, nbins):
    lspace = np.linspace(xmin, xmax, nbins+1)
    arr = array.array('f', lspace)
    return arr

def logbins(xmin, xmax, nbins):
    arr = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
    centers = np.sqrt(arr[:-1] * arr[1:])
    lowerr = centers - arr[:-1]
    uperr = arr[1:] - centers
    bin_widths = arr[1:] - arr[:-1]
    return arr, centers, lowerr, uperr, bin_widths
def logbins_old(xmin, xmax, nbins):
    lspace = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
    arr = array.array('f', lspace)
    return arr

class Combiner(object):
    def __init__(self, job_id, config_file, data_dir, output_dir, root_data_name):
        with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)
        if not os.path.exists(f"{data_dir}/{job_id}"):
            raise FileNotFoundError(f"Data directory '{data_dir}/{job_id}' not found.")
        if not os.path.isfile(f"{data_dir}/{job_id}/1/1/{root_data_name}"):
            raise FileNotFoundError(f"No ROOT files named '{root_data_name}' were found.")
        if not os.path.exists(f"{output_dir}/{job_id}"):
            os.makedirs(f"{output_dir}/{job_id}")
        self.data_dir = data_dir
        self.output_dir = f"{output_dir}/{job_id}"
        self.job_id = job_id
        self.root_data_name = root_data_name

        self.RL_min, self.RL_max, self.RL_nbins = config["RL_binning"]
        self.pT_min, self.pT_max, self.pT_nbins = config["pT_binning"]

        # particle level - ALICE tracking restriction
        self.max_eta_hadron = config["eta_max"] # used to be explicitly 0.9
        self.min_det_trk_pT = config["min_det_trk_pT"] # 0.15
        self.min_jet_trk_pT = config["min_jet_trk_pT"] # 1
        self.min_jet_pT = config["min_jet_pT"] # 5
        self.leading_jet_const_pT = config["leading_jet_const_pT"]

        # ENC settings
        self.npoint = config["npoint"]
        self.npower = config["npoint"]


        # manually setting these
        self.ipoint = self.npoint
        self.jet_level = "ch"
        self.jetR = 0.4
        self.R_label = str(self.jetR).replace('.', '') + 'Scaled'
        
    def combine(self, ptlow = -1, pthigh = -1, root_out = "combined"):
        # need bin num ptlow+1 to pthigh for given range
        self.ptlow = ptlow
        self.pthigh = pthigh
        pT_bin_min = round(ptlow) + 1
        pT_bin_max = self.pT_nbins if pthigh == -1 else round(pthigh)
        self.pt_bins, _ = linbins(self.pT_min, self.pT_max, self.pT_nbins)
        self.RL_bins, self.RL_centers, self.xerrlow, self.xerrup, self.bin_widths = logbins(self.RL_min, self.RL_max, self.RL_nbins)
        hist = ROOT.TH2F(f"combined_{ptlow}_{pthigh}", "{ptlow} to {pthigh}", self.pT_nbins, self.pt_bins, self.RL_nbins, self.RL_bins)
        Njet_scaled_total = 0
        for root_dir, subdirs, _ in os.walk(f"{self.data_dir}/{self.job_id}"):
            if not subdirs:
                # Njets = 0
                # we are in a lowest subdirectory, and should now extract ROOT file and text file
                with ROOT.TFile(f"{root_dir}/{self.root_data_name}", "READ") as infile:
                    hin = infile.Get(f'h_ENC{self.ipoint}_JetPt_{self.jet_level}_R{self.R_label}_trk10')
                    hist.Add(hin)
                    hNjets = infile.Get("jetpT")
                    Njets = hNjets.Integral(pT_bin_min, pT_bin_max) # should not use GetEntries() since that also includes flowbins
                    # Njets = hNjets.GetEntries()
                # hist.Sumw2() # maybe not necessary?
                with open(f"{root_dir}/scales.txt", "r") as f:
                    scale_f=float(f.read())
                    Njet_scaled_total += (scale_f * Njets)
                    
        print(Njet_scaled_total)
        hist.Scale(1 / Njet_scaled_total, "width")
        RLproj = hist.ProjectionY("proj", pT_bin_min, pT_bin_max, "e")
        with ROOT.TFile(f"{self.output_dir}/{root_out}_{ptlow}_{pthigh}.root", "RECREATE") as outf:
            outf.WriteTObject(hist)
            outf.WriteTObject(RLproj)

        self.toNumpy(RLproj, "combined")

    def toNumpy(self, hist, name):
        print("now converting")
        heights = np.zeros(self.RL_nbins)
        yerrlow = np.zeros(self.RL_nbins)
        yerrup = np.zeros(self.RL_nbins)
        for i in range(self.RL_nbins):
            heights[i] = hist.GetBinContent(i+1)
            yerrlow[i] = hist.GetBinErrorLow(i+1)
            yerrup[i] = hist.GetBinErrorUp(i+1)

        fig, ax = plt.subplots(constrained_layout=True)
        ax.errorbar(self.RL_centers, heights, [yerrlow, yerrup], [self.xerrlow, self.xerrup], marker = 'o', linestyle = '', markersize = 2, label = "chtr")
        # ax.errorbar(self.RL_centers, self.histQQ, None, [self.lowerr, self.uperr], 'go', label = "QQ")
        # ax.errorbar(self.RL_centers, self.histPM, None, [self.lowerr, self.uperr], 'bo', label = "PM")
        # ax.errorbar(self.RL_centers, self.histPP, None, [self.lowerr, self.uperr], 'co', label = "PP")
        # ax.errorbar(self.RL_centers, self.histMM, None, [self.lowerr, self.uperr], 'mo', label = "MM")
        ax.set_xscale('log')
        ax.set_title('5 < $\hat{p}_\mathrm{T}$ < 260')
        ax.set_xlabel('$R_\mathrm{L}$', fontsize=14)
        ax.set_ylabel(r'$\frac{1}{N_\mathrm{jet}}\times d\sigma/dR_\mathrm{L}$', fontsize=14)
        fig.suptitle('cEECs')
        beam_energy = "PYTHIA 8: pp $\sqrt{s} = 5.02$ TeV\n"
        jet_pT_range = "anti-$k_\mathrm{T}$ jets, " + f"$R= {self.jetR}$, ${self.ptlow}$ GeV" + "$ < p_\mathrm{T}^\mathrm{jet} < " + f"{self.pthigh}$ GeV\n"
        jet_eta_range = "$|\eta_\mathrm{jet}| < " + f"{self.max_eta_hadron - self.jetR}$\n"
        det_tr_pT_range = "$p_\mathrm{T}^\mathrm{tr} > " + f"{self.min_det_trk_pT}$ GeV\n"
        jet_tr_pT_range = "$p^\mathrm{jet,tr}_\mathrm{T} > " + f"{self.min_jet_trk_pT}$ GeV"
        # lab2 = "$N_\mathrm{ev}=$" + f"{self.nev}"
        label = beam_energy + jet_pT_range + jet_eta_range + det_tr_pT_range + jet_tr_pT_range
        ax.text(0.05, 0.95, label, fontsize = 9, transform=ax.transAxes, verticalalignment='top')
        ax.legend()

        fig.savefig(f"{self.output_dir}/{name}_{self.ptlow}_{self.pthigh}.png")
                

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='combining ENC histograms',
                                     prog=os.path.basename(__file__))
    # Could use --py-seed
    parser.add_argument('-j', '--job-id', action='store', type=int, default=0, 
                        help='Slurm job ID')
    parser.add_argument('-d', '--data-dir', default="/global/cfs/cdirs/alice/alicepro/hiccup/rstorage/alice/AnalysisResults/thwang", type=str,
                        help="Directory where data can be found")
    parser.add_argument('-o', '--output-dir', default="/global/cfs/cdirs/alice/mhwang/cEEC/results", type=str,
                        help="Directory to put output plots")
    parser.add_argument('-r', '--root-data-name', default="AnalysisResults.root", type=str,
                        help="Name of ROOT file")
    parser.add_argument('-c', '--config_file', action='store', type=str, default='/global/cfs/cdirs/alice/mhwang/mypyjetty/pyjetty/pyjetty/alice_analysis/config/cEEC/ceec_1e-3.yaml',
                        help="Path of config file for observable configurations")

    args = parser.parse_args()
    combiner = Combiner(job_id = args.job_id, config_file=args.config_file, data_dir=args.data_dir, output_dir = args.output_dir, root_data_name = args.root_data_name)
    # print(linbins(0,5,4))
    combiner.combine(60, 80)
    # combiner.combine(20, 40)
    # name = f'h_ENC2_JetPt_ch_R04_trk00'
    # with ROOT.TFile("/global/cfs/cdirs/alice/mhwang/cEEC/results/test/AnalysisResults.root") as infile:
    #     hin = infile.Get('jetpT')
    #     # hin.SetDirectory(ROOT.nullptr)
    #     height = hin.GetBinContent(5)
    #     print(height)
    #     height = hin.GetBinContent(6)
    #     print(height)
