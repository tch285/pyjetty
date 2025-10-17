#!/usr/bin/env python3
import ROOT

output_file="/global/cfs/cdirs/alice/mhwang/fspyjetty/results.root"

ROOT.gStyle.SetStatH(0.0001)
ROOT.gStyle.SetStatW(0.0001)
ROOT.gStyle.SetOptFit(True)
ROOT.gStyle.SetOptStat(0)

c=ROOT.TCanvas("c","",800,400)
c.Divide(2)

pTmins=  [20, 40, 60]

with ROOT.TFile(output_file,'read') as f:
    for i, pTmin in enumerate(pTmins):
        h_jet = f.Get("jet_pT_det")
        njets = h_jet.GetBinContent(i+1)
        print(njets)
        c.cd(1)
        h=f.Get(f"EEC_det_{pTmin}_{pTmin+20}")
        h.Scale(1 / njets, "width")
        ROOT.gPad.SetLogx()
        h.Draw()

        h_jet = f.Get("jet_pT_gen")
        njets = h_jet.GetBinContent(i+1)
        print(njets)
        c.cd(2)
        h=f.Get(f"EEC_gen_{pTmin}_{pTmin+20}")
        h.Scale(1 / njets, "width")
        ROOT.gPad.SetLogx()
        h.Draw()

        c.SaveAs(f"plots_{pTmin}.pdf")