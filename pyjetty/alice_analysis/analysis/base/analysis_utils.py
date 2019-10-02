#!/usr/bin/env python3

"""
  Analysis utilities for jet analysis with track dataframe.
  
  Author: James Mulligan (james.mulligan@berkeley.edu)
"""

from __future__ import print_function

# General
import os
import sys
import math

# Data analysis and plotting
import uproot
import pandas
import numpy as np
import ROOT

# Fastjet via python (from external library fjpydev)
import fastjet as fj
import fjext

# Base class
from pyjetty.alice_analysis.analysis.base import base

################################################################
class analysis_utils(base.base):
  
  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, **kwargs):
    super(analysis_utils, self).__init__(**kwargs)

  #---------------------------------------------------------------
  # Normalize a histogram by its integral
  #---------------------------------------------------------------
  def scale_by_integral(self, h):
    
    if h.GetSumw2() is 0:
      h.Sumw2()
    
    integral = h.Integral()
    if integral > 0:
      h.Scale(1./integral)
    else:
      print('Integral is 0, check for problem')

  #---------------------------------------------------------------
  # Remove periods from a label
  #---------------------------------------------------------------
  def remove_periods(self, text):
  
    string = str(text)
    return string.replace('.', '')

  #---------------------------------------------------------------
  # Plot and save a 1D histogram
  #---------------------------------------------------------------
  def plotHist(self, h, outputFilename, drawOptions = "", setLogy = False, setLogz = False):
    
    h.SetLineColor(1)
    h.SetLineWidth(1)
    h.SetLineStyle(1)
    
    c = ROOT.TCanvas("c","c: hist",600,450)
    c.cd()
    ROOT.gPad.SetLeftMargin(0.15)
    if setLogy:
      c.SetLogy()
    if setLogz:
      c.SetLogz()
    ROOT.gPad.SetLeftMargin(0.15)

    h.Draw(drawOptions)
    c.SaveAs(outputFilename)
    c.Close()