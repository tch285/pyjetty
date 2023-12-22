#!/usr/bin/env python

import ROOT
import numpy as np
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
results_path = f"{dir_path}/results"

class Histogram1D:
    """Class to efficiently and easily fill, store, and plot histograms.

    Contains conversions to ROOT because pyroot sux
    """
    def __init__(self, data = None, weights = None, nbins = 50, bin_min = 1e-4, bin_max = 1, binning = 'log', title = "") -> None:
        self.nbins = nbins
        self.binning = binning
        self.bin_min = bin_min
        self.bin_max = bin_max
        self._create_bins()
        if data is None:
            self.bin_contents = np.zeros(self.nbins)
            if weights is not None:
                print("Weights passed but no data, weights ignored.")
        elif isinstance(data, (int, float, np.ndarray, list)):
            assert len(data) == len(weights)
            self.bin_contents = np.histogram(data, bins = self.bin_edges, weights = weights)[0]
        else:
            raise TypeError("`bin_info` is not an acceptable type!")
        self.yerr = np.zeros(self.nbins) # no up or down error since it's the same both ways
        self.title = title
    
    def __add__(self, hist):
        self.bin_contents += hist.bin_contents
        return self
    
    def __mul__(self, f):
        assert isinstance(f, (float, int))
        self.yerr *= abs(f) # REVIEW: not sure if this behavior is desired or not actually...
        self.bin_contents *= f
        return self
    
    def __len__(self): return self.nbins
    def __getitem__(self, idx): return self.bin_contents[idx]
    def __setitem__(self, idx, val): self.bin_contents[idx] = val
    def __repr__(self): return f"{self.title}:{self.xtitle}:{self.ytitle}"
    def __truediv__(self, f): return self * (1 / f)
    def __sub__(self, hist): return self + (hist * -1)

    def _create_bins(self) -> None:
        if self.binning in ["linear", "lin", 1]:
            self.bin_edges, step = np.linspace(self.bin_min, self.bin_max, self.nbins + 1, endpoint = True, retstep = True)
            self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
            self.xlowerr = self.xuperr = np.full(self.nbins, step / 2)
        elif self.binning in ['logarithmic', 'log', 2]:
            self.bin_edges = np.logspace(np.log10(self.bin_min), np.log10(self.bin_max), self.nbins + 1)
            self.bin_centers = np.sqrt(self.bin_edges[:-1] * self.bin_edges[1:])
            self.xlowerr = self.bin_centers - self.bin_edges[:-1]
            self.xuperr = self.bin_edges[1:] - self.bin_centers
        else:
            raise ValueError("Binning option is not recognized: must be 'lin', 'log', 1, 'logarithmic', 'log', 2.")
    
    def fill(self, vals, weights):
        assert vals.size() == weights.size() # REVIEW: size? len? I think in principle len is probably the most robust option
        self.bin_contents += np.histogram(vals, bins = self.bin_edges, weights = weights)[0]

    def calc_error(self):
        self.yerr = np.sqrt(self.bin_contents)
    
    def set_bin_error(self, bin, err):
        self.yerr[bin] = err
        
    def normalize(self):
        self.scale(self.bin_contents.sum()) # FIXME: rewrite in terms of __mul__

    def toROOT(self, name): # try to avoid doing this until the very end
        self.set_error() # just as a last check
        hist = ROOT.TH1F(name, self.title, self.nbins, self.bin_edges)
        hist.GetXaxis().SetTitle(self.xtitle)
        hist.GetYaxis().SetTitle(self.ytitle)

        for i, freq in enumerate(self.bin_contents):
            hist.SetBinContent(i+1,freq)
            hist.SetBinError(i+1, self.yerr[i])
    
    def save(self, filename: str, show: bool):
        fig, ax = plt.subplots()
        ax.errorbar(self.bin_centers, self.bin_contents, self.yerr, [self.xlowerr, self.xuperr], 'ro')
        try:
            ax.set_title(self.title.split(":")[0])
            ax.set_xlabel(self.title.split(":")[1])
            ax.set_ylabel(self.title.split(":")[2])
        except IndexError:
            pass # HACK: not sure if this actually works; will it actually set the successful lines if it fails at L360?

        if self.binning in ['logarithmic', 'log', 2]:
            ax.set_yscale('log')

        fig.savefig(f"{results_path}/{filename}.png")
        if show:
            plt.show()

if __name__ == '__main__':
    
    hist1 = Histogram1D(data = [2], weights = [0.6], nbins = 18, bin_min = 1e-4, bin_max = 4, binning = 'log', title = "a:b:c")
    hist1.calc_error()
    print(len(hist1))
    print(hist1)
    # print(hist1.bin_edges)
    # print(hist1.bin_contents)
    # print(hist1.yerr)
    # hist1.save("test.png", True)
