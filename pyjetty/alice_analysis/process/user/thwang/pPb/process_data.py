#!/usr/bin/env python3

"""
    Analysis class to read a ROOT TTree of track information
    and do jet-finding, and save basic histograms.
    
    Author: James Mulligan (james.mulligan@berkeley.edu)
"""

# General
import os
import sys
import argparse
import logging
# import sys

# Data analysis and plotting
import ROOT
import yaml
import numpy as np
import array 
import math

# Fastjet via python (from external library heppy)
import fastjet as fj
# import fjcontrib
# import fjtools
import ecorrel

# Base class
from pyjetty.alice_analysis.process.user.substructure import process_data_base_pPb

def linbins(xmin, xmax, nbins):
    lspace = np.linspace(xmin, xmax, nbins+1)
    arr = array.array('f', lspace)
    return arr

def logbins(xmin, xmax, nbins):
    lspace = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
    arr = array.array('f', lspace)
    return arr

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

logger = logging.getLogger(__name__)

################################################################
class ProcessData_ENC(process_data_base_pPb.ProcessDataBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
    
        # Initialize base class
        super(ProcessData_ENC, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)
        
        self.observable = self.observable_list[0]
        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)
        self.do_bsub_hists = config['do_bsub_hists'] if 'do_bsub_hists' in config else False
        self.charge_types = ["P", "M", "PM"]
        self.charge_types_ext = ["P", "M", "PM", "T", "Q"]

    #---------------------------------------------------------------
    # Initialize histograms
    #---------------------------------------------------------------
    def initialize_user_output_objects(self):
        for jetR in self.jetR_list:
            for observable in self.observable_list:
                for trk_thrd in self.obs_settings[observable]:

                    obs_label = self.utils.obs_label(trk_thrd, None) 
                    if self.is_pp or self.is_pA:
                        # name = f'h_{observable}Pt_JetPt_R{jetR}_{trk_thrd}'
                        # pt_bins = linbins(0,200,200)
                        # ptRL_bins = logbins(1E-3,1E2,60)
                        # h = ROOT.TH2D(name, name, 200, pt_bins, 60, ptRL_bins)
                        # h.GetXaxis().SetTitle('p_{T,ch jet}')
                        # h.GetYaxis().SetTitle('p_{T,ch jet}R_{L}') # NB: y axis scaled by jet pt (applied jet by jet)
                        # setattr(self, name, h)

                        if not self.mult_threshold:
                            self.mult_labels = ['']
                        elif len(self.mult_threshold) == 1:
                            self.mult_labels = ['_lm','_hm']
                        else:
                            self.mult_labels = ['_lm','_hm','_mm']

                        if 'jet_pt' in observable:
                            name = f'h_{observable}_JetPt_R{jetR}_{obs_label}'
                            pt_bins = linbins(0,200,200)
                            h = ROOT.TH1D(name, name, 200, pt_bins)
                            h.GetXaxis().SetTitle('p_{T,ch jet}')
                            h.GetYaxis().SetTitle('Counts')
                            setattr(self, name, h)

                            if self.mult_labels[0]:
                                for mult_label in self.mult_labels:
                                    name = 'h_{}_JetPt_R{}_{}'.format(observable + mult_label, jetR, obs_label)
                                    pt_bins = linbins(0,200,200)
                                    h = ROOT.TH1D(name, name, 200, pt_bins)
                                    h.GetXaxis().SetTitle('p_{T,ch jet}')
                                    h.GetYaxis().SetTitle('Counts')
                                    setattr(self, name, h)
                        
                        self.pair_type_labels = ['']
                        if self.do_perpendicular_cone:
                            self.pair_type_labels = ['_jj','_jp','_pp']
                            if self.mixed_cone:
                                self.pair_type_labels.append("_mx")

                        if 'ENC' in observable:
                            for pair_type_label in self.pair_type_labels:
                                for charge_label in self.charge_types_ext:
                                    for mult_label in self.mult_labels:
                                        name = f'h_{observable+"_"+charge_label+pair_type_label+mult_label}_JetPt_R{jetR}_{trk_thrd}'
                                        pt_bins = linbins(0,200,200)
                                        RL_bins = logbins(1E-3,1,30)
                                        h = ROOT.TH2D(name, name, 200, pt_bins, 30, RL_bins)
                                        h.GetXaxis().SetTitle('p_{T,ch jet}')
                                        h.GetYaxis().SetTitle('R_{L}')
                                        setattr(self, name, h)

    #---------------------------------------------------------------
    # Calculate pair distance of two fastjet particles
    #---------------------------------------------------------------
    def calculate_distance(self, p0, p1):   
        dphiabs = math.fabs(p0.phi() - p1.phi())
        dphi = dphiabs

        if dphiabs > math.pi:
            dphi = 2*math.pi - dphiabs

        deta = p0.eta() - p1.eta()
        return math.sqrt(deta*deta + dphi*dphi)

    #---------------------------------------------------------------
    # This function is called once for each jet subconfiguration
    #---------------------------------------------------------------
    def fill_jet_histograms(self, jet, jet_groomed_lund, jetR, obs_setting,
                            grooming_setting, obs_label, jet_pt_subtracted, suffix):
        constituents = fj.sorted_by_pt(jet.constituents())
        c_select = fj.vectorPJ()
        trk_thrd = obs_setting

        for c in constituents:
            if c.pt() < trk_thrd:
                break
            # getattr(self, 'hSigJetpt_Partpt_R{}_{}'.format(jetR,obs_label)).Fill(jet.perp(), c.pt())
            c.set_user_index(0)
            c_select.append(c)
        if self.do_perpendicular_cone:
            c_select2 = fj.vectorPJ()
            for c in constituents:
                if c.pt() < trk_thrd:
                    break
                c.set_user_index(0)
                c_select2.append(c)

            for part in jet.python_info().perpcone1:
                if part.pt() > trk_thrd:
                    c_select.append(part)
            for part in jet.python_info().perpcone2:
                if part.pt() > trk_thrd:
                    c_select2.append(part)

        # if self.ENC_pair_cut:
        #     dphi_cut = -9999 # means no dphi cut
        #     deta_cut = 0.008
        # else:
        #     dphi_cut = -9999
        #     deta_cut = -9999
            
        #HACK: implemented cut here
        # deta_cut = 0.008
        # dphi_cut = 0.005

        # jet_pt = jet_pt_subtracted

        hname = 'h_{}_JetPt_R{}_{}'
        for observable in self.observable_list:
            if 'ENC' in observable:
                ipoint = 2
                corr = ecorrel.CorrelatorBuilder(c_select, jet_pt_subtracted, ipoint, 1, -9999, -9999)
                for indices, RL, weight in zip(corr.correlator(ipoint).indices(), corr.correlator(ipoint).rs(), corr.correlator(ipoint).weights()):
                    idx1, idx2 = indices
                    uidx1 = c_select[idx1].user_index()
                    uidx2 = c_select[idx2].user_index()
                    pair_type = self.get_pair_type(uidx1, uidx2)
                    # logger.info(c_select[idx1].has_user_info())
                    q1 = c_select[idx1].python_info().charge
                    q2 = c_select[idx2].python_info().charge
                    charge_type = self.get_charge_type(q1, q2)
                    # dphistar = self.calc_dphistar(c_select[idx1], c_select[idx2], q1, q2)
                    # deta = np.abs(c_select[idx1].eta() - c_select[idx2].eta())
                    # if dphistar < dphi_cut and deta < deta_cut:
                    #     continue

                    getattr(self, hname.format(observable+charge_type+pair_type, jetR, obs_label)).Fill(jet_pt_subtracted, RL, weight)
                    getattr(self, hname.format(observable+"_T"+pair_type, jetR, obs_label)).Fill(jet_pt_subtracted, RL, weight)
                    getattr(self, hname.format(observable+"_Q"+pair_type, jetR, obs_label)).Fill(jet_pt_subtracted, RL, q1*q2*weight)
                if self.do_perpendicular_cone:
                    corr2 = ecorrel.CorrelatorBuilder(c_select2, jet_pt_subtracted, ipoint, 1, -9999, -9999)
                    for indices, RL, weight in zip(corr2.correlator(ipoint).indices(), corr2.correlator(ipoint).rs(), corr2.correlator(ipoint).weights()):
                        idx1, idx2 = indices
                        uidx1 = c_select2[idx1].user_index()
                        uidx2 = c_select2[idx2].user_index()
                        pair_type = self.get_pair_type(uidx1, uidx2)
                        q1 = c_select2[idx1].python_info().charge
                        q2 = c_select2[idx2].python_info().charge
                        charge_type = self.get_charge_type(q1, q2)
                        if pair_type == "_jj":
                            continue
                        # dphistar = self.calc_dphistar(c_select2[idx1], c_select2[idx2], q1, q2)
                        # deta = np.abs(c_select2[idx1].eta() - c_select2[idx2].eta())
                        # if dphistar < dphi_cut and deta < deta_cut:
                        #     continue
                        getattr(self, hname.format(observable+charge_type+pair_type, jetR, obs_label)).Fill(jet_pt_subtracted, RL, weight)
                        getattr(self, hname.format(observable+"_T"+pair_type, jetR, obs_label)).Fill(jet_pt_subtracted, RL, weight)
                        getattr(self, hname.format(observable+"_Q"+pair_type, jetR, obs_label)).Fill(jet_pt_subtracted, RL, q1*q2*weight)
                if self.mixed_cone:
                    cone1_w_ptcut = [p for p in jet.python_info().perpcone1 if p.pt() > trk_thrd]
                    cone2_w_ptcut = [p for p in jet.python_info().perpcone2 if p.pt() > trk_thrd]
                    for p1 in cone1_w_ptcut:
                        for p2 in cone2_w_ptcut:
                            pair_weight = p1.pt() * p2.pt() / (jet_pt_subtracted ** 2)
                            pair_RL = np.sqrt((p1.eta() - p2.eta()) ** 2 + (p1.delta_phi_to(p2)) ** 2)
                            q1 = p1.python_info().charge
                            q2 = p2.python_info().charge
                            charge_type = self.get_charge_type(q1, q2)
                            getattr(self, hname.format(observable+charge_type+"_mx", jetR, obs_label)).Fill(jet_pt_subtracted, pair_RL, pair_weight)
                            getattr(self, hname.format(observable+"_T_mx", jetR, obs_label)).Fill(jet_pt_subtracted, pair_RL, pair_weight)
                            getattr(self, hname.format(observable+"_Q_mx", jetR, obs_label)).Fill(jet_pt_subtracted, pair_RL, q1*q2*pair_weight)
                            # we fill twice to get the reverse pair order as well
                            getattr(self, hname.format(observable+charge_type+"_mx", jetR, obs_label)).Fill(jet_pt_subtracted, pair_RL, pair_weight)
                            getattr(self, hname.format(observable+"_T_mx", jetR, obs_label)).Fill(jet_pt_subtracted, pair_RL, pair_weight)
                            getattr(self, hname.format(observable+"_Q_mx", jetR, obs_label)).Fill(jet_pt_subtracted, pair_RL, q1*q2*pair_weight)

            if 'jet_pt' in observable:
                getattr(self, hname.format(observable, jetR, obs_label)).Fill(jet_pt_subtracted)  
                # if self.mult_threshold != 0:
                #     if self.isHighMult:
                #         mult_label = self.mult_labels[1]
                #     elif self.isLowMult:
                #         mult_label = self.mult_labels[0]
                #     else:
                #         mult_label = self.mult_labels[2]
                #     getattr(self, hname.format(observable + mult_label, jetR, obs_label)).Fill(jet_pt)  

    def get_charge_type(self, q1, q2):
        if q1 > 0 and q2 > 0:
            return '_P'
        elif q1 * q2 == -1:
            return '_PM'
        else:
            return '_M'
        
    def get_pair_type(self, uidx1, uidx2):
        # jet parts have uidx 0, cone1 (+angle) has uidx +1, cone2 (-angle) has uidx -1
        if uidx1 == 0 and uidx2 == 0:
            return '_jj'
        elif uidx1 * uidx2 == 0: # take advantage of lazy if
            return '_jp'
        else:
            return '_pp'
    def calc_dphistar(self, p1, p2, q1, q2):
        R = 1.1
        Bz = 0.5
        return np.abs(p1.delta_phi_to(p2) + q1*np.arcsin(0.15*Bz*R/p1.pt()) - q2*np.arcsin(0.15*Bz*R/p2.pt()))
    # double R = 1.1; // reference radius for TPC
    #     double Bz = 0.5;
                # double phi_star = phi12 + q1*asin(-0.15*Bz*R/pt1) - q2*asin(-0.15*Bz*R/pt2);
                # _phi12 = fabs(parts[idx_i].delta_phi_to(parts[idx_j]))
    #---------------------------------------------------------------
    # Perp cone background pair subtraction histograms
    #---------------------------------------------------------------
    def fill_perp_cone_histograms(self, cone_parts, cone_R, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting, obs_label, jet_pt_ungroomed, suffix, rho_bge = 0):

        # calculate perp cone pt after subtraction. Notice that the perp cone already contain the particles from signal "jet". Signal and background can be identified using user_index()
        cone_px = 0
        cone_py = 0
        cone_npart = 0
        for part in cone_parts:
            if part.user_index() < 0:
                cone_px = cone_px + part.px()
                cone_py = cone_py + part.py()
                cone_npart = cone_npart + 1
        cone_pt = math.sqrt(cone_px*cone_px + cone_py*cone_py)
        # print('cone pt', cone_pt-rho_bge*jet.area(), '(', cone_pt, ')')
        cone_pt = cone_pt-rho_bge*jet.area() # ideally this should fluctuate around 0
        # print('jet pt', jet_pt_ungroomed, '(', jet.perp(), ')')

        # combine sig jet and perp cone with trk threshold cut
        trk_thrd = obs_setting
        c_select = fj.vectorPJ()
        c_select_perp = fj.vectorPJ()

        cone_parts_sorted = fj.sorted_by_pt(cone_parts)
        # print('perp cone nconst:',len(cone_parts_sorted))
        for part in cone_parts_sorted:
            if part.pt() < trk_thrd:
                break
            c_select.append(part) # NB: use the break statement since constituents are already sorted
            if part.user_index() < 0:
                c_select_perp.append(part)

        nconst_perp = len(c_select_perp)
        # print('cone R',cone_R)
        # print('total cone nconst (with thrd cut):',len(c_select))
        # print('perp cone nconst (with thrd cut):',nconst_perp)

        if self.ENC_pair_cut:
            dphi_cut = -9999 # means no dphi cut
            deta_cut = 0.008
        else:
            dphi_cut = -9999
            deta_cut = -9999

        hname = 'h_perpcone{}_{}_JetPt_R{}_{}{}'
        if self.do_median_subtraction:
            jet_pt = jet_pt_ungroomed # jet_pt_ungroomed stores subtracted jet pt for energy weight calculation and pt selection for there is a non-zero UE energy density
        else:
            jet_pt = jet.perp()
        # print('analyze perpcone', jet_pt_ungroomed)
        new_corr = ecorrel.CorrelatorBuilder(c_select, jet_pt, 2, 1, dphi_cut, deta_cut)
        for observable in self.observable_list:

            if 'jet_pt' in observable:
                getattr(self, hname.format(cone_R, 'pt', jetR, obs_label, suffix)).Fill(cone_pt)
                getattr(self, hname.format(cone_R, 'Nconst', jetR, obs_label, suffix)).Fill(jet_pt, nconst_perp)

            if 'ENC' in observable or 'EEC_noweight' in observable or 'EEC_weight2' in observable:
                for ipoint in range(2, 3):
                    for index in range(new_corr.correlator(ipoint).rs().size()):

                        # # processing only like-sign pairs when self.ENC_pair_like is on
                        # if self.ENC_pair_like and (not self.is_same_charge(new_corr, ipoint, c_select, index)):
                        # 	continue

                        # # processing only unlike-sign pairs when self.ENC_pair_unlike is on
                        # if self.ENC_pair_unlike and self.is_same_charge(new_corr, ipoint, c_select, index):
                        # 	continue

                        # separate out sig-sig, sig-bkg, bkg-bkg correlations for EEC pairs
                        pair_type_label = ''
                        mult_label = ''
                        if self.do_median_subtraction:
                            pair_type = self.check_pair_type(new_corr, ipoint, c_select, index)
                            pair_type_label = self.pair_type_labels[pair_type]
                            if self.mult_threshold != 0:	
                                if self.isHighMult: 
                                    mult_label = self.mult_labels[1]
                                elif self.isLowMult: 
                                    mult_label = self.mult_labels[0]
                                else:
                                    mult_label = self.mult_labels[2]
                                

                        if 'ENC' in observable:
                            # print('hname is',hname.format(cone_R, observable + str(ipoint) + pair_type_label, jetR, obs_label))
                            getattr(self, hname.format(cone_R, observable + str(ipoint) + pair_type_label + mult_label, jetR, obs_label, suffix)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index])
                            getattr(self, hname.format(cone_R, observable + str(ipoint) + pair_type_label + mult_label + 'Pt', jetR, obs_label, suffix)).Fill(jet_pt, jet_pt*new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index]) # NB: fill pt*RL

                        if ipoint==2 and 'EEC_noweight' in observable:
                            getattr(self, hname.format(cone_R, observable + pair_type_label + mult_label, jetR, obs_label, suffix)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index])

                        if ipoint==2 and 'EEC_weight2' in observable:
                            getattr(self, hname.format(cone_R, observable + pair_type_label + mult_label, jetR, obs_label, suffix)).Fill(jet_pt, new_corr.correlator(ipoint).rs()[index], pow(new_corr.correlator(ipoint).weights()[index],2))


##################################################################
if __name__ == '__main__':
    # Define arguments
    parser = argparse.ArgumentParser(description='Process data')
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
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(funcName)s - %(message)s'))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.info('Configuring...')
    logger.info(f"Input file: \'{args.input_file}\'")
    logger.info(f"Config file: \'{args.config_file}\'")
    logger.info(f"Ouput directory: '{args.output_dir}'")

    # If invalid inputFile is given, exit
    if not os.path.exists(args.input_file):
        logger.critical(f'File {args.input_file} does not exist! Exiting!')
        sys.exit(1)

    # If invalid configFile is given, exit
    if not os.path.exists(args.config_file):
        logger.critical(f'File {args.config_file} does not exist! Exiting!')
        sys.exit(1)

    analysis = ProcessData_ENC(input_file=args.input_file, config_file=args.config_file, output_dir=args.output_dir)
    analysis.process_data()