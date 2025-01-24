#!/usr/bin/env python3

"""
  Analysis IO class for jet analysis with track dataframe.
  Each instance of the class handles the IO of a *single* track tree.
  
  Authors: James Mulligan
           Mateusz Ploskon
           Ezra Lesser
"""

from __future__ import print_function

import os   # for creating file on output
import sys

# Data analysis and plotting
import uproot
import pandas
import numpy as np
from particle import PDGID


# Fastjet via python (from external library fjpydev)
# import fastjet as fj
import fjext
import logging

# Base class
from pyjetty.alice_analysis.process.base import common_base
from pyjetty.alice_analysis.process.base import jet_info

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

handler.setFormatter(ColoredFormatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(funcName)s - %(message)s'))

logger.addHandler(handler)
logger.setLevel(logging.INFO)

################################################################
class ProcessIO(common_base.CommonBase):
  
  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', tree_dir='PWGHF_TreeCreator',
               track_tree_name='tree_Particle', event_tree_name='tree_event_char',
               output_dir='', is_pp=True, min_cent=0., max_cent=10.,
               use_ev_id_ext=True, is_jetscape=False, holes=False,
               event_plane_range=None, skip_event_tree=False, is_ENC=False,
               is_det_level=False, is_mc = True, load_mult = False, **kwargs):
    super(ProcessIO, self).__init__(**kwargs)
    self.input_file = input_file
    self.output_dir = output_dir
    self.tree_dir = tree_dir
    if len(tree_dir) and tree_dir[-1] != '/':
      self.tree_dir += '/'
    self.track_tree_name = track_tree_name
    self.event_tree_name = event_tree_name
    self.is_pp = is_pp
    self.use_ev_id_ext = use_ev_id_ext
    self.is_jetscape = is_jetscape
    self.holes = holes
    self.event_plane_range = event_plane_range
    self.skip_event_tree = skip_event_tree
    self.is_ENC = is_ENC
    self.is_det_level = is_det_level
    if len(output_dir) and output_dir[-1] != '/':
      self.output_dir += '/'
    self.reset_dataframes()
    self.is_mc = is_mc
    self.load_mult = load_mult
    
    # Set the combination of fields that give a unique event id
    self.unique_identifier =  ['run_number', 'ev_id']
    if self.use_ev_id_ext:
      if self.is_pp:
        self.unique_identifier += ['ev_id_ext']
      else:
        pass # somehow for PbPb there is no 'ev_id_ext'
      
    # Set relevant columns of event tree
    self.event_columns = self.unique_identifier + ['z_vtx_reco', 'is_ev_rej']
    if not self.is_pp:
      self.event_columns += ['centrality']
      self.min_centrality = min_cent
      self.max_centrality = max_cent
    if self.load_mult:
      self.event_columns += ['V0Amult']
    if is_jetscape:
      self.event_columns += ['event_plane_angle']
    
    # Set relevant columns of track tree
    self.track_columns = self.unique_identifier + ['ParticlePt', 'ParticleEta', 'ParticlePhi']
    if is_jetscape:
      self.track_columns += ['status']
    if is_ENC:
      if is_det_level:
        self.track_columns += ['ParticleMCIndex']
      else:
        self.track_columns += ['ParticlePID']
    else:
      self.track_columns += ['ParticleCharge']
      if self.is_mc:
        self.track_columns += ['ParticleMCid']

    
    #print(self)
    
  #---------------------------------------------------------------
  # Clear dataframes
  #---------------------------------------------------------------
  def reset_dataframes(self):
    self.event_df_orig = None
    self.track_df = None
  
  #---------------------------------------------------------------
  # Convert ROOT TTree to SeriesGroupBy object of fastjet particles per event.
  # Optionally, define the mass assumption used in the jet reconstruction;
  #             remove a certain random fraction of tracks;
  #             randomly assign proton and kaon mass to some tracks
  #---------------------------------------------------------------
  def load_data(self, m=0.1396, reject_tracks_fraction=0., offset_indices=False,
                group_by_evid=True, random_mass=False, min_pt=0.):
    
    self.reject_tracks_fraction = reject_tracks_fraction
    self.reset_dataframes()

    logger.info('Converting ROOT trees to pandas dataframes...')
    logger.info(f'Track TTree name = {self.track_tree_name}')
    if self.is_mc and not self.is_det_level and not self.is_ENC:
      # must be MC and generated level and not the fast sim
      self.charge_factor = 3
    else:
      self.charge_factor = 1
    logger.info(f"Charge normalization factor set to {self.charge_factor}.")

    self.track_df = self.load_dataframe()

    if self.reject_tracks_fraction > 1e-3:
      n_remove = int(reject_tracks_fraction * len(self.track_df.index))
      logger.warning(f'Removing {n_remove} of {len(self.track_df.index)} tracks from {self.track_tree_name}.')
      np.random.seed()
      indices_remove = np.random.choice(self.track_df.index, n_remove, replace=False)
      self.track_df.drop(indices_remove, inplace=True)

    if random_mass:
      logger.warning('Randomly assigning proton and kaon mass to some tracks.')

    df_fjparticles = self.group_fjparticles(m, offset_indices, group_by_evid, random_mass, min_pt=min_pt)

    return df_fjparticles
  
  #---------------------------------------------------------------
  # Convert ROOT TTree to pandas dataframe
  # Return merged track+event dataframe from a given input file
  # Returned dataframe has one row per jet constituent:
  #     run_number, ev_id, ParticlePt, ParticleEta, ParticlePhi
  #---------------------------------------------------------------
  def load_dataframe(self):
    # Load event tree into dataframe
    if not self.skip_event_tree:
      event_df = None
      event_tree_name = self.tree_dir + self.event_tree_name
      with uproot.open(self.input_file)[event_tree_name] as event_tree:
        self.event_df_orig = uproot.concatenate(event_tree, self.event_columns, library="pd")
    
      # Check if there are duplicated event ids
      n_duplicates = sum(self.event_df_orig.duplicated(self.unique_identifier))
      if n_duplicates > 0:
        raise ValueError(f"There appear to be {n_duplicates} duplicate events in the event dataframe")
      
      # Apply event selection
      self.event_df_orig.reset_index(drop=True)
      if self.is_pp:
        event_criteria = 'is_ev_rej == 0'
      else:
        event_criteria = 'is_ev_rej == 0 and centrality > @self.min_centrality and centrality < @self.max_centrality'
      if self.event_plane_range:
        event_criteria += ' and event_plane_angle > @self.event_plane_range[0] and event_plane_angle < @self.event_plane_range[1]'
      event_df = self.event_df_orig.query(event_criteria)
      event_df.reset_index(drop=True)

    # Load track tree into dataframe
    track_df_orig = None
    track_tree_name = self.tree_dir + self.track_tree_name
    with uproot.open(self.input_file)[track_tree_name] as track_tree:
      track_df_orig = uproot.concatenate(track_tree, self.track_columns, library="pd")
    
    # Apply hole selection, in case of jetscape
    if self.is_jetscape:
      if self.holes:
        track_criteria = 'status == -1'
      else:
        track_criteria = 'status == 0'
      track_df_orig = track_df_orig.query(track_criteria)
      track_df_orig.reset_index(drop=True)
    
    # Check if there are duplicated tracks
    n_duplicates = sum(track_df_orig.duplicated(self.track_columns))
    if n_duplicates > 0:
      raise ValueError(
        "There appear to be %i duplicate particles in the track dataframe" % n_duplicates)
    logger.debug(f"Track original dataframe:\n{track_df_orig}")
    logger.debug(f"Event original dataframe:\n{event_df}")
    # Merge event info into track tree
    if self.skip_event_tree:
      self.track_df = track_df_orig
    else:
      self.track_df = pandas.merge(track_df_orig, event_df, on=self.unique_identifier)

    # Check if there are duplicated tracks in the merge dataframe
    n_duplicates = sum(self.track_df.duplicated(self.track_columns))
    if n_duplicates > 0:
      logger.critical(f'There appear to be {n_duplicates} duplicate particles in the merged dataframe')
      sys.exit(1)
    self.event_df = event_df
    return self.track_df

  #---------------------------------------------------------------
  # Opposite operation as load_dataframe above. Takes a dataframe
  # with the same formatting and saves to class's output_file.
  # histograms is list of tuples: [ ("title", np.histogram), ... ]
  #---------------------------------------------------------------
  def save_dataframe(self, filename, df, df_true=False, histograms=[], is_jetscape=False, is_ENC=False):

    # Create output directory if it does not already exist
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    # Open output directory and (re)create rootfile
    with uproot.recreate(self.output_dir + filename) as f:

      branchdict = {"run_number": int, "ev_id": int, "ParticlePt": float,
                      "ParticleEta": float, "ParticlePhi": float, "ParticleCharge": int}
      branchdict_true = {"run_number": int, "ev_id": int, "ParticlePt": float,
                      "ParticleEta": float, "ParticlePhi": float, "ParticleCharge": int}
      if is_jetscape:
        branchdict_true["status"] = int
        branchdict["status"] = int

      if is_ENC:
        branchdict_true["ParticlePID"] = int
        branchdict["ParticleMCIndex"] = int

      if df_true:
        # Create tree with truth particle info
        title = 'tree_Particle_gen'
        print("Length of truth track tree: %i" % len(self.track_df))
        f.mktree(name=title, branch_types=branchdict_true, title=title)
        if is_jetscape:
            f[title].extend( { "run_number": self.track_df["run_number"],
                               "ev_id": self.track_df["ev_id"],
                               "ParticlePt": self.track_df["ParticlePt"],
                               "ParticleEta": self.track_df["ParticleEta"],
                               "ParticlePhi": self.track_df["ParticlePhi"],
                               "status": self.track_df["status"] } )
        elif is_ENC:
          f[title].extend( { "run_number": self.track_df["run_number"],
                               "ev_id": self.track_df["ev_id"],
                               "ParticlePt": self.track_df["ParticlePt"],
                               "ParticleEta": self.track_df["ParticleEta"],
                               "ParticlePhi": self.track_df["ParticlePhi"],
                               "ParticlePID": self.track_df["ParticlePID"] } ) # to get charge info
        else:
            f[title].extend( { "run_number": self.track_df["run_number"],
                               "ev_id": self.track_df["ev_id"],
                               "ParticlePt": self.track_df["ParticlePt"],
                               "ParticleEta": self.track_df["ParticleEta"],
                               "ParticlePhi": self.track_df["ParticlePhi"] } )

      # Create tree with detector-level particle info
      title = 'tree_Particle'
      print("Length of detector-level track tree: %i" % len(df))
      f.mktree(name=title, branch_types=branchdict, title=title)
      if is_jetscape:
        f[title].extend( { "run_number": df["run_number"],
                           "ev_id": df["ev_id"],
                           "ParticlePt": df["ParticlePt"],
                           "ParticleEta": df["ParticleEta"],
                           "ParticlePhi": df["ParticlePhi"],
                           "status": df["status"] } )
      elif is_ENC:
        f[title].extend( { "run_number": df["run_number"],
                               "ev_id": df["ev_id"],
                               "ParticlePt": df["ParticlePt"],
                               "ParticleEta": df["ParticleEta"],
                               "ParticlePhi": df["ParticlePhi"],
                               "ParticleMCIndex": df["ParticleMCIndex"] } ) # associated MC particle index
      else:
        f[title].extend( { "run_number": df["run_number"],
                           "ev_id": df["ev_id"],
                           "ParticlePt": df["ParticlePt"],
                           "ParticleEta": df["ParticleEta"],
                           "ParticlePhi": df["ParticlePhi"] } )

      # Create tree with event char
      title = self.event_tree_name
      branchdict = {"is_ev_rej": int, "run_number": int, "ev_id": int, "z_vtx_reco": float}
      if is_jetscape:
        branchdict["event_plane_angle"] = float
      f.mktree(name=title, branch_types=branchdict, title=title)
      if is_jetscape:
        f[title].extend( {"is_ev_rej": self.event_df_orig["is_ev_rej"], 
                        "run_number": self.event_df_orig["run_number"], 
                        "ev_id": self.event_df_orig["ev_id"],
                        "z_vtx_reco": self.event_df_orig["z_vtx_reco"],
                        "event_plane_angle": self.event_df_orig["event_plane_angle"] } )
      else:
        f[title].extend( {"is_ev_rej": self.event_df_orig["is_ev_rej"], 
                        "run_number": self.event_df_orig["run_number"], 
                        "ev_id": self.event_df_orig["ev_id"],
                        "z_vtx_reco": self.event_df_orig["z_vtx_reco"] } )
        
      # Write hNevents histogram: number of accepted events at detector level
      f["hNevents"] = ( np.array([ 0, df["ev_id"].nunique() ]), np.array([ -0.5, 0.5, 1.5 ]) )

      # Write histograms to file too, if any are passed
      for title, h in histograms:
        f[title] = h

  #---------------------------------------------------------------
  # Transform the track dataframe into a SeriesGroupBy object
  # of fastjet particles per event.
  #---------------------------------------------------------------
  def group_fjparticles(self, m, offset_indices=False, group_by_evid=True, random_mass=False, min_pt=0.):

    logger.info(f'is_ENC on: {self.is_ENC}')
    logger.info(f'Detector level: {self.is_det_level}')
    logger.debug(f'Track df:\n{self.track_df}')
    if group_by_evid:
      logger.info("Transforming the track DataFrame into a Series object of FJ particles per event...")

      # (i) Group the track dataframe by event
      #     track_df_grouped is a DataFrameGroupBy object with one track dataframe per event
      track_df_grouped = self.track_df.groupby(self.unique_identifier)

      if self.is_ENC:
        df_fjparticles_orig = track_df_grouped.apply(
        self.get_fjparticles, m=m, offset_indices=offset_indices, random_mass=random_mass, min_pt=min_pt)
        if self.is_det_level:
          df_fjparticles_aux = track_df_grouped.apply(
          self.get_particles_mc_index, m=m, offset_indices=offset_indices, random_mass=random_mass, min_pt=min_pt)
          df_fjparticles = pandas.DataFrame({"fj_particle": df_fjparticles_orig, "ParticleMCIndex": df_fjparticles_aux})
        else:
          df_fjparticles_aux = track_df_grouped.apply(
          self.get_particles_pid, m=m, offset_indices=offset_indices, random_mass=random_mass, min_pt=min_pt)
          df_fjparticles = pandas.DataFrame({"fj_particle": df_fjparticles_orig, "ParticlePID": df_fjparticles_aux})
      else:
        df_fjparticles = track_df_grouped.apply(
        self.get_fjparticles, m=m, offset_indices=offset_indices, random_mass=random_mass, min_pt=min_pt)
      
      logger.debug(f'Combined:\n{df_fjparticles}')
      
      # df_fjparticles = pandas.DataFrame({"fj_particle": track_df_grouped.apply(
      #   self.get_fjparticles, m=m, offset_indices=offset_indices, random_mass=random_mass, min_pt=min_pt), "ParticleMCIndex": track_df_grouped["ParticleMCIndex"]})
    
    else:
      logger.info("Transforming the track DataFrame into a DataFrame of FJ particles per track...")

      # Transform into a DataFrame of fastjet particles
      # if it's for energy correlator analysis, add particle id and associated MC info for truth and det level input respectively
      df = self.track_df
      if self.is_ENC:
        if self.is_det_level:
          df_fjparticles = pandas.DataFrame( 
            {"run_number": df["run_number"], "ev_id": df["ev_id"],
            "fj_particle": self.get_fjparticles(self.track_df, m, offset_indices, random_mass, min_pt=  min_pt), "ParticleMCIndex": df["ParticleMCIndex"]} )
        else:
          df_fjparticles = pandas.DataFrame( 
            {"run_number": df["run_number"], "ev_id": df["ev_id"],
            "fj_particle": self.get_fjparticles(self.track_df, m, offset_indices, random_mass, min_pt=  min_pt), "ParticlePID": df["ParticlePID"]} )
      else:
        df_fjparticles = pandas.DataFrame( 
          {"run_number": df["run_number"], "ev_id": df["ev_id"],
          "fj_particle": self.get_fjparticles(self.track_df, m, offset_indices, random_mass, min_pt=  min_pt)} )

    return df_fjparticles

  #---------------------------------------------------------------
  # Return fastjet:PseudoJets from a given track dataframe
  #---------------------------------------------------------------
  def get_fjparticles(self, df_tracks, m, offset_indices=False, random_mass=False, min_pt=0.):
    # If offset_indices is true, then offset the user_index by a large negative value
    user_index_offset = 0
    if offset_indices:
        user_index_offset = int(-1e6)

    # Apply a pt cut
    df_tracks_accepted = df_tracks[df_tracks.ParticlePt > min_pt]

    m_array = np.full((df_tracks_accepted['ParticlePt'].values.size), m)

    # Randomly assign K and p mass for systematic check
    if random_mass:
      rand_val = np.random.random((len(m_array)))
      K_mass = 0.4937     # GeV/c^2
      p_mass = 0.938272   # GeV/c^2
      # (p + pbar) / (pi+ + pi-) ~ 5.5%
      # (K+ + K-) / (pi+ + pi-) ~ 13%
      # But these are numbers with respect to the final _unreplaced_ pions, so there is
      # an additional factor of 1/(1 + 5.5% + 13%) to get things right
      K_factor = 0.13
      p_factor = 0.055
      K_prob = K_factor / (1 + K_factor + p_factor)
      p_prob = 1 - p_factor / (1 + K_factor + p_factor)   # 1- just to look at diff random vals
      m_array = np.where(rand_val < K_prob, K_mass, m_array)
      m_array = np.where(rand_val > p_prob, p_mass, m_array)

    # Use swig'd function to create a vector of fastjet::PseudoJets from numpy arrays of pt,eta,phi
    fj_particles = fjext.vectorize_pt_eta_phi_m(
      df_tracks_accepted['ParticlePt'].values, df_tracks_accepted['ParticleEta'].values,
      df_tracks_accepted['ParticlePhi'].values, m_array, user_index_offset)

    if self.is_ENC:
      if self.is_det_level:
        for i, mcid in enumerate(df_tracks_accepted['ParticleMCIndex'].values):
          info = jet_info.JetInfo()
          info.mcid = int(mcid)
          fj_particles[i].set_python_info(info)
          # charge attached later, since we need truth pid info
      else: # is truth level
        for i, pid in enumerate(df_tracks_accepted['ParticlePID'].values):
          info = jet_info.JetInfo()
          info.charge = PDGID(pid).charge
          info.mcid = i
          fj_particles[i].set_python_info(info)
    else:
      if self.is_mc:
        for i, (charge, mcid) in enumerate(zip(df_tracks_accepted['ParticleCharge'].values, df_tracks_accepted['ParticleMCid'].values)):
          info = jet_info.JetInfo()
          info.charge = charge / self.charge_factor
          info.mcid = mcid
          fj_particles[i].set_python_info(info)
      else:
        for i, charge in enumerate(df_tracks_accepted['ParticleCharge'].values):
          info = jet_info.JetInfo()
          info.charge = charge / self.charge_factor
          fj_particles[i].set_python_info(info)
    # return fj_particles
    if self.load_mult:
      assert (df_tracks_accepted['V0Amult'].values[0] == df_tracks_accepted['V0Amult'].values).all()
      return pandas.DataFrame({
          'parts': [fj_particles],
          'mult': df_tracks_accepted['V0Amult'].values[0],
      })
    else:
      return fj_particles
    # return pandas.Series({
    #     'fj': fj_particles,
    #     'v0amult': df_tracks_accepted['V0Amult'].values[0],
    # })`

  #---------------------------------------------------------------
  # Return associated mc indices from a given track dataframe
  #---------------------------------------------------------------
  def get_particles_mc_index(self, df_tracks, m, offset_indices=False, random_mass=False, min_pt=0.):
    
    # If offset_indices is true, then offset the user_index by a large negative value
    # NOTE: commented out this section, doesn't seem used
    # user_index_offset = 0
    # if offset_indices:
    #     user_index_offset = int(-1e6)
        
    # Apply a pt cut
    df_tracks_accepted = df_tracks[df_tracks.ParticlePt > min_pt]

    return df_tracks_accepted['ParticleMCIndex'].values

  #---------------------------------------------------------------
  # Return particle id from a given track dataframe
  #---------------------------------------------------------------
  def get_particles_pid(self, df_tracks, m, offset_indices=False, random_mass=False, min_pt=0.):
    
    # If offset_indices is true, then offset the user_index by a large negative value
    # user_index_offset = 0
    # if offset_indices:
    #     user_index_offset = int(-1e6)
        
    # Apply a pt cut
    df_tracks_accepted = df_tracks[df_tracks.ParticlePt > min_pt]

    return df_tracks_accepted['ParticlePID'].values
