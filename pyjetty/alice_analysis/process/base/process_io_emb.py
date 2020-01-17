#!/usr/bin/env python3

"""
  Analysis IO class for jet analysis with track dataframe.
  The class stores a list of Pb-Pb files, and keeps track of
  a current file and current event -- and returns the current
  event when requested.
  
  Authors: James Mulligan
           Mateusz Ploskon
"""

from __future__ import print_function

# Data analysis and plotting
import uproot
import pandas
import numpy as np
import random

# Base class
from pyjetty.alice_analysis.process.base import common_base

# Main file IO class
from pyjetty.alice_analysis.process.base import process_io

################################################################
class process_io_emb(common_base.common_base):
  
  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, file_list='PbPb_file_list.txt', track_tree_name='tree_Particle', **kwargs):
    super(process_io_emb, self).__init__(**kwargs)
    
    self.file_list = file_list
    self.track_tree_name = track_tree_name
    
    self.list_of_files = []
    with open(self.file_list) as f:
        self.list_of_files = [fn.strip() for fn in f.readlines()]
    
    self.current_file_df = None
    self.current_file_nevents = 0
    self.current_event_index = 0
  
    # Initialize by loading a file
    self.load_file()
    
  #---------------------------------------------------------------
  # Return current event in current file, and increment current_event.
  # If there are no more events in the current file, open a new file.
  #---------------------------------------------------------------
  def load_event(self):
      
    if self.current_event_index >= self.current_file_nevents:
        self.load_file()
    
    current_event = self.current_file_df[self.current_event_index]
    self.current_event_index += 1
    #print('Get Pb-Pb event {}/{}'.format(self.current_event_index, self.current_file_nevents))
    return current_event

  #---------------------------------------------------------------
  # Pick a random file from the file list, and load it as the
  # current file as a dataframe.
  #---------------------------------------------------------------
  def load_file(self):
    
    input_file = random.choice(self.list_of_files)
    print('Opening Pb-Pb file: {}'.format(input_file))

    io = process_io.process_io(input_file=input_file, track_tree_name=self.track_tree_name)
    self.current_file_df = io.load_data(offset_indices=True)
    self.current_file_nevents = len(self.current_file_df.index)
    self.current_event_index = 0