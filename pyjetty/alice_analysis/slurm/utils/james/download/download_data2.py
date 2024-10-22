#!/usr/bin/env python3

'''
Script to download Pb-Pb train output from AliEn.

python download_data.py -p LHC18q

On hiccup:
  - ssh to hiccupds
  - start a screen session
  - enter alidock, then `alienv enter AliRoot/latest`, then get token
  - python download_data.py -c LHC18q.yaml

On perlmutter:
  - start screen session with `screen`
  - enter shifter image with `shifter --image=docker:sweisz/alice-grid-strace:1.0 --module=cvmfs`
  - generate AliEn token with `alien-token-init`
  - enter environment with `/cvmfs/alice.cern.ch/bin/alienv enter VO_ALICE@AliPhysics::vAN-20220719_ROOT6-1`
    or the AliPhysics environment of your choice
  - run script with `python download_data.py -c LHC18q.yaml`
  
Note that if token expires or otherwise crashes, the script will automatically detect where to start copying again

'''

import argparse
import os
import sys
import yaml
import subprocess
import multiprocessing as mp
from pathlib import Path

#---------------------------------------------------------------------------
def download_data(config_file, nattempts):

    # Initialize config
    with open(config_file, 'r') as stream:
      config = yaml.safe_load(stream)

    period = config['period']
    parent_dir = config['parent_dir']
    year = config['year']
    train_name = config['train_name']
    train_PWG = config['train_PWG']
    train_number = config['train_number']
    runlist = config['runlist']
    output_dir = config['output_dir']
    
    if 'pt_hat_bins' in config:
        pt_hat_bins = config['pt_hat_bins']
    else:
        pt_hat_bins = None

    # Create output dir and cd into it
    output_dir = os.path.join(output_dir, period)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    os.chdir(output_dir)
    print(f'output dir: {output_dir}')
    
    # Loop through runs, and start a download for each run in parallel
    for run in runlist:
        p = mp.Process(target=download_run, args=(parent_dir, year, period, run, train_PWG, train_name, train_number, pt_hat_bins, nattempts))
        p.start()

#---------------------------------------------------------------------------
def download_run(parent_dir, year, period, run, train_PWG, train_name, train_number, pt_hat_bins, nattempts):

    if parent_dir == 'data':
    
        train_output_dir = f'/alice/{parent_dir}/{year}/{period}/{run}/{train_PWG}/{train_name}/{train_number}'
        
        download(train_output_dir, run, nattempts)
        
    elif parent_dir == 'sim':
    
        for pt_hat_bin in pt_hat_bins:
            train_output_dir = f'/alice/{parent_dir}/{year}/{period}/{pt_hat_bin}/{run}/{train_PWG}/{train_name}/{train_number}'
            
            download(train_output_dir, run, pt_hat_bin, nattempts)

#---------------------------------------------------------------------------
def download(train_output_dir, run, pt_hat_bin=None, nattempts = 5):

    # print(f'train_output_dir: {train_output_dir}')
    
    if pt_hat_bin:
        run_path = f'{pt_hat_bin}/{run}'
    else:
        run_path = run

    # Construct list of subdirectories (i.e. list of files to download)
    temp_filelist_name = f'subdirs_temp_{run}.txt'
    cmd = f'alien_ls {train_output_dir} > {temp_filelist_name}'
    result = subprocess.run(cmd, shell = True)
    if result.returncode != 0:
        print(f'alien_ls failed with code {result.returncode}')
    with open(temp_filelist_name) as f:
        subdirs_all = f.read().splitlines()
        subdirs = [ x[:-1] for x in subdirs_all if x[-1] == "/" ] # fixed: identifies subdirectories
    os.remove(temp_filelist_name)

    # Remove any empty directories
    if os.path.exists(run_path):
        subprocess.run(f'find {run_path} -empty -type d -delete', shell = True)

    # Copy the files
    for subdir in subdirs:
        
        # Skip any directory that already exists
        subdir_path = f'{run_path}/{subdir}'
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
            print(f'downloading: {subdir_path}')
        else:
            continue
        
        for i in range(nattempts):
            with open(f'log_{run}.txt', "a") as logfile:
                cmd = f'alien_cp -f alien:{train_output_dir}/{subdir}/AnalysisResults.root file:{subdir_path}' # modified: file: prefix needed
                print(cmd, file=logfile)
                try:
                    subprocess.run(cmd, check=True, shell=True, stdout=logfile, stderr=logfile)
                    break
                except subprocess.CalledProcessError as e:
                    print(f"alien_cp failed with return code {e.returncode} on try {i+1}/{nattempts}, reattempting...", file=logfile)
                    if os.path.isfile(f"{subdir_path}/AnalysisResults.root"):
                        os.remove(f"{subdir_path}/AnalysisResults.root")
                        print("File removed.")
                    else:
                        print("File not found, no removal.")
        else:
            with open(f'errors_{run}.txt', "a") as errfile:
                print(cmd, file=errfile)

        

#----------------------------------------------------------------------
if __name__ == '__main__':

    # Define arguments
    parser = argparse.ArgumentParser(description='Download train output')
    parser.add_argument('-c', '--config', action='store',
                        type=Path, metavar='config',
                        default='config.yaml',
                        help='Path of config file')
    parser.add_argument('-n', '--nattempts',
                        type=int, default=5,
                        help='Number of alien_cp attempts before stopping')

    # Parse the arguments
    args = parser.parse_args()

    print('Configuring...')
    print(f'Configuration file set: "{args.config}"')

    # If invalid configFile is given, exit
    if not args.config.is_file():
        print(f'File "{args.config}" does not exist; exiting.')
        sys.exit(404)

    download_data(config_file = args.config, nattempts = args.nattempts)
