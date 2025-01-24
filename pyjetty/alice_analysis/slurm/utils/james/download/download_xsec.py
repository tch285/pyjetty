#!/usr/bin/env python3

'''
Script to download train output from AliEn.

On hiccup:
  - ssh to hiccupds
  - start a screen session
  - enter alidock, then `alienv enter AliRoot/latest`, then get token
  - python download_data2.py -c LHC20g4.yaml

On perlmutter:
  - start screen session with `screen`
  - enter shifter image with `shifter --image=docker:sweisz/alice-grid-strace:1.0 --module=cvmfs`
  - enter environment with `/cvmfs/alice.cern.ch/bin/alienv enter AliPhysics/vAN-20241023_O2-1`
    or the AliPhysics environment of your choice
  - generate AliEn token with `alien-token-init`
  - run script with `python download_data.py -c LHC20g4.yaml`
  
Note that if token expires or otherwise crashes, the script will automatically detect where to start copying again

'''

import argparse
import logging
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import time
import re
from pathlib import Path

import yaml
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logger(level):
    logger = logging.getLogger(__name__)
    level = getattr(logging, level.upper())
    logger.setLevel(level)
    tqdm_handler = TqdmLoggingHandler()
    # tqdm_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'))
    tqdm_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(tqdm_handler)
    return logger

def setup_process_logger(idx, run, level = 'DEBUG'):
    logger = logging.getLogger(f"{__name__}.{idx}")
    level = getattr(logging, level.upper())
    logger.setLevel(level)
    logger.propagate = False
    file_handler = logging.FileHandler(f"log_{run}.log")
    file_handler.setLevel(level)
    # file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'))
    file_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    return logger

#---------------------------------------------------------------------------
def download_data(config_file, log_level):
    # global logger
    logger = setup_logger(log_level)

    if not args.config.is_file():
        logger.critical(f'File "{args.config}" does not exist; exiting.')
        sys.exit(404)
    
    # Initialize config
    logger.info('Configuring...')
    logger.info(f'Configuration file set: "{args.config}"')

    with open(config_file, 'r') as stream:
      config = yaml.safe_load(stream)
    
    alien_dir = config['alien_dir']
    runlist = config['runlist']
    nruns = len(runlist)
    output_dir = config['output_dir']
    pt_hat_bins = config['pt_hat_bins']
    n_pt_hat_bins = len(pt_hat_bins)
    is_AOD = config['is_AOD']
    max_attempts = config['max_attempts'] if 'max_attempts' in config else 5
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.chdir(output_dir)
    logger.info(f'Output: {output_dir}')

    prog_queue = mp.Queue()
    nloops = n_pt_hat_bins * nruns
    
    processes = []
    # Loop through runs, and start a download for each run in parallel
    logger.info(f"Downloading {nruns} runs.")
    start = time.time()
    for run_idx, run in enumerate(runlist):
        p = mp.Process(target=download_run, args=(run_idx, prog_queue, alien_dir, run, pt_hat_bins, is_AOD, max_attempts))
        processes.append(p)
        p.start()

    total_prog_bar = tqdm(total=nloops, unit='loop', desc='Overall', position=nruns)

    run_prog_bars = [tqdm(total=n_pt_hat_bins, unit='bin', desc=f"{i+1:02}: {run}", position=i) for i, run in enumerate(runlist)]
    completed_runs = 0
    while completed_runs < nruns:
        try:
            run_idx, increment = prog_queue.get(timeout=0.5)
            if increment == "DONE":
                completed_runs += 1
                total_prog_bar.update(1)
            elif isinstance(increment, tuple):
                _, nsubruns = increment
                run_prog_bars[run_idx].reset(total = nsubruns)
                total_prog_bar.total = total_prog_bar.total + nsubruns
                total_prog_bar.refresh()
            else:
                total_prog_bar.update(increment)
                run_prog_bars[run_idx].update(increment)
        except queue.Empty:
            pass

    for p in processes:
        p.join()
    
    for bar in run_prog_bars:
        bar.close()
    total_prog_bar.close()

    logger.info(f"Download complete: {format_time_difference(time.time() - start)} elapsed.")

#---------------------------------------------------------------------------
def download_run(run_idx, prog_queue, alien_dir, run, pt_hat_bins, is_AOD, max_attempts):
    logger = setup_process_logger(run_idx, run)
    logger.info(f"Run {run}: starting download.")

    for pt_hat_bin in pt_hat_bins:
        logger.info(f"Bin {pt_hat_bin}: starting download.")
        train_output_dir = f'{alien_dir}/{pt_hat_bin}/{run}'
        if is_AOD:
            train_output_dir += '/AOD'
        download(train_output_dir, run, pt_hat_bin, max_attempts, logger, prog_queue, run_idx)
        logger.info(f"Bin {pt_hat_bin}: download complete.")
        prog_queue.put((run_idx, 1))

    prog_queue.put((run_idx, "DONE"))
    logger.info(f"Run {run}: download complete.")

#---------------------------------------------------------------------------
def download(train_output_dir, run, pt_hat_bin, max_attempts, logger, queue, run_idx):
    logger.debug(f'Train output dir: {train_output_dir}')
    run_path = f'{pt_hat_bin}/{run}'

    # Construct list of subdirectories (i.e. list of files to download)
    cmd = f'alien_ls {train_output_dir}'
    result = subprocess.run(cmd, shell = True, encoding='utf-8', stdout=subprocess.PIPE)
    if result.returncode != 0:
        logger.error(f'alien_ls failed with code {result.returncode}')
        return
    output = result.stdout
    all_subdirs = [line.strip().rstrip('/') for line in output.split('\n') if line.endswith('/')]
    subruns = [subdir for subdir in all_subdirs if re.match(r"^\d+$", subdir)]
    logger.info(f"Subruns: {subruns}")

    # if 'data' in train_output_dir:
    queue.put((run_idx, ("UPDATE", len(subruns))))

    # Remove any empty directories
    if os.path.exists(run_path):
        logger.warning(f"Empty directory found: {run_path}")
        subprocess.run(f'find {run_path} -empty -type d -delete', shell = True)

    # Copy the files
    for subrun in subruns:
        # Skip any directory that already exists
        subrun_path = f'{run_path}/{subrun}'
        if not os.path.exists(subrun_path):
            os.makedirs(subrun_path)
        else:
            queue.put((run_idx, 1))
            continue
        
        # find the stat file
        for i in range(max_attempts):
            cmd = f'alien_ls {train_output_dir}/{subrun}'
            try:
                result = subprocess.run(cmd, shell = True, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                output = result.stdout
                all_files = [filename.strip() for filename in output.splitlines()]
                if not all_files:
                    statfile = ""
                    break
                else:
                    statfile_matches = [filename for filename in all_files if re.match(r'^\d+_\d+_\d+_\d+\.stat$', filename)]
                    xsecfile_matches = [filename for filename in all_files if filename == "pyxsec_hists.root"]
                    if len(statfile_matches) == 1 and len(xsecfile_matches) == 1:
                        statfile = statfile_matches[0]
                    elif len(statfile_matches) == 0 and len(xsecfile_matches) == 0:
                        logger.warning(f"Could not find either file for subrun {subrun}, directory contains {all_files}")
                        statfile = ""
                    elif len(statfile_matches) == 1 and len(xsecfile_matches) == 0:
                        if statfile_matches[0] == "0_0_0_0.stat":
                            logger.warning(f"Missing xsec but zero statfile. {subrun}, directory contains {all_files}")
                            statfile = ""
                        else:
                            logger.error(f"Missing xsec but nonzero statfile in {subrun}: {all_files}")
                            raise ValueError(f"Missing xsec but nonzero statfile in {subrun}: {all_files}")
                    else:
                        logger.error(f"Missing statfile but xsecfile exists in {subrun}: {all_files}")
                        raise ValueError(f"Missing statfile but xsecfile exists in {subrun}: {all_files}")
                    break
            except subprocess.CalledProcessError as e:
                logger.warning(f'Statfile finding with alien_ls failed with code {e.returncode} on try {i+1}/{max_attempts}, reattempting...')
        else:
            logger.error(f"Statfile finding failed {max_attempts} times: {cmd}")

        if statfile:
            files_to_copy = []
            if not os.path.isfile(f'{subrun_path}/{statfile}'):
                files_to_copy.append(statfile)
            else:
                logger.info(f"Found statfile in {subrun_path}/{statfile}, skipping.")
            if not os.path.isfile(f'{subrun_path}/pyxsec_hists.root'):
                files_to_copy.append("pyxsec_hists.root")
            else:
                logger.info(f"Found xsec in {subrun_path}/pyxsec_hists.root, skipping.")

            for filename in files_to_copy:
                for i in range(max_attempts):
                    cmd = f'alien_cp -f alien:{train_output_dir}/{subrun}/{filename} file:{subrun_path}' # modified: file: prefix needed
                    logger.info(f"Copy: {cmd}")
                    try:
                        result = subprocess.run(cmd, check=True, encoding='utf-8', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                        for line in result.stdout.splitlines():
                            logger.info(line)
                        break
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"Copy failed with return code {e.returncode} on try {i+1}/{max_attempts}, reattempting...")
                        if os.path.isfile(f"{subrun_path}/{filename}"):
                            os.remove(f"{subrun_path}/{filename}")
                            logger.info("File removed.")
                        else:
                            logger.info("File not found, no removal.")
                else:
                    logger.error(f"Copy failed {max_attempts} times: {cmd}")
        queue.put((run_idx, 1))


def format_time_difference(time_diff):
    hours, remainder = divmod(time_diff, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"

#----------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download train output')
    parser.add_argument('-c', '--config', action = 'store',
                        type = Path, metavar = 'config',
                        default = 'config.yaml',
                        help = 'Path of config file')
    parser.add_argument("--log", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help = "Set the logging level")

    args = parser.parse_args()

    download_data(config_file = args.config, log_level = args.log)
