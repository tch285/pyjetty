#!/usr/bin/bash

nev=100000

workdir=/global/cfs/cdirs/alice/$USER/fspyjetty # set your own workdir here
module load python/3.11
source ${workdir}/pyjettyenv/bin/activate
module use /global/cfs/cdirs/alice/heppy_soft/15-09-2024/yasp/software/modules
module load cmake gsl root/default HepMC2/2.06.11 LHAPDF6/6.5.4 pcre2/default swig/4.1.1 HepMC3/3.2.5
module use ${workdir}/pyjetty/modules
module load pyjetty/1.0

cd $workdir/pyjetty/pyjetty/alice_analysis/

python3 process/user/fastsim_run3/tree_fs.py \
    -o ${workdir}/results.root \
    -c config/fastsim/pythia_basic.yaml \
    --py-cmnd config/fastsim/pythia_settings.cmnd \
    --nev $nev \
    --py-seed 1111
