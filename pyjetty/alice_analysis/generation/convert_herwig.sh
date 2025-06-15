#!/usr/bin/bash

cd /global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/herwig73_alice/tree_gen
JOBID=38545055
mkdir -p "$JOBID"
cd "$JOBID"

JOB_DIR=/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/herwig73_alice/hepmc/$JOBID

for pthatbin in $( seq 1 20 ); do
  for core in $( seq 1 50 ); do
# for pthatbin in $( seq 1 1 ); do
#   for core in $( seq 1 1 ); do
    filename=$(find "$JOB_DIR/$pthatbin/$core/" -type f -name "*.hepmc" -print -quit)
    output_dir="$pthatbin/$core/"
    echo "Input file to be converted: $filename"
    echo "Output directory: $output_dir"
    python3 /global/cfs/cdirs/alice/mhwang/mypyjetty/pyjetty/pyjetty/alice_analysis/generation/hepmc2antuple_tn.py -i "$filename" -o "$output_dir" -g herwig --no-progress-bar -d --hepmc 2 -s 0
  done
done
