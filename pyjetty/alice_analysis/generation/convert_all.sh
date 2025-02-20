#!/usr/bin/bash

# AHADIC files
# afiles=("/rstorage/ploskon/eec_sherpa_inclusive_5TeV/sherpa2x/jetpt20/sherpa_LHC_jets_20.hepmc"
#         "/rstorage/ploskon/eec_sherpa_inclusive_5TeV/sherpa2x/jetpt40/sherpa_LHC_jets_40.hepmc"
#         "/rstorage/ploskon/eec_sherpa_inclusive_5TeV/sherpa2x/jetpt60/sherpa_LHC_jets_60.hepmc"
# )
afiles=("/rstorage/ploskon/eec_sherpa_inclusive_5TeV/sherpa2x/jetpt40/sherpa_LHC_jets_40.hepmc"
        "/rstorage/ploskon/eec_sherpa_inclusive_5TeV/sherpa2x/jetpt60/sherpa_LHC_jets_60.hepmc"
)

# aoutput=("/rstorage/mhwang/sherpa/ahadic_20"
#          "/rstorage/mhwang/sherpa/ahadic_40"
#          "/rstorage/mhwang/sherpa/ahadic_60"
# )
aoutput=("/rstorage/mhwang/sherpa/ahadic_40"
         "/rstorage/mhwang/sherpa/ahadic_60"
)

# LUND files
lfiles=("/rstorage/ploskon/eec_sherpa_inclusive_5TeV/sherpa2x/jetpt20_lund/sherpa_LHC_jets_20.0.hepmc"
        "/rstorage/ploskon/eec_sherpa_inclusive_5TeV/sherpa2x/jetpt40_lund/sherpa_LHC_jets_40.0.hepmc"
        "/rstorage/ploskon/eec_sherpa_inclusive_5TeV/sherpa2x/jetpt60_lund/sherpa_LHC_jets_60.0.hepmc"
)

loutput=("/rstorage/mhwang/sherpa/lund_20"
         "/rstorage/mhwang/sherpa/lund_40"
         "/rstorage/mhwang/sherpa/lund_60"
)

for i in $(seq 0 $((${#afiles[@]} - 1))); do
  idx=$i
  filename=${afiles[$i]}
  outname=${aoutput[$i]}
  echo "$idx"
  echo "$filename"
  echo "$outname"
  python3 hepmc2antuple_tn.py -i "$filename" -o "$outname" -g sherpa --no-progress-bar -d --hepmc 3 -s 50000
done

for j in $(seq 0 $((${#lfiles[@]} - 1))); do
  idx=$j
  filename=${lfiles[$j]}
  outname=${loutput[$j]}
  echo "$idx"
  echo "$filename"
  echo "$outname"
  python3 hepmc2antuple_tn.py -i "$filename" -o "$outname" -g sherpa --no-progress-bar -d --hepmc 3 -s 50000
done