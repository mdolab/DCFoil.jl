#!/bin/bash

# @File          :   run.sh
# @Created       :   2025/05/26
# @Last modified :   2025/05/26
# @Author        :   Galen Ng
# @Desc          :   Cases I ran for the final paper

set -e # 

# --- 2025-05-26 ---
# python run_OMDCFoil.py --task trim --name trim # trim aoa to meet lift [0/1]
# python run_OMDCFoil.py --task opt --fixStruct --name opt2 --restart dcfoil-trim # elliptical lift distrbution test case using only induced drag and wave drag as the objective

# --- 2025-05-28 ---
# python run_OMDCFoil.py --task opt --name opt3 # test case with all DVs

# THIS IS CALLED opt2 in the paper
# python run_OMDCFoil.py --task opt --name opt4 # test case with all DVs and all static constraints [0/3]
# python run_OMDCFoil.py --task opt --name opt4-fs --freeSurf # test case with all DVs and all static constraints [0/3]

# --- 2025-05-30 ---
# python run_OMDCFoil.py --task opt --flutter --fixHydro --name opt5a # static and dynamic optimization with frozen geo variables
# These are opt3
# python run_OMDCFoil.py --task opt --flutter --restart dcfoil-trim --name opt5 # static and dynamic optimization [0/1] after increasing major step limit
# 2025-06-04 tweaking step limit size and scaling
# python run_OMDCFoil.py --task opt --flutter --freeSurf --restart dcfoil-trim --pts 3 --name opt5-fs # static and dynamic optimization [TODO NEXT GGGGGGGGGGGGGGGGGGGG randomly died probably due to memory issues] 

# --- 2025-06-02 ---
# Test multipoint
# python run_OMDCFoil.py --task trim --pts 123 --restart dcfoil-trim --name trimMP # multipoint [0/1]
# python run_OMDCFoil.py --task trim --freeSurf --pts 123 --restart dcfoil-trimMP --name trimMP-fs # multipoint [0/1]

# python run_OMDCFoil.py --task opt --pts 123 --restart dcfoil-trimMP --name opt1MP # multipoint  [0/3 need to post process running right now and into '-optMP.sql'; since this, I have scaled a bunch]
# python run_OMDCFoil.py --task opt --pts 123 --freeSurf --restart dcfoil-trimMPfs --name opt1MP-fs # multipoint [TODO]

# python run_OMDCFoil.py --task opt --flutter --pts 123 --restart dcfoil-trimMP --name opt1MP # multipoint [TODO]
# python run_OMDCFoil.py --task opt --flutter --pts 123 --freeSurf --restart dcfoil-trimMPfs --name opt1MP-fs # multipoint [TODO got 60/63 and lift was not met]


# ==============================================================================
#                             Switching to AMC full scale
# ==============================================================================
# python run_OMDCFoil.py --foil amcfull --task trim --pts 123 --name trimMP # multipoint [0/1]
# python run_OMDCFoil.py --foil amcfull --task trim --freeSurf --pts 123 --name trimMP-fs # multipoint [0/1]

python run_OMDCFoil.py --foil amcfull --task trim --restart dcfoil-trimMP --pts 123 --name trim1MP # multipoint []
python run_OMDCFoil.py --foil amcfull --task trim --restart dcfoil-trimMP-fs --freeSurf --pts 123 --name trim1MP-fs # multipoint []

python run_OMDCFoil.py --foil amcfull --task opt --restart dcfoil-trim1MP --pts 123 --name opt1MP # multipoint  []
python run_OMDCFoil.py --foil amcfull --task opt --restart dcfoil-trim1MP-fs --pts 123 --freeSurf --name opt1MP-fs # multipoint []

python run_OMDCFoil.py --foil amcfull --task opt --flutter --freeSurf --pts 3 --name opt2-fs # static and dynamic optimization

# may need to nohup ./run.sh > run.out 2>&1 