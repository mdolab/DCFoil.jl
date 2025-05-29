#!/bin/bash

# @File          :   run.sh
# @Created       :   2025/05/26
# @Last modified :   2025/05/26
# @Author        :   Galen Ng
# @Desc          :   Cases I ran for the final paper

# 2025-05-26
# python run_OMDCfoil.py --task trim --name trim # trim aoa to meet lift [0/1]
# python run_OMDCfoil.py --task opt --fixStruct --name opt2 --restart dcfoil-trim # elliptical lift distrbution test case using only induced drag and wave drag as the objective
# 2025-05-28
# python run_OMDCfoil.py --task opt --name opt3 # test case with all DVs
python run_OMDCfoil.py --task opt --name opt4 # test case with all DVs and all static constraints