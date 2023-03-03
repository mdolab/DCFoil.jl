#!/bin/bash

############################################################
# This executable script runs program from terminal 
############################################################

set -e

# Run the program
# nohup julia main.jl > IMOCACFRP.out
nohup julia runscripts/main_imoca-ss.jl > IMOCASS.out
# nohup julia main.jl > akcabay_forced.out
