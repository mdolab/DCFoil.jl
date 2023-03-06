#!/bin/bash

############################################################
# This executable script runs program from terminal
############################################################

set -e

# Run the program
nohup julia runscripts/main_imoca-ss.jl &
