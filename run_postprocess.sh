#!/bin/bash

# Run post processing
cd POSTPROCESSING 

python ./postprocessing.py --case testModal 
# julia ./POSTPROCESSING/postprocessing.jl

cd ..