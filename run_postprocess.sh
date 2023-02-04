#!/bin/bash

# ==============================================================================
#                             Plots
# ==============================================================================
# Run post processing
cd POSTPROCESSING

# python ./run_postprocessing.py --case testModal
python ./run_postprocessing.py --case testModal --debug_plots
# julia ./POSTPROCESSING/postprocessing.jl

cd ..

# # ==============================================================================
# #                             Movies
# # ==============================================================================
# # ffmpeg -r <fps> -i <files>
# fps=30
# mkdir ./POSTPROCESSING/MOVIES/

# # ************************************************
# #     Debug output
# # ************************************************
# ffmpeg -r $fps -i ./DebugOutput/kCross-qiter-%03d.png movie.mp4
# mv movie.mp4 ./POSTPROCESSING/MOVIES/kCrossings.mp4
