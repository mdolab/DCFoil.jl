#!/bin/bash

# Set the -e option so it exits if a command fails
set -e

# ==============================================================================
#                             Plots
# ==============================================================================
# Run post processing
cd POSTPROCESSING

# python ./run_postprocessing.py --case testModal
# python ./run_postprocessing.py --case testWater --debug_plots --is_modal
python ./run_postprocessing.py --case testWaterAkcabay --is_flutter
# julia ./POSTPROCESSING/postprocessing.jl

cd ..

# # ==============================================================================
# #                             Movies
# # ==============================================================================
# # ffmpeg -r <fps> -i <files>
# fps=50
# mkdir -p ./POSTPROCESSING/MOVIES/

# # ************************************************
# #     Debug output
# # ************************************************
# ffmpeg -r $fps -i ./DebugOutput/kCross-qiter-%03d.png movie.mp4
# mv movie.mp4 ./POSTPROCESSING/MOVIES/kCrossings.mp4

# ffmpeg -r $fps -i ./DebugOutput/Vf-qiter-%03d.png movie.mp4
# mv movie.mp4 ./POSTPROCESSING/MOVIES/V-f.mp4

# ffmpeg -r $fps -i ./DebugOutput/Vg-qiter-%03d.png movie.mp4
# mv movie.mp4 ./POSTPROCESSING/MOVIES/V-g.mp4

# ffmpeg -r $fps -i ./DebugOutput/RL-qiter-%03d.png movie.mp4
# mv movie.mp4 ./POSTPROCESSING/MOVIES/RL.mp4

# ffmpeg -r $fps -i ./DebugOutput/corr-%03d.pdf movie.mp4
# mv movie.mp4 ./POSTPROCESSING/MOVIES/corr.mp4
