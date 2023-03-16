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
# python ./run_postprocessing.py --case testWaterAkcabay --is_flutter --is_forced
# julia ./POSTPROCESSING/postprocessing.jl

# # ************************************************
# #     IMOCA cases
# # ************************************************
python ./run_postprocessing.py --case IMOCA60KeelSS --is_flutter
# python ./run_postprocessing.py --case IMOCA60Keel_ss_f15.0_w0.0 --is_modal
python ./run_postprocessing.py --case IMOCA60KeelCFRP --is_flutter
# python ./run_postprocessing.py --case IMOCA60Keel_cfrp_f15.0_w0.0 --is_modal

# # ---------------------------
# #     No bulb IMOCA
# # ---------------------------
# python ./run_postprocessing.py --cases IMOCA60Keel_ss_f15.0_w0.0 IMOCA60Keelnobulb_ss_f15.0_w0.0 --is_modal
# python ./run_postprocessing.py --cases IMOCA60Keel_cfrp_f15.0_w0.0 IMOCA60Keelnobulb_cfrp_f15.0_w0.0 --is_modal

# ************************************************
#     Akcabay plots
# ************************************************
# python ./run_postprocessing.py --case akcabay_f-15_w0 --is_flutter
# python ./run_postprocessing.py --case akcabay-swept_cfrp_f15.0_w-15.0 --is_flutter

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
