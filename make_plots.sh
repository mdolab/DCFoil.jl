#!/bin/bash

# Set the -e option so it exits if a command fails
set -e

# ==============================================================================
#                             Plots
# ==============================================================================
# Run post processing
cd postprocessing

# python ./run_postprocessing.py --case testModal
# python ./run_postprocessing.py --case testWater --debug_plots --is_modal
# python ./run_postprocessing.py --case testWaterAkcabay --is_flutter --is_forced
# julia ./POSTPROCESSING/postprocessing.jl

# # ************************************************
# #     IMOCA cases
# # ************************************************
# python ./run_postprocessing.py --case IMOCA60Keel_ss_f0.0_w0.0 --is_flutter --is_static
# # python ./run_postprocessing.py --case IMOCA60Keel_ss_f15.0_w0.0 --is_modal
# python ./run_postprocessing.py --case IMOCA60Keel_cfrp_f15.0_w0.0 --is_flutter --is_static --is_forced
# python ./run_postprocessing.py --case IMOCA60Keel_cfrp_f15.0_w0.0 --is_modal
# python ./run_postprocessing.py --case IMOCA60Keel_ss_f0.0_w0.0 IMOCA60Keel_cfrp_f15.0_w0.0 --is_static
# python ./run_postprocessing.py --case 2023-09-26_IMOCA60Keel_cfrp_f15.0_w0.0 --is_flutter --is_static --is_forced --is_modal
# python ./run_postprocessing.py --case 2023-09-26_IMOCA60Keel_cfrp_f15.0_w0.0 --is_flutter --is_static --is_forced --is_modal
# python ./run_postprocessing.py --case 2023-12-02_IMOCA60Keel_ss_f-10.0_w0.0 --is_flutter --is_static --is_forced --is_modal
# # python ./run_postprocessing.py --case 2023-12-03_IMOCA60Keel_wov-ud-wov_f15.0_w0.0 2023-12-03_IMOCA60Keel_cfrp_f15.0_w0.0 --is_flutter --is_static --is_forced --is_modal
# python ./run_postprocessing.py --case 2023-12-07_IMOCA60Keel_wov-ud-wov_f30.0_w0.0 --is_flutter --is_static --is_forced --is_modal
# python ./run_postprocessing.py --case 2023-12-03_IMOCA60Keel_cfrp_f15.0_w0.0 --is_flutter --is_static --is_forced --is_modal


# python ./run_postprocessing.py --case 2024-01-14_IMOCA60Keel_cfrp_f15.0_w0.0 --is_flutter --is_static --is_forced --is_modal --is_paper
# python ./run_postprocessing.py --case 2024-01-14_IMOCA60Keel_IM6-epoxy_f15.0_w0.0 --is_flutter --is_static --is_forced --is_modal --is_paper
# python ./run_postprocessing.py --case 2024-01-14_IMOCA60Keel_IM6-epoxy_f30.0_w0.0 --is_flutter --is_static --is_forced --is_modal --is_paper

# python ./run_postprocessing.py --case 2024-01-28_IMOCA60Keel_cfrp_f15.0_w0.0 --is_flutter --is_static --is_forced --is_modal --is_paper
# # --- Journal paper ---
# # python ./run_postprocessing.py --case 2024-01-14_IMOCA60Keel_ss_f-10.0_w0.0 --is_flutter --is_paper
# python ./run_postprocessing.py --cases 2024-02-24_IMOCA60Keel_cfrp_f15.0_w0.0 --is_flutter --is_paper
# python ./run_postprocessing.py --cases 2024-03-06_IMOCA60Keel_cfrp_f15.0_w0.0 2024-02-24_IMOCA60Keel_cfrp_f15.0_w0.0 --is_flutter --is_paper

# # ---------------------------
# #     No bulb IMOCA
# # ---------------------------
# python ./run_postprocessing.py --cases IMOCA60Keel_ss_f15.0_w0.0 IMOCA60Keelnobulb_ss_f15.0_w0.0 --is_modal
# python ./run_postprocessing.py --cases IMOCA60Keel_cfrp_f15.0_w0.0 IMOCA60Keelnobulb_cfrp_f15.0_w0.0 --is_modal

# ************************************************
#     Akcabay plots
# ************************************************
# python ./run_postprocessing.py --cases akcabay_f-15_w0 --is_flutter
# python ./run_postprocessing.py --cases akcabay-swept_cfrp_f15.0_w-15.0 --is_flutter --is_static --is_forced
# python ./run_postprocessing.py --cases akcabay_cfrp_f-15.0_w0.0 --is_flutter --is_forced
# python ./run_postprocessing.py --cases 2023-10-05_akcabay_cfrp_f-15.0_w0.0_comp2 --is_flutter --is_forced --is_modal 
# python ./run_postprocessing.py --cases 2023-09-26_akcabay-swept_cfrp_f15.0_w-15.0 --is_flutter --is_forced --is_modal
# python ./run_postprocessing.py --cases 2023-09-26_akcabay-swept_cfrp_f15.0_w-15.0_bt2 --is_modal --is_static --elem 0
# python ./run_postprocessing.py --cases 2023-09-26_akcabay-swept_cfrp_f15.0_w-15.0_comp2 --is_modal --is_static
# python ./run_postprocessing.py --cases 2023-10-06_akcabay-swept_cfrp_f15.0_w-15.0_comp2 2023-10-06_akcabay-swept_cfrp_f15.0_w-15.0_bt2 --is_flutter 
# python ./run_postprocessing.py --cases 2023-10-06_akcabay_cfrp_f-15.0_w0.0_comp2 2023-10-06_akcabay_cfrp_f-15.0_w0.0_bt2 --is_flutter 
# python ./run_postprocessing.py --cases  2023-11-06_akcabay-swept_cfrp_f15.0_w-15.0  --is_flutter 
# # --- PAPER ---
# # python ./run_postprocessing.py --cases 2024-01-08_akcabay-swept_cfrp_f15.0_w-15.0 --is_flutter --is_paper
# # python ./run_postprocessing.py --cases 2024-01-08_akcabay_cfrp_f-15.0_w0.0 --is_flutter --is_paper
# python ./run_postprocessing.py --cases 2024-02-24_akcabay-swept_cfrp_f15.0_w-15.0 --is_flutter --is_paper
# python ./run_postprocessing.py --cases 2024-02-24_akcabay_cfrp_f-15.0_w0.0 --is_flutter --is_paper

# python ./run_postprocessing.py --cases 2025-01-29_akcabay-swept_cfrp_f15.0_w-15.0 --is_flutter --is_paper
# python ./run_postprocessing.py --cases 2025-01-29_akcabay_cfrp_f-15.0_w0.0 --is_flutter --is_paper

# ************************************************
#     t-foil paper plots
# ************************************************
# python ./plot_polar.py --cases 2024-04-26_t-foil_cfrp_polar 2024-04-26_t-foil_ss_polar --is_paper

# ************************************************
#     Final T-foil cases
# ************************************************
# python ./run_postprocessing.py --cases 2024-12-16_mothrudder_cfrp_f0.0_w0.0 --is_forced --is_paper
# python ./run_postprocessing.py --cases 2024-12-30_mothrudder_cfrp_f0.0 --is_flutter --is_paper
# python ./run_postprocessing.py --cases 2025-01-05_mothrudder-fs_cfrp_f0.0 2025-01-05_mothrudder-nofs_cfrp_f0.0 --is_flutter --is_paper

# python ./run_postprocessing.py --cases 2025-01-07_mothrudder-nofs_cfrp_f0.0 --is_forced --is_paper
# python ./run_postprocessing.py --cases 2025-01-15_mothrudder-nofs_cfrp_f0.0 --is_forced --is_paper --is_flutter

# ************************************************
#     Thesis cases
# ************************************************
# opt1
python read_case.py --cases 2025-06-23-opt-p3-amcopt1 2025-06-23-trim-p123-amctrimMP --base 2025-06-23-trim-p123-amctrimMP --use_serif --shorternames opt1 trim1 --pts 3 --name 2025-06-23-opt-p3-amcopt1 --flutter
# opt1-fs
python read_case.py --cases 2025-06-23-opt-p3-amcopt1-fs 2025-06-23-opt-p3-amcopt1 --base 2025-06-23-opt-p3-amcopt1-fs --use_serif --shorternames opt1-fs opt1 --pts 3 --name 2025-06-23-opt-p3-amcopt1-fs --flutter
# opt2-fs
python read_case.py --cases 2025-06-24-opt-p3-opt2-fs 2025-06-23-trim-p123-amctrimMP --base 2025-06-23-trim-p123-amctrimMP --use_serif --shorternames opt2-fs trim1 --pts 3 --name 2025-06-24-opt-p3-opt2-fs --flutter
python read_case.py --cases 2025-06-24-opt-p3-opt2-fs 2025-06-23-trim-p123-amctrimMP-fs --base 2025-06-23-trim-p123-amctrimMP --use_serif --shorternames opt2-fs trim1 --pts 3 --name 2025-06-24-opt-p3-opt2-fs --flutter --forced

python read_case.py --cases 2025-06-24-opt-p3-opt2 2025-06-24-opt-p3-opt2-fs --base 2025-06-24-opt-p3-opt2-fs --use_serif --shorternames opt2 opt2-fs --pts 3 --name 2025-06-24-opt-p3-opt2 --flutter --forced

python read_case.py --cases 2025-06-25-opt-p123-opt2MP-fs 2025-06-23-trim-p123-amctrimMP --base 2025-06-23-trim-p123-amctrimMP --use_serif --shorternames opt3-fs trim1MP --pts 123 --name 2025-06-25-opt-p123-opt2MP-fs --flutter
python read_case.py --cases 2025-06-25-opt-p123-opt2MP-fs 2025-06-23-trim-p123-amctrimMP-fs --base 2025-06-23-trim-p123-amctrimMP --use_serif --shorternames opt3-fs trim1MP --pts 123 --name 2025-06-25-opt-p123-opt2MP-fs --flutter --forced
# python read_case.py --cases 2025-06-24-opt-p123-amcopt1MP 2025-06-23-trim-p123-amctrimMP --base 2025-06-23-trim-p123-amctrimMP --use_serif --shorternames opt1 trim1 --pts 3 --name 2025-06-24-opt-p123-amcopt1MP
# python read_case.py --cases 2025-06-23-opt-p123-amcopt1MP 2025-06-23-trim-p123-amctrimMP --base 2025-06-23-trim-p123-amctrimMP --use_serif --shorternames opt1 trim1 --pts 3 --name 2025-06-23-opt-p123-amcopt1MP
# python read_case.py --cases 2025-06-23-opt-p3-opt1 2025-06-23-trim-p123-amctrimMP --base 2025-06-23-trim-p123-amctrimMP --use_serif --shorternames opt1 trim1 --pts 3 --name 2025-06-23-opt-p3-opt1

cd ..

# ==============================================================================
#                             Movies
# ==============================================================================
# # ffmpeg -r <fps> -i <files>
# fps=50
# mkdir -p ./POSTPROCESSING/MOVIES/

# # ************************************************
# #     Debug output (handy for seeing the jumps)
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


# ************************************************
#     Hydroelastic mode migration
# ************************************************
# m=1 # mode number
# ffmpeg -r $fps -i ./POSTPROCESSING/PLOTS/IMOCA60KeelCFRP/hydroelastic-mode-shapes/mode$m_%04d.png movie.mp4
# mv movie.mp4 ./POSTPROCESSING/MOVIES/IMOCA60KeelCFRP/mode$m.mp4
