# export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1
source set_env.sh
# python model/magic_point_v2.py
# python export_detections_repeatability.py
python homo_export_labels_color.py