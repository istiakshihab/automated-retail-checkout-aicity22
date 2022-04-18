# conda activate timm-env
# accelerate launch --config_file accelerate_config.yaml timm_playground.py --data_dir dataset/
python3 clsf_training.py \
--data_dir <PATH TRAIN AND VALIDATION FOLDERS> \
--log_dir <PATH TO LOGGING DIRECTORY> \
--model_dir <PATH TO SAVED MODELS>
