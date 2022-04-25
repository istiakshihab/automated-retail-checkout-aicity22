# conda activate timm-env
# accelerate launch --config_file accelerate_config.yaml timm_playground.py --data_dir dataset/

python3 synth_image_creator.py \
--source_dir dataset/Auto-retail-syndata-release/syn_image_train \
--target_dir dataset/Auto-retail-syndata-release/bgr_images \
--segmentation_dir dataset/Auto-retail-syndata-release/segmentation_labels


python3 databuilder.py --data_dir dataset/Auto-retail-syndata-release/bgr_images 


python3 clsf_training.py \
--data_dir dataset/Auto-retail-syndata-release/ \
--log_dir training_logs \
--model_dir models