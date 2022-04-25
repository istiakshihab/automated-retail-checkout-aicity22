python3 synth_image_creator.py \
--source_dir <PATH TO RAW IMAGE DATASET> \
--target_dir <PATH TO BACKGROUND REMOVED IMAGES DIRECTORY> \
--segmentation_dir <PATH TO SEGMENTATION LABELS>


python3 databuilder.py --data_dir <PATH TO BACKGROUND REMOVED IMAGES> 


python3 clsf_training.py \
--data_dir <PARENT DIRECTORY OF TRAIN AND VALIDATION FOLDERS> \
--log_dir <PATH TO LOGGING DIRECTORY> \
--model_dir <PATH TO SAVED MODELS>