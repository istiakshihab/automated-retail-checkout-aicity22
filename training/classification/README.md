## Training Folder Structure

The training folder is structured as followed: 
```
.
├── logs
├── models
├── README.md
├── synth_image_creator.py
├── databuilder.py
├── experiment_log.py
├── clsf_training.py
├── complete_process.sh
└── train_only.sh
```



`training_logs` : Logs generated after model training based on the info defined in `experiment_log.py`. This path is defined by the *log_dir* parameter in the *train.sh* file and **must be created manually, if not already exists**

`models` : Best model generated after model training . This path is defined by the *model_dir* parameter in the *train.sh* file and **must be created manually, if not already exists**

`synth_image_creator.py` : Processing file used to convert raw image data to background replaced training data.

`databuilder.py` :  python script for structuring background removed images into label-wise folders. 

`experiment_log.py` : defines how logs are stored in the log directory. 

`clsf_training.py` : main classifier training file. All the used hyperparameters are internally defined. 

`train_only.sh` : standalone ViT training code, on backgroudn replaced and structured imaged dataset. 

`complete_process.sh` : combines background replacement, folder structuring and training code in the same pipeline. 

## Training Procedure 
To run the whole training from scratch, use the [`complete_process.sh`](complete_process.sh) script. The script has three stages. 
1. Preprocessing raw image data from AiCity-Track 4 to replace their background with a programmatically simulated tray background

2. Relocate the background-replace images into folders according to their labels. 

3. Run the training script and save the best model in a defined folder. 

Here's an example script we have used :
```python
python3 synth_image_creator.py \
--source_dir ../dataset/raw_images \
--target_dir ../dataset/bgr_images \
--segmentation_dir ../dataset/segmentation_labels


python3 databuilder.py \ 
--data_dir ../dataset/bgr_images \ 
--train_file_ratio 0.8 \


python3 clsf_training.py \
--data_dir ../dataset \
--log_dir training_logs/ \
--model_dir ../test/models/
```

To only run training on already preprocessed and formatted dataset, use the `training_only.sh` and set the arguments accordingly. 


