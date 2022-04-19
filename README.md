# VISTA: Vision Transformer enhanced by U-Net and Image Colorfulness Frame Filtration for Automatic Retail Checkout
Multi-class product counting and recognition identifies product items from images or videos for automated retail checkout. The task is challenging due to the real-world scenario of occlusions where product items overlap, fast movement in conveyor belt, large similarity in overall appearance of the items being scanned, novel products, the negative impact of misidentifying items. Further there is a domain bias between training and test sets, specifically the provided training dataset consists of synthetic images and the test set videos consist of foreign objects such as hands and tray. To address these aforementioned issues, we propose to segment and classify individual frames from a video sequence. The segmentation method consists of a unified single product item- and hand-segmentation followed by entropy masking to address the domain bias problem. The multi-class classification method is based on Vision Transformers (ViT). To identify the frames with target objects, we utilize several image processing methods and propose a custom metric to discard frames not having any product items. Combining all these mechanisms, our best system achieves 3rd place in the AI City Challenge 2022 Track 4 with F1 score of 0.4545.


There are Two cascading Stages required to reproduce the result :

## Training
### Segmentation Model Training 
Go to the [`training/segmentation`](training/segmentation/) folder and follow the instructions presented in [`README.md`](training/segmentation/README.md) .

### Classification Model Training
Go to the [`training/classification`](training/classification/) folder and follow the instructions presented in [`README.md`](training/classification/README.md) .


## Inference 
* Make sure both segmentation and classification models are present in the [`test/models`](test/models/) directory. You can download our pretrained models using the [`download-model.sh`](test/download-model.sh) script. 

* test videos must be placed inside the [`test/test-videos`](test/test-videos/) folder. No files other than the videos should be in this directory.

* Follow the instructions in [`README.md`](test/README.md) to generate the submission file. 



