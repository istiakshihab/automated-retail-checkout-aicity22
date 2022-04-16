# Segment objects from synthetic images

This code requires Python 3.8.12 and PyTorch 1.8.2.

To use this code:

* clone repo
* open a folder `datasets` in the root folder and keep `Auto-retail-syndata-release` inside it
* run `train_segmentation.ipynb` which trains U-Net
* run `test_segmentation.ipynb` for making predictions and visualization on validation set using trained model

note on running `test_segmentation.ipynb`: either run your trained model or get pretrained weights from [here](https://github.com/acc-track-4/product-segmentation/releases/tag/v0.0.1). If you use the pretrained model, open a folder `logs` in root dir and keep it there.


Todo: 
* inference (give one image/frame, get mask)

