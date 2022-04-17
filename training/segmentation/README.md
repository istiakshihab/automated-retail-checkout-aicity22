# this sub-folder is for the segmentation pipeline for our approach

This code requires Python 3.8.12 and PyTorch 1.8.2.

To use this code:

* open a folder `datasets` in the root folder and keep `Auto-retail-syndata-release` inside it, see dataloader for details
* run `training/train_segmentation.py` which trains U-Net
* see `test_segmentation.ipynb` for making predictions and visualization on validation set using trained model
* see `test_segmentation.ipynb` for making predictions and visualization on validation set using trained model and our post processing approach

Note on running `test_segmentation.ipynb`: either run your trained model or get pretrained weights from [here](https://github.com/istiakshihab/automated-retail-checkout-aicity22/releases/tag/v0.0.1). If you use the pretrained model, open a folder `logs` in root dir and keep it there.

The main output from this sub-folder is the `*.pth` file of the trained U-Net model which is to be used in the final testing stage.

