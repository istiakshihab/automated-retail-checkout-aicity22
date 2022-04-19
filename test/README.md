## Inference

Download `vit` and `unet` models from GitHub repository releases. Models are named as `vit_base_patch32_224` and `unet_aicityt4`respectively. Unzip them and keep the `vit_base_patch32_224.pt` and `unet_aicityt4.pth` files in models folder.
Alternatively, Download the models from this [link](https://drive.google.com/file/d/1J9psH6M5LwR09e0kGAsuvOnRMZRqVDH2/view?usp=sharing).  and store them as mentioned above.

Copy and paste your mp4 test videos on `test-videos` directory. 

Run `infer.py` to generate `submission.txt`
