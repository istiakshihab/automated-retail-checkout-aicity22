## Inference

Download `vit` and `unet` models from GitHub repository releases. Models are named as `vit_base_patch32_224` and `unet_aicityt4`respectively. Unzip them and keep the `vit_base_patch32_224.pt` and `unet_aicityt4.pth` files in models folder.
Alternatively, Download the models from this [link](https://drive.google.com/file/d/1J9psH6M5LwR09e0kGAsuvOnRMZRqVDH2/view?usp=sharing).  and store them as mentioned above.

Copy and paste test folder in this directory. For example, `test-videos` contains the mp4 files and `video_id.txt` file in this order:
```
training/
test/
    models/
    test-videos/
                testA_1.mp4
                testA_2.mp4
                testA_3.mp4
                ...
                video_id.txt
    infer.py
    ....
```

Run `infer.py "<folder-name>"` to generate `submission.txt`
For above example, it would be like:
`python infer.py "test-videos"`
