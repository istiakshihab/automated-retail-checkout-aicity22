wget https://github.com/istiakshihab/automated-retail-checkout-aicity22/releases/download/classification/vit_base_patch32_224.zip
wget https://github.com/istiakshihab/automated-retail-checkout-aicity22/releases/download/v0.0.1/unet_aicityt4.zip

unzip vit_base_patch32_224.zip
unzip unet_aicityt4.zip

mv vit_base_patch32_224.pt models/
mv unet_aicityt4.pt models/

rm vit_base_patch32_224.zip
rm unet_aicityt4.zip
