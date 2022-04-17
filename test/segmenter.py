import torch
import torch.nn as nn
from torchvision import datasets, transforms, utils
import torch.hub
import torchvision
import numpy as np

from PIL import Image
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters.rank import entropy
from skimage.morphology import disk

#To use this block, run `pip install pytorch-lightning`
#Reference: https://github.com/guglielmocamporese/hands-segmentation-pytorch

torch.manual_seed(0)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using: ", DEVICE)


# Full path to the product segmentation model
model_path = './models/unet_aicityt4.pth'


########### Product segmentation model ###########
""" Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
"""
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        
        # NOTE: 
        # nn.Conv2d(64, 1, kernel_size=1, padding=0) is mathematically same as 
        # nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)
        return outputs

# Define model
segmentation_model = build_unet()
checkpoint = torch.load(model_path, map_location="cpu")
segmentation_model.load_state_dict(checkpoint)
# Send to GPU
segmentation_model = segmentation_model.to(DEVICE) # runs on GPU
segmentation_model.eval()

########### Hand segmentation model ###########

# Create the model
hand_seg_model = torch.hub.load(
    repo_or_dir='guglielmocamporese/hands-segmentation-pytorch', 
    model='hand_segmentor', 
    pretrained=True
)

hand_seg_model.eval() # runs on CPU


transforms_image = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.CenterCrop((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


########### Entropy masking ###########

def entropy_based_seg(img):
    image_gray = rgb2gray(img)
    image_gray = img_as_ubyte(image_gray)
    entropy_image = entropy(image_gray, disk(6)) # 6 default
    scaled_entropy = entropy_image / entropy_image.max()
    threshold = scaled_entropy > 0.8 # 0.8 default
    image_seg = np.dstack([img[:,:,0]*threshold,
                            img[:,:,1]*threshold,
                            img[:,:,2]*threshold])
    return image_seg


## This is the final segmentation step we use for our approach
def crop_with_hand_entropy_seg(img):
    """
    Crops product image using product, hand and entropy masking based segmentation.
    
    Reference: https://towardsdatascience.com/image-processing-with-python-working-with-entropy-b05e9c84fc36
    """
        
    image_b = torch.unsqueeze(img, 0).to(img) # same data type as the variable 'img'
    image = (image_b[0].permute(1,2,0).detach().cpu().numpy()+1)/2
    image = (image*255).astype(np.uint8)
    pred = hand_seg_model(image_b).argmax(1).detach().cpu().numpy().squeeze()
    # invert mask, use plt.imshow(hand_pred) for vis
    hand_pred = np.logical_not(pred).astype(int) 
    
    # Stack preds for getting ROI
    hand_pred3d = np.stack((hand_pred, hand_pred, hand_pred), axis=-1)
    # Get rid off hands
    image_no_hands = np.array(image) * hand_pred3d # use plt.imshow(image_no_hands) for vis
    
    # Convert new image to tensor
    image_no_hands = Image.fromarray(np.uint8(image_no_hands)).convert('RGB')
    image_tensor = transforms_image(image_no_hands)

    image_tensor_b = torch.unsqueeze(image_tensor, 0).to(DEVICE)
    output = torch.sigmoid(segmentation_model(image_tensor_b.float()))
    pred = output.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy() > 0.5
    
    # Stack preds for getting ROI
    mask = np.stack((pred, pred, pred), axis=-1)
    
    # Image ROI
    image_roi = image_no_hands * mask
    
    image_roi = entropy_based_seg(image_roi)
    
    data = {"roi": image_roi,
            "mask": pred,
            "image": image
           }
    
    return data
