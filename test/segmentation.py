from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters.rank import entropy
from skimage.morphology import disk

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch
torch.manual_seed(0)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transforms_image = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.CenterCrop((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#To use this block, run `pip install pytorch-lightning`
#Reference: https://github.com/guglielmocamporese/hands-segmentation-pytorch

# Imports

def crop_with_hand_seg(img, hand_seg_model, segmentation_model):
    """
    Crops product image using product and hand segmentation.
    """
    image_b = torch.unsqueeze(img, 0)
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
    data = {"roi": image_roi,
            "mask": pred,
            "image": image
           }
    
    return data

def crop_without_hand_seg(img, segmentation_model):
    """
    Crops product image using product segmentation.
    """
    image_tensor_b = torch.unsqueeze(img, 0).to(DEVICE)
    output = torch.sigmoid(segmentation_model(image_tensor_b.float()))
    pred = output.permute(0, 2, 3, 1).squeeze().detach().cpu().numpy() > 0.5
    
    # Stack preds for getting ROI
    mask = np.stack((pred, pred, pred), axis=-1)
    
    # Image ROI
    image = (image_tensor_b[0].permute(1,2,0).detach().cpu().numpy()+1)/2
    image = (image*255).astype(np.uint8)
    image_roi = image * mask
    
    data = {"roi": image_roi,
            "mask": pred,
            "image": image
           }
    
    return data


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


def crop_with_hand_entropy_seg(img, hand_seg_model, segmentation_model):
    """
    Crops product image using product, hand and entropy based segmentation.
    
    Reference: https://towardsdatascience.com/image-processing-with-python-working-with-entropy-b05e9c84fc36
    """
        
    image_b = torch.unsqueeze(img, 0).to(DEVICE)
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
