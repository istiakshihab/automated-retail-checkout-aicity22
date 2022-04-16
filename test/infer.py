import imp
import torch
import torch.hub
import torch.nn as nn
from .segmentation import *
from .unet import build_unet
from .utils import get_blobs, preprocess_image_classification

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the model
hand_seg_model = torch.hub.load(
    repo_or_dir='guglielmocamporese/hands-segmentation-pytorch', 
    model='hand_segmentor', 
    pretrained=True
)
hand_seg_model.eval()

model_path = '../models/unet_aicityt4.pth'

# Define model
segmentation_model = build_unet()
checkpoint = torch.load(model_path, map_location="cpu")
segmentation_model.load_state_dict(checkpoint)
# Send to GPU
segmentation_model = segmentation_model.to(DEVICE)
segmentation_model.eval()

def infer_frame(frame, model, hand_seg_model, segmentation_model):
   img = Image.fromarray(frame).convert('RGB')
   img = transforms_image(img)
   segmented_image = crop_with_hand_entropy_seg(img, hand_seg_model, segmentation_model)["roi"]
   x,y,w,h = get_blobs(segmented_image)
   image_roi = frame[y:y+h, x:x+w]
   img_normalized = preprocess_image_classification(frame=image_roi[:,:,[2,1,0]])
   img_normalized = img_normalized.unsqueeze_(0)
   img_normalized = img_normalized.to("cpu")
   with torch.no_grad():
      model.eval()
      output =model(img_normalized)
      index = output.data.cpu().numpy().argmax()
      op_array = output.data.cpu().numpy()
      print(op_array[0][index])
      return index+1



video_location = "../test-videos/"
videos = ["testA_1.mp4"]
images_list = []
ratios_list = []
colors_list = []
metrics_list = []

for video in videos:
  vidcap = cv2.VideoCapture(video_location+video)
  fps = vidcap.get(cv2.CAP_PROP_FPS)      
  frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
  resized_image = []

  for i in tqdm(range(frame_count)):
    try:
      success,image = vidcap.read()
      image = image[256:896, 512:1400]
      image = automatic_brightness_and_contrast(image)
      image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
      resized_image.append(image)
    except:
      pass
  ratios = []
  colors = []
  metrics = []
  for img in resized_image:
    # ratio = get_ratio(img)
    color = image_colorfulness(img)
    # metric = color*color*ratio
    colors.append(color)
    # ratios.append(ratio)
    # metrics.append(metric)
  
  images_list.append(resized_image)
  # ratios_list.append(ratios)
  colors_list.append(colors)
  # metrics_list.append(metrics)
  # print(video)
  # plt.plot(metrics)
  # plt.show()
  # plt.plot(ratios)
  # plt.show()
  plt.plot(colors)
  plt.show()
