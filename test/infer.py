import torchvision.transforms as transforms
import torch
import torch.hub
from segmenter import crop_with_hand_entropy_seg
from utils import get_sharpness, smooth_data_fft, get_blobs, preprocess_image_classification, get_ratio, image_colorfulness, automatic_brightness_and_contrast
from PIL import Image
from tqdm import tqdm
import cv2
from scipy.signal import find_peaks
import math
import timm
import os
import matplotlib.pyplot as plt

# if torch.cuda.is_available():
#     torch.backends.cudnn.deterministic = True

# Device
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
model = timm.create_model("vit_base_patch32_224", num_classes=116)
trained_model = torch.load(
    "models/vit_base_patch32_224.pt", map_location=torch.device("cpu"))
model.load_state_dict(trained_model["model_state_dict"])

transforms_image = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.CenterCrop((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def infer_frame(frame, model):
    img = Image.fromarray(frame).convert('RGB')
    img = transforms_image(img)
    segmented_image = crop_with_hand_entropy_seg(img)["roi"]
    x, y, w, h = get_blobs(segmented_image)
    image_roi = frame[y:y+h, x:x+w]
    img_normalized = preprocess_image_classification(
        frame=image_roi[:, :, [2, 1, 0]])
    img_normalized = img_normalized.unsqueeze_(0)
    img_normalized = img_normalized.to("cpu")
    with torch.no_grad():
        model.eval()
        output = model(img_normalized)
        index = output.data.cpu().numpy().argmax()
        return index+1


video_id = []
object_id = []
timestamp = []

video_location = "./test-videos/"
videos = os.listdir(video_location)
videos.sort()
video_index = 1
print(videos)

for video in videos:
    images = []
    ratios = []
    colors = []
    metrics = []
    frames_with_object = []

    vidcap = cv2.VideoCapture(video_location+video)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Loading Video {video}")

    for i in tqdm(range(frame_count)):
        try:
            success, image = vidcap.read()
            image = image[256:896, 512:1400]
            image = automatic_brightness_and_contrast(image)
            image = cv2.resize(image, dsize=(224, 224),
                               interpolation=cv2.INTER_CUBIC)
            images.append(image)
        except:
            pass

    for img in images:
        color = image_colorfulness(img)
        colors.append(color)

    rolling_colors = smooth_data_fft(colors, 20)
    maximas = find_peaks(rolling_colors)

    print("Calculating Frame of Interest")

    for maxima in tqdm(maximas[0]):
        candidate_frame = maxima
        max_sharpness = -1
        max_metric = -1
        for i in range(-21, 22, 7):
            current_frame = maxima+i
            if(current_frame >= len(images)):
                continue
            sharpness = get_sharpness(images[current_frame])
            colorfulness = image_colorfulness(images[current_frame])
            ratio = get_ratio(images[current_frame])
            metric = colorfulness * colorfulness * ratio
            metric = math.sqrt(metric)
            if(sharpness > max_sharpness and metric > max_metric):
                max_sharpness = sharpness
                max_metric = metric
                candidate_frame = current_frame
        if(max_metric > 110):
            frames_with_object.append(
                (images[candidate_frame], candidate_frame))

    print("Running Inference")

    for image, frame in tqdm(frames_with_object):
        output = infer_frame(image, model)
        video_id.append(video_index)
        object_id.append(output)
        timestamp.append(int(frame/60))
    video_index += 1


print("Creating Submission File")

with open('submission.txt', 'w') as f:
    prev_vid = -1
    prev_obj = -1
    for vid, obj, tim in zip(video_id, object_id, timestamp):
        if(vid == prev_vid and obj == prev_obj):
            continue
        f.write(f'{vid} {obj} {tim}\n')
        prev_vid = vid
        prev_obj = obj
    f.close()
