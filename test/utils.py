
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import timm
import scipy.fftpack as fftpack

def get_sharpness(image):
    im = Image.fromarray(image).convert('L')
    array = np.asarray(im, dtype=np.int32)

    gy, gx = np.gradient(array)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness = np.average(gnorm)
    return sharpness

def smooth_data_fft(arr, span): 
    w = fftpack.rfft(arr)
    spectrum = w ** 2
    cutoff_idx = spectrum < (spectrum.max() * (1 - np.exp(-span / 2000)))
    w[cutoff_idx] = 0
    return fftpack.irfft(w)
    
def get_ratio(image):
    h, w, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]

    pixels = cv2.countNonZero(thresh)
    ratio = (pixels/(h * w)) * 100
    return ratio

def image_colorfulness(image):
	(B, G, R) = cv2.split(image.astype("float"))
	rg = np.absolute(R - G)
	yb = np.absolute(0.5 * (R + G) - B)
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))
	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
	return stdRoot + (0.3 * meanRoot)

def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result

def get_blobs(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tv, thresh = cv2.threshold(gray[:,:,0], 1, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    c = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    return x,y,w,h

transforms_image = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.CenterCrop((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def preprocess_image_classification(frame):
   img = Image.fromarray(frame).convert('RGB')
   mean = [0.4124, 0.3856, 0.3493] 
   std = [0.2798, 0.2703, 0.2726]
   test_transforms = timm.data.create_transform(
      input_size=224, mean=mean, std=std
   )
   img_normalized = test_transforms(img).float()
   return img_normalized