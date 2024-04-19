'''
Evaluate Datasets
'''
Name = "ATD-12K"
path = '/home/michael/hilbert/Desktop/Datasets/atd-12k/test_2k_540p/'

exp = 4
# inference ratio between two images with 0 - 1 range
ratio = 0.5
# rReturns image when actual ratio falls in given range threshold
rthreshold = 0.02
# Limit max number of bisectional cycles
rmaxcycles = 8
# Directory with trained model files
modelDir = 'train_log'

import os
import cv2
import torch
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

try:
    try:
        try:
            from model.RIFE_HDv2 import Model
            model = Model()
            model.load_model(modelDir, -1)
            print("Loaded v2.x HD model.")
        except:
            from train_log.RIFE_HDv3 import Model
            model = Model()
            model.load_model(modelDir, -1)
            print("Loaded v3.x HD model.")
    except:
        from model.RIFE_HD import Model
        model = Model()
        model.load_model(modelDir, -1)
        print("Loaded v1.x HD model")
except:
    from model.RIFE import Model
    model = Model()
    model.load_model(modelDir, -1)
    print("Loaded ArXiv-RIFE model")

model.eval()
model.device()

l = len(os.listdir(path))
index = 1
print()
for triplet in os.listdir(path):
    img = [
        path + triplet + "/frame1.png", 
        path + triplet + "/frame3.png", 
          ]
    img0 = cv2.imread(img[0], cv2.IMREAD_UNCHANGED)
    img1 = cv2.imread(img[1], cv2.IMREAD_UNCHANGED)
    img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)

    if ratio:
        img_list = [img0]
        img0_ratio = 0.0
        img1_ratio = 1.0
        if ratio <= img0_ratio + rthreshold / 2:
            middle = img0
        elif ratio >= img1_ratio - rthreshold / 2:
            middle = img1
        else:
            tmp_img0 = img0
            tmp_img1 = img1
            for inference_cycle in range(rmaxcycles):
                middle = model.inference(tmp_img0, tmp_img1)
                middle_ratio = ( img0_ratio + img1_ratio ) / 2
                if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
                    break
                if ratio > middle_ratio:
                    tmp_img0 = middle
                    img0_ratio = middle_ratio
                else:
                    tmp_img1 = middle
                    img1_ratio = middle_ratio
        img_list.append(middle)
        img_list.append(img1)
    else:
        img_list = [img0, img1]
        for i in range(exp):
            tmp = []
            for j in range(len(img_list) - 1):
                mid = model.inference(img_list[j], img_list[j + 1])
                tmp.append(img_list[j])
                tmp.append(mid)
            tmp.append(img1)
            img_list = tmp

    if not os.path.exists('output/' + Name + "/" + triplet):
        os.makedirs('output/' + Name + "/" + triplet)

    for i in range(1, len(img_list) + 1):
        cv2.imwrite('output/' + Name + '/' + triplet + '/' + 'frame{}.png'.format(i), (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

    print(f"\rProcessing {triplet} {index} / {l}", end="")
    index += 1