import argparse
import warnings
import numpy as np
import math
import random
import torch
import os

warnings.filterwarnings('ignore')

from model.RIFE import Model
from dataset import ATD12KDataset
from torch.utils.data import DataLoader

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")

log_path = 'checkpoints'
epoches = 50
step_per_epoch = 10000
batch_size = 16

parser = argparse.ArgumentParser()
parser.add_argument('--train_root', type=str)
parser.add_argument('--eval_root', type=str)
args = parser.parse_args()

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
        return 3e-4 * mul
    else:
        mul = np.cos((step - 2000) / (epoches * step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-6) * mul + 3e-6

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

model = Model()
dataset = ATD12KDataset(args.train_root, randomCropSize=(224, 224), train=True)
train_data = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
dataset_val = ATD12KDataset(args.eval_root, train=False)
val_data = DataLoader(dataset_val, batch_size=16, pin_memory=True, num_workers=8)

print("Training...")
step = 0
for epoch in range(1, epoches + 1):
    loss = 0.0
    for i, data in enumerate(train_data):
        frames, timestep = data
        frames = frames.to(device)
        timestep = timestep.to(device)
        imgs = frames[:, :6]
        gt = frames[:, 6:9]
        learning_rate = get_learning_rate(step)
        pred, info = model.update(imgs, gt, learning_rate, training=True)
        loss += info['loss_l1']
        step += 1
    print('Epoch:{:<3} loss_l1:{:.4e}'.format(epoch, loss))

    if not os.path.exists(os.path.join(log_path, str(epoch))):
        os.mkdir(os.path.join(log_path, str(epoch)))
    model.save_model(os.path.join(log_path, str(epoch)))
