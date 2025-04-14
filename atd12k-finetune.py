import numpy as np
import torch
import torch.optim as optim
import os
import random

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
train_root = ''
eval_root = ''
learning_rate = 1e-6
epoches = 50
batch_size = 16

model = Model().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
dataset = ATD12KDataset(train_root, randomCropSize=(224, 224), train=True)
train_data = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
dataset_val = ATD12KDataset(eval_root, train=False)
val_data = DataLoader(dataset_val, batch_size=16, pin_memory=True, num_workers=4)

for epoch in range(1, epoches + 1):
    checkpoint = os.sep.join(("checkpoints", str(epoch) + ".pth"))
    if os.path.exists(checkpoint):
        if os.path.exists(os.sep.join(("checkpoints", str(epoch + 1) + ".pth"))):
            continue
        temp = torch.load(checkpoint, map_location=device)
        model.flownet.load_state_dict(temp["flownet"])
        continue
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

    model.save_model(log_path)
    checkpoints = {
        "flownet": model.flownet.state_dict(),
        "step": step
    }
    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")
    torch.save(checkpoints, os.sep.join(("checkpoints", str(epoch) + ".pth")))
