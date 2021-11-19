
import os
import torch
import torch.optim as optim
from pathlib import Path
from torch import nn
from torchvision import transforms, utils, datasets

from unet import UNet

data_folder = "data"
model_folder = Path("model")
model_folder.mkdir(exist_ok=True)
model_path = "model/unet-voc.pt"
saving_interval = 10
epoch_number = 100
shuffle_data_loader = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
