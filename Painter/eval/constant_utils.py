import numpy as np
import torch
from torchvision import transforms

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


imagenet_mean_ts=torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
imagenet_std_ts=torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

images_normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


criterion = torch.nn.KLDivLoss(reduction='batchmean')
