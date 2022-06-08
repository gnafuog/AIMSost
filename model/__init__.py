from torchvision.models import MobileNetV2
import torchsummary
import torch

torchsummary.summary(MobileNetV2().to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')),(3,3,3))