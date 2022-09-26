#small script to allow manual check for GPU availability

import torch

print(torch.cuda.is_available())