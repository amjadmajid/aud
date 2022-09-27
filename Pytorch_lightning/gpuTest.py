#small script to allow manual check for GPU availability

import torch

device = torch.device("cuda")
torch.rand(10).to(device)
print(torch.cuda.is_available())