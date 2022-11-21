#file to check if a created tar file stores a proper class

import webdataset as wds
import torch
import numpy as np

path = r"file:N:\\AUD_Data\\sampled\\tars\\"

dataset = wds.WebDataset(path + "test.tar")

sample = next(iter(dataset))

for k,v in sample.items():
    print("%20s = %s"%(k, repr(v)[:60]))
    print(sample['pt'][:60])
    print("ended sample")
    x  = torch.tensor(np.frombuffer(sample['pt'], dtype=np.float32))
    print(type(x))

# x =torch.load(path)
# print(x.shape)