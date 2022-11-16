#file to check if a created tar file stores a proper class

import webdataset as wds
import torch

path = r"N:\\AUD_Data\\sampled\\tars\\train\\rec_029cm_155_locH2-FSO-0-Session1.pt"

# dataset = wds.WebDataset(path + "train.tar")

# sample = next(iter(dataset))

# for k,v in sample.items():
#     print("%20s = %s"%(k, repr(v)[:60]))

x =torch.load(path)
print(x.shape)