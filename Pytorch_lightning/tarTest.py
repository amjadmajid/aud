#file to check if a created tar file store a proper class

import webdataset as wds

path = "file:N://AUD_Data/sampled/tars/"

dataset = wds.WebDataset(path + "train.tar")

sample = next(iter(dataset))

for k,v in sample.items():
    print("%20s = %s"%(k, repr(v)[:60]))