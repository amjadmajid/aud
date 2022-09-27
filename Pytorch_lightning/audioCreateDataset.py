import numpy as np
from torch.utils.data import Dataset
from read_audio_data import read_all_audio_in_dir

np.random.seed(123)

#local test path
localPath = "N:\AUD_Data\sampled"
fileToFind = "rec_050cm_000_locH2-FS.wav"

class audioDataset(Dataset):
    """
    Fake dataset generator. Only for demonstration purpose. It create 10000 samples of 64 dimension with 2 classes
    """

    def __init__(self, mode):
        super(audioDataset, self).__init__()

        #TODO: create singular data entry
        self.x, self.y = None, None

        if mode == "singular":
            
            self.x, self.y = read_all_audio_in_dir(localPath, "FS")
            # print(self.x)
            # print(self.x.shape)
            # print(self.y)
        else:

            if mode == "train":
                self.x, self.y = read_all_audio_in_dir(localPath, "FS")
            if mode == "validation":
                self.x, self.y = read_all_audio_in_dir(localPath, "FS")
            if mode == "test":
                self.x, self.y = read_all_audio_in_dir(localPath, "FS")

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
