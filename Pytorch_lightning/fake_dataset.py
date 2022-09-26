import numpy as np
from torch.utils.data import Dataset

np.random.seed(123)


class FakeDataset(Dataset):
    """
    Fake dataset generator. Only for demonstration purpose. It create 10000 samples of 64 dimension with 2 classes
    """

    def __init__(self, mode):
        super(FakeDataset, self).__init__()
        all_data = np.random.randint(0, 1500, (12000, 64)) / 1500.0
        mean_value = np.mean(all_data, axis=1)
        labels = [0 if a < 0.5 else 1 for a in mean_value]
        self.x, self.y = None, None
        if mode == "train":
            self.x = all_data[:6000]
            self.y = labels[:6000]

        if mode == "validation":
            self.x = all_data[6000:10000]
            self.y = labels[6000:10000]

        if mode == "test":
            self.x = all_data[10000:]
            self.y = labels[10000:]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


