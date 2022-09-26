import pytorch_lightning as pl
from torch.utils.data import DataLoader
from fake_dataset import FakeDataset
from audioCreateDataset import audioDataset


class AudioDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def prepare_data(self):
        # download, tokenize or operations that use disk and it's done on a single gpu in a distributed scenario
        # DO NOT ASSIGN self.variable in this methods!!!!!!!!!!
        self.train_dataset = audioDataset(mode="singular")
        self.val_dataset = audioDataset(mode="singular")
        self.test_dataset = audioDataset(mode="singular")
        self.batch_size = 1
        

    def setup(self, stage=None):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=1)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=1)