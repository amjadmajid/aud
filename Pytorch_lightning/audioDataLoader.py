import pytorch_lightning as pl
from torch.utils.data import DataLoader
from audioCreateDataset import audioDataset
import webdataset as wds


class AudioDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def prepare_data(self):
        # download, tokenize or operations that use disk and it's done on a single gpu in a distributed scenario
        # DO NOT ASSIGN self.variable in this methods!!!!!!!!!!
        # data = audioDataset(mode="singular")
        # wds.WebDataset("file:N://AUD_Data//sampled//tars/validation.tar").decode(wds.torch_audio).to_tuple("pt", "txt")
        self.train_dataset = audioDataset("train")
        self.val_dataset = audioDataset("validation")
        self.test_dataset = audioDataset("test")
        

    def setup(self, stage=None):
        pass

    def train_dataloader(self) -> DataLoader:
        return wds.WebLoader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return wds.WebLoader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return wds.WebLoader(self.test_dataset)