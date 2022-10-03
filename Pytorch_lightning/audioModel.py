from pyexpat import model
import pytorch_lightning as pl
from torch.nn.modules import Sequential, Linear, ReLU, Conv2d, MSELoss, LeakyReLU
from torch.optim import SGD
import torch
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.CRITICAL)


class AudioModel(pl.LightningModule):
    def __init__(self):
        super().__init__(),

        self.conv1 = Sequential(
            Conv2d(1,64, (1,50)),
            LeakyReLU()
        )

        self.conv2 = Sequential(
            Conv2d(64, 64, (6, 20)),
            LeakyReLU()
        )

        self.conv3 = Sequential(
            Conv2d(64,64, (3,10)),
            LeakyReLU()
        )

        self.ll = Sequential(
            Linear(4*4333*64, 64),
            LeakyReLU(),
            Linear(64, 32),
            LeakyReLU(),
            Linear(32, 2),
        )
        self.criterion = MSELoss()

    def forward(self, x):
        x1 = self.conv1(x[0])
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x3 = x3.view(1, -1)
        flatten = self.ll(x3)
        return flatten

    def training_step(self, batch, batch_index):
        x, y = batch
        output = self(x)
        return {"out": output, "label": y}

    def training_step_end(self, outputs):
        out, y = outputs["out"], outputs["label"]
        loss = self.criterion(out, y)
        return loss

    def training_epoch_end(self, outputs) -> None:
        all_epoch_losses = [out["loss"] for out in outputs]
        final_loss = torch.stack(all_epoch_losses).mean()
        logging.warning(f"Loss: {final_loss}")

    def validation_step(self, batch, batch_index):
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)
        self.log("val_loss", loss)
        # print(y)
        # print(output)
        return {"out": output, "label": y}

    def validation_step_end(self, outputs):
        return 0

    def validation_epoch_end(self, outputs) -> None:
        pass
    def test_step(self, a, b):
        pass

    def test_step_end(self, a,b):
        pass

    def test_epoch_end(self, outputs) -> None:
        pass
    
    def configure_optimizers(self):
        return SGD(self.parameters(), lr=1e-1)
        