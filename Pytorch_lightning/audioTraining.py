#Handles the training and saving of the DNN to predict x and y values


import pytorch_lightning as pl
from audioDataLoader import AudioDataModule
from audioModel import AudioModel
from pytorch_lightning.callbacks import ModelCheckpoint
from audioCallBacks import CustomCallback
from pytorch_lightning.loggers import CSVLogger

def train():
    data_m = AudioDataModule()
    model = AudioModel()
    custom_call = CustomCallback()

    checkpoint_p = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        verbose=True,
        filename="modello-figo-{epoch}--{val_loss:0.4f}",
        save_top_k=3,
        
    )

    trainer = pl.Trainer(
        # gpus = -1,
        accelerator="gpu",
        callbacks=[checkpoint_p, custom_call],
        min_epochs=1,
        max_epochs=1000,
        logger=CSVLogger(save_dir="")
    )

    trainer.fit(model, data_m)


if __name__ == "__main__":
    train()
    