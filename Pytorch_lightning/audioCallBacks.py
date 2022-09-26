from pytorch_lightning.callbacks import Callback


class CustomCallback(Callback):
    def __init__(self):
        super(CustomCallback, self).__init__()

    def on_train_epoch_start(self, trainer, pl_module):
        """Called when the train epoch begins."""
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""
        pass

    def on_validation_epoch_start(self, trainer, pl_module):
        """Called when the val epoch begins."""
        pass

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the val epoch ends."""
        pass

    def on_test_epoch_start(self, trainer, pl_module):
        """Called when the test epoch begins."""
        pass

    def on_test_epoch_end(self, trainer, pl_module):
        """Called when the test epoch ends."""
        pass