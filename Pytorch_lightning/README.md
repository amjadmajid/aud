To install:

pip install -r dependencies.txt

Basic instructions:

Collect data to train over, audio file is split by audioSplit.py into the desired amount of files (now set at 200)

After running audioSplit.py, we can run audioTraining.py to create a new model.
audioTraining.py can be set to use either a cpu or gpu setup. The Trainer.pl class has a parameter accelerator, which can be set to "cpu" or "gpu", depending on the system resources. To use a gpu, pytorch needs to be configured with a version of CUDA, which requires additional installation steps.

The model will be stored into lightning_logs, where it will be created in a new folder named "version_x", where x will be the smallest available version number. Within a version folder, multiple things are stored:

hyperparameters.yaml
metrics.csv : log of the val_loss achieved per epoch
modello-figo-{epoch}--{val_loss:0.4f} : model, named after its validation loss. It keeps the best k models, based on the k specified in Trainer.pl
