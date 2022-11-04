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


Creating a new webdataset:

First run tarCreation.py to instantiate a cls file to store the normalized class labels, which should have the same name as the audio file

Then, create a .tar using the following command on Windows:

tar -cvzf archive name.tar path to folder to compress

After this, a tar file should be found in the directory the command was run. This tar can be used as a webdataset. 

In order to be able to read from a webdataset, we need to provide a url, which, if we have data locally, is just a path name. If it is not a url, we need to write it as "file:{absolutepath}". If the data is stored remotely, we can simply use the remote url.
