import numpy as np 
import matplotlib.pyplot as plt 
import pandas
import glob
import os
import ntpath


version = 0
#find path to latest version that has been logged into "./lightning_logs"
for f in os.listdir('lightning_logs'):
    print(f)
    new_version = int(f[f.index('_')+1:])
    print(new_version)
    if new_version > version:
        version = new_version
#load up data collected data
data = pandas.read_csv("lightning_logs/version_"+ str(version) + "/metrics.csv")

data.plot(x="epoch", y="val_loss")

plt.show()