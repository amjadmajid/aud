import numpy as np 
import matplotlib.pyplot as plt 
import pandas

#load up data collected data
data = pandas.read_csv("lightning_logs/version_3/metrics.csv")

data.plot(x="epoch", y="val_loss")

plt.show()