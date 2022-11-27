import matplotlib.pyplot as plt
import pandas as pd

path = r"C:\Users\Nils\Documents\GitHub\aud\Pytorch_lightning\lightning_logs\version_33\metricsFSoffgrid.csv"

table = pd.read_csv(path)


fig = plt.figure(figsize =(10, 7))

table.boxplot('test_loss')

plt.show()