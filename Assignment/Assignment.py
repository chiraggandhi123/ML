import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
X=pd.read_csv('./DATASET/Linear_X_Train.csv')
Y=pd.read_csv('./DATASET/Linear_Y_Train.csv')
print(X)
print(Y)
plt.scatter(X)
plt.show()
