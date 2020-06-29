import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print(train.shape)
print(test.shape)

plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10,6)

print(train.SalePrice.describe())

print("skew is:",train.SalePrice.skew())
plt.hist(train.SalePrice,color='blue')
plt.show()


