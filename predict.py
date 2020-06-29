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

# print("skew is:",train.SalePrice.skew())
# plt.hist(train.SalePrice,color='blue')
# plt.show()

target = np.log(train.SalePrice)
# print("\n skew is:",target.skew())
# plt.hist(target,color='blue')
# plt.show()

numeric_features = train.select_dtypes(include=[np.number])
corr = numeric_features.corr()

print(corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print(corr['SalePrice'].sort_values(ascending=False)[-5:])

# plt.scatter(x = train['GarageArea'], y =target)
# plt.ylabel('SalePrice')
# plt.xlabel('Garage Area')
# plt.show()

train = train[train['GarageArea'] < 1200]

# plt.scatter(x = train['GarageArea'], y =np.log(train.SalePrice))
# plt.xlim(-200,1600)
# plt.ylabel('SalePrice')
# plt.xlabel('Garage Area')
# plt.show()

nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns =['Null Count']
nulls.index.name = 'Frature'

# print(nulls)

categoricals = train.select_dtypes(exclude=[np.number])

print(categoricals.describe())

print("Original:\n")
print(train.Street.value_counts())

train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)

print("Encoded:\n")
print(train.enc_street.value_counts())
