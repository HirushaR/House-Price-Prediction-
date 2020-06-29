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

condition_pivot = train.pivot_table(index='SaleCondition',values='SalePrice',aggfunc=np.median)
condition_pivot.plot(kind='bar',color='blue')
plt.xlabel('SaleCondition')
plt.ylabel('Median SalePrice')
plt.xticks(rotation=0)
plt.show()

def encode(x): return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)

condition_pivot = train.pivot_table(index='enc_condition',values='SalePrice',aggfunc=np.median)
condition_pivot.plot(kind='bar',color='blue')
plt.xlabel('Encoded SaleCondition')
plt.ylabel('Median SalePrice')
plt.xticks(rotation=0)
plt.show()

data = train.select_dtypes(include = [np.number]).interpolate().dropna()
print(sum(data.isnull().sum() != 0))

y = np.log(train.SalePrice)
x = data.drop(['SalePrice','Id'], axis=1)

X_train , X_test ,Y_train,Y_test = train_test_split(x,y,random_state=42,test_size=.33)

lr = linear_model.LinearRegression()
model = lr.fit(X_train,Y_train)

print(" R^2 is:\n",model.score(X_test,Y_test))

predictions = model.predict(X_test)
print("RMSE is :\n", mean_squared_error(Y_test,predictions))

actual_value =Y_test
plt.scatter(predictions,actual_value,alpha=0.75,color='b')
plt.xlabel('Prediction Price')
plt.ylabel('actual Price')
plt.title('LinearRegression Model')
plt.show()

for i in range(-2,3):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, Y_train)
    preds_ridge = ridge_model.predict(X_test)

    plt.scatter(preds_ridge,actual_value,alpha=.75,color='b')
    plt.xlabel('Prediction Price')
    plt.ylabel('actual Price')
    plt.title('Ridge Regression with alpha = {}'.format(alpha))
    overlay = 'R*2 is : {}\n RMSE is: {}'.format(
        ridge_model.score(X_test,Y_test),
        mean_squared_error(Y_test,preds_ridge))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show()

submission = pd.DataFrame()
submission['Id'] = test.Id

feats = test.select_dtypes(include = [np.number]).drop(['Id'],axis=1).interpolate()

predictions = model.predict(feats)

final_predictions =np.exp(predictions)

print("Original prediction: \n", predictions[:10],"\n")
print("Final prediction: \n", final_predictions[:10],"\n")

submission['SalePrice'] = final_predictions
print(submission.head())

submission.to_csv('data/submission.csv',index=False)

