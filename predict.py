import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('data/train.csv')


######################### clean data #########################

#print(df.info())
# check how many column with null values
#print(sum(df.isna().any()))

#print(df['MasVnrType'].value_counts())

#select all the columns with object data type
list_obj_columns = list(df.select_dtypes(include='object').columns)

#select all the columns data type isnt object
list_num_columns = list(df.select_dtypes(exclude='object').columns)


# create function to fill na values
def fillna_all(df):
    for col in list_obj_columns:
        df[col].fillna(value=df[col].mode()[0],inplace=True)
    for col in list_num_columns:
        df[col].fillna(value=df[col].mean(), inplace=True)

fillna_all(df)

#################### features encoding ############################

# for col in list_obj_columns:
#     print(col, ':', df[col].nunique())

temp = df['Id']
dummy = pd.get_dummies(df[list_obj_columns], prefix=list_obj_columns)
df.drop(list_obj_columns, axis=1, inplace=True)
#

df_finl = pd.concat([df,dummy],axis=1)
# print(dummy.shape)
# print(df_finl.shape)

#print(df_finl.info())

df_test = pd.read_csv('data/test.csv')

list_num_columns.remove('SalePrice')
fillna_all(df_test)

dummy1 = pd.get_dummies(df_test[list_obj_columns], prefix=list_obj_columns)

df_test = pd.read_csv('data/test.csv')
df_train_test = pd.concat([df.drop('SalePrice',axis=1), df_test], axis=0)

fillna_all(df_train_test)

dummy3 = pd.get_dummies(df_train_test[list_obj_columns], prefix=list_obj_columns)

#print(dummy.shape, dummy1.shape, dummy3.shape)
df_train_test.drop(list_obj_columns, axis=1 ,inplace=True)
#print(df_train_test.shape)

df_train_test_final = pd.concat([df_train_test,dummy3], axis=0)
#print(df_train_test_final.shape)

x_train = df_train_test_final.iloc[0:1460]
x_test = df_train_test_final.iloc[1460:]

y = df['SalePrice']

