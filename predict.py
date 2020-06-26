import pandas as pd

df = pd.read_csv('data/train.csv')

#print(df.info())
# check how many column with null values
#print(sum(df.isna().any()))

#print(df['MasVnrType'].value_counts())

#select all the columns with object data type
list_obj_columns = list(df.select_dtypes(include='object').columns)

#select all the columns data type isnt object
list_num_columns = list(df.select_dtypes(exclude='object').columns)
print(len(list_num_columns))
