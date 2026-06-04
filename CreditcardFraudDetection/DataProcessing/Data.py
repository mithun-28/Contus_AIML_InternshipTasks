import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("fraudTest.csv")
print( df.shape)
drop_cols = ['Unnamed: 0','first','last','street','trans_num','cc_num']
df.drop(columns=drop_cols, inplace=True)


df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['dob'] = pd.to_datetime(df['dob'])


df['transaction_hour'] = (df['trans_date_trans_time'].dt.hour)
df['transaction_day'] = (df['trans_date_trans_time'].dt.day)
df['transaction_month'] = (df['trans_date_trans_time'].dt.month)
df['transaction_weekday'] = (df['trans_date_trans_time'].dt.weekday)
df['age'] = (df['trans_date_trans_time'].dt.year -df['dob'].dt.year)


df.drop(columns=['trans_date_trans_time','dob'],inplace=True)


df['distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 +(df['long'] - df['merch_long'])**2)
df['amt_per_pop'] = (df['amt'] /(df['city_pop'] + 1))

for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])


cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

df.to_csv("processed_fraud_data.csv",index=False)
print("\nPreprocessing Completed")
print("Saved as processed_fraud_data.csv")
print("Final Shape:", df.shape)