import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
# Set the seaborn style to 'darkgrid'
sns.set_style("darkgrid")


df = pd.read_csv('./data/data_final.csv')
df_seattle=df[(df['city']=='Seattle')]
q_low = df_seattle["price"].quantile(0.01)
q_hi  = df_seattle["price"].quantile(0.99)
df_seattle=df_seattle.drop_duplicates()

df_seattle = df_seattle[(df["price"] < q_hi) & (df["price"] > q_low)]

df_seattle['date'] = pd.to_datetime(df_seattle['date'],format='%Y-%m-%d')

df_seattle
df_price=df_seattle.groupby(pd.Grouper(key='date',freq='7D')).mean()


fig,ax = plt.subplots(1,1,figsize=(20,6))
ax.plot(df_price['price']);
plt.xticks(rotation='vertical')
ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
plt.title('Price Trend of House sold in Seattle')



df_seattle=df[(df['city']=='Bellevue')]
q_low = df_seattle["price"].quantile(0.01)
q_hi  = df_seattle["price"].quantile(0.99)
df_seattle=df_seattle.drop_duplicates()

df_seattle = df_seattle[(df_seattle["price"] < q_hi) & (df_seattle["price"] > q_low)]

df_seattle['date'] = pd.to_datetime(df_seattle['date'],format='%Y-%m-%d')

df_price=df_seattle.groupby(pd.Grouper(key='date',freq='7D')).mean()

fig,ax = plt.subplots(1,1,figsize=(20,6))
ax.plot(df_price['price']);
plt.xticks(rotation='vertical')
ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
plt.title('Price Trend of House sold in Bellevue')


df = pd.read_csv('./data/data_final.csv')
q_low = df["price"].quantile(0.01)
q_hi  = df["price"].quantile(0.99)
df=df.drop_duplicates()

df = df[(df["price"] < q_hi) & (df["price"] > q_low)]

df['date'] = pd.to_datetime(df['date'],format='%Y-%m-%d')
df.info()

df_price=df.groupby(pd.Grouper(key='date',freq='7D')).mean()

fig,ax = plt.subplots(1,1,figsize=(20,6))
ax.plot(df_price['price']);
plt.xticks(rotation='vertical')
ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
plt.title('Price Trend of House sold in WA stastes')

