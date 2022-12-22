import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
# Set the seaborn style to 'darkgrid'
sns.set_style("darkgrid")
df = pd.read_csv('./data/data_final.csv')
price_median=round(df['price'].median(),2)
df['price'].replace(0,price_median)

df.loc[df['waterfront'] > 0, 'waterfront'] = 'Yes'
df.loc[df['waterfront'] == 0, 'waterfront'] = 'No'
df.loc[df['yr_renovated'] > 0, 'yr_renovated'] = 'Yes'
df.loc[df['yr_renovated'] == 0, 'yr_renovated'] = 'No'
df.loc[df['sqft_basement'] > 0, 'sqft_basement'] = 'Yes'
df.loc[df['sqft_basement'] == 0, 'sqft_basement'] = 'No'
df[['state','zip']] = df.statezip.str.split(expand=True)


q_low_price = df["price"].quantile(0.05)
q_hi_price  = df["price"].quantile(0.95)

df= df[(df["price"] < q_hi_price) & (df["price"] > q_low_price)]


q_low_sqft_living = df["sqft_living"].quantile(0.05)
q_hi_sqft_living  = df["sqft_living"].quantile(0.95)

df= df[(df["sqft_living"] < q_hi_sqft_living) & (df["sqft_living"] > q_low_sqft_living)]



q_low_sqft_lot = df["sqft_lot"].quantile(0.05)
q_hi_sqft_lot  = df["sqft_lot"].quantile(0.95)

df= df[(df["sqft_lot"] < q_hi_sqft_lot) & (df["sqft_lot"] > q_low_sqft_lot)]

q_low_yr_built = df["yr_built"].quantile(0.05)
q_hi_yr_built  = df["yr_built"].quantile(0.95)
df= df[(df["yr_built"] < q_hi_yr_built) & (df["yr_built"] > q_low_yr_built)]

df=df.drop(['statezip', 'state','country','street'], axis=1)

sqft_living_bins=df.sqft_living.quantile([0,.33,.66,1])
df['sqft_living']=pd.cut(df.sqft_living,sqft_living_bins,right=True,include_lowest=True)



sqft_lot_bins=df.sqft_lot.quantile([0,.33,.66,1])
df['sqft_lot']=pd.cut(df.sqft_lot,sqft_lot_bins,right=True,include_lowest=True)


yr_built_bins=df.yr_built.quantile([0,.33,.66,1])
df['yr_built']=pd.cut(df.yr_built,yr_built_bins,right=True,include_lowest=True)

sqft_above_bins=df.sqft_above.quantile([0,.33,.66,1])
df['sqft_above']=pd.cut(df.sqft_above,sqft_above_bins,right=True,include_lowest=True)




df.to_csv('./data_clean.csv')