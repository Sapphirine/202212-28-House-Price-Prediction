import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel



df = pd.read_csv('./data_clean.csv')
y=df['price']
df_features=df.loc[:,'bedrooms':'yr_renovated']
df_features_ohe=pd.get_dummies(df_features)


X_train, X_test, y_train, y_test = train_test_split(df_features_ohe,y,train_size=.8,test_size=.2,random_state=512)
gbr=GradientBoostingRegressor(n_estimators=10,max_depth=5)

gbr.fit(X_train,y_train)
gbr_feature_importances=pd.DataFrame(gbr.feature_importances_,index=X_train.columns,columns=['importances'])
gbr_feature_importances.sort_values(by=['importances'],ascending=False)

gbr_feature_importances.plot.bar(stacked=True)




sfm = SelectFromModel(gbr,threshold='mean',prefit=True)
X_train.columns[sfm.get_support()]

X_train_fs=sfm.transform(X_train)
X_test_fs=sfm.transform(X_test)
gbr_fs=GradientBoostingRegressor(n_estimators=10,max_depth=5)
gbr_fs.fit(X_train_fs,y_train)
print(f'training accuracy: {gbr_fs.score(X_train_fs,y_train).round(2)}')
print(f'test accuracy    : {gbr_fs.score(X_test_fs,y_test).round(2)}')



X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(df_features_ohe,y,train_size=.8,test_size=.2,random_state=512)
lr = LinearRegression().fit(X_train_r,y_train_r)
lr_training_r2 = lr.score(X_train_r,y_train_r)
print(f'lr training set R^2: {lr_training_r2.round(2)}')
print(f'lr training set coe: {lr.coef_.round(2)}')


