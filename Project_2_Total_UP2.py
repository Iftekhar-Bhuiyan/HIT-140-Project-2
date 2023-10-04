import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
import seaborn as sns

from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

df = pd.read_csv('/Users/eshayathasan/Documents/HIT140/HIT140_Project 2/po2_data.csv')
df = df[['subject#', 'age', 'sex', 'test_time','jitter(%)','jitter(abs)',	
        'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)', 'shimmer(%)', 
        'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 
        'shimmer(dda)', 'nhr', 'hnr', 'rpde', 'dfa', 'ppe', 'motor_updrs', 'total_updrs']]

# # print(df)

# #Predicting two varibales at a time

print('\n* * * * * * * * * Predicting two varibales at a time * * * * * * * * *\n')
print('\n------------ total UPDRS Prediction -------------\n')

X_total = df.iloc[:, 1:-1].values
y_total = df.iloc[:,-1].values

X_total_train, X_total_test, y_total_train, y_total_test = train_test_split(X_total, y_total, train_size = 0.8, random_state = 0)
model_total = LinearRegression()
model_total.fit(X_total_train, y_total_train)

print('Intercept: ', model_total.intercept_)
print('Coefficient: ', model_total.coef_)

y_total_pred = model_total.predict(X_total_test)
# # print(y_total_pred)

y_total_test = y_total_test.reshape(-1)
y_total_pred = y_total_pred.reshape(-1)

df_total_pred = pd.DataFrame({'Actual': y_total_test, 'Predicted': y_total_pred})
print(df_total_pred)

mae_total = metrics.mean_absolute_error(y_total_test, y_total_pred)
mse_total = metrics.mean_squared_error(y_total_test, y_total_pred)
rmse_total = math.sqrt(mse_total)
y_total_max = y_total_test.max()
y_total_min = y_total_test.min()
rmse_norm_total = rmse_total/(y_total_max-y_total_min)
r_2_total = metrics.r2_score(y_total_test, y_total_pred)
Adj_r_2_total = 1-(1-r_2_total)*(len(y_total)-1)/(len(y_total)-X_total.shape[1]-1)

print('MLR Performance for total UPDRS Prediction\n')
print('MAE for total UPDRS Prediction Model: ', mae_total)
print('MSE for total UPDRS Prediction Model: ', mse_total)
print('RMSE for total UPDRS Prediction Model: ', rmse_total)
print('Normalised RMSE for total UPDRS Prediction Model: ', rmse_norm_total)
print('R-Sqaured for total UPDRS Prediction Model: ', r_2_total)
print('Adjusted R-Sqaured for total UPDRS Prediction Model: ', Adj_r_2_total)

y_base = np.mean(y_total_train)
y_pred_base = [y_base] * len(y_total_test)
print('Y mean: ', y_base)

df_pred_base = pd.DataFrame({'Actual': y_total_test, 'Predicted': y_pred_base})
# print(df_pred_base)

mae_base = metrics.mean_absolute_error(y_total_test , y_pred_base)
mse_base = metrics.mean_squared_error(y_total_test, y_pred_base)
rmse_base = math.sqrt(mse_base)
y_max = y_total_test.max()
y_min = y_total_test.min()
rmse_norm_base = rmse_base/(y_max - y_min)
r_2_base = metrics.r2_score(y_total_test, y_pred_base)

print('MAE Base: ', mae_base)
print('MSE Base; ', mse_base)
print('RMSE Base: ', rmse_base)
print('RMSE Base (Normalised): ', rmse_norm_base)
print('R-squared Base: ', r_2_base)

fig, axs = plt.subplots(4,3)
fig.tight_layout()

axs[0,0].scatter(x = df['age'], y = df['total_updrs'])
axs[0,0].set_xlabel('Age')
axs[0,0].set_ylabel('total_UPDRS')

axs[0,1].scatter(x = df['sex'], y = df['total_updrs'])
axs[0,1].set_xlabel('Sex')
axs[0,1].set_ylabel('total_UPDRS')

axs[0,2].scatter(x = df['test_time'], y = df['total_updrs'])
axs[0,2].set_xlabel('Test Time')
axs[0,2].set_ylabel('total_UPDRS')

axs[1,0].scatter(x = df['jitter(%)'], y = df['total_updrs'])
axs[1,0].set_xlabel('Jitter %')
axs[1,0].set_ylabel('total_UPDRS')

axs[1,1].scatter(x = df['jitter(abs)'], y = df['total_updrs'])
axs[1,1].set_xlabel('Absolute Jitter')
axs[1,1].set_ylabel('total_UPDRS')

axs[1,2].scatter(x = df['jitter(rap)'], y = df['total_updrs'])
axs[1,2].set_xlabel('Jitter - RAP')
axs[1,2].set_ylabel('total_UPDRS')

axs[2,0].scatter(x = df['jitter(ppq5)'], y = df['total_updrs'])
axs[2,0].set_xlabel('Jitter - PPQ5')
axs[2,0].set_ylabel('total_UPDRS')

axs[2,1].scatter(x = df['jitter(ddp)'], y = df['total_updrs'])
axs[2,1].set_xlabel('Jitter - DDP')
axs[2,1].set_ylabel('total_UPDRS')

axs[2,2].scatter(x = df['shimmer(%)'], y = df['total_updrs'])
axs[2,2].set_xlabel('Shimmer %')
axs[2,2].set_ylabel('total_UPDRS')

axs[3,0].scatter(x = df['shimmer(abs)'], y = df['total_updrs'])
axs[3,0].set_xlabel('Absolute Shimmer')
axs[3,0].set_ylabel('total_UPDRS')

axs[3,1].scatter(x = df['shimmer(apq3)'], y = df['total_updrs'])
axs[3,1].set_xlabel('Shimmer - APQ3')
axs[3,1].set_ylabel('total_UPDRS')

axs[3,2].scatter(x = df['shimmer(apq5)'], y = df['total_updrs'])
axs[3,2].set_xlabel('Shimmer - APQ5')
axs[3,2].set_ylabel('total_UPDRS')


fig_2, axs_2 = plt.subplots(3,3)
fig_2.tight_layout()

axs_2[0,0].scatter(x = df['shimmer(apq11)'], y = df['total_updrs'])
axs_2[0,0].set_xlabel('Shimmer - APQ11')
axs_2[0,0].set_ylabel('total_UPDRS')

axs_2[0,1].scatter(x = df['shimmer(dda)'], y = df['total_updrs'])
axs_2[0,1].set_xlabel('Shimmer DDA')
axs_2[0,1].set_ylabel('total_UPDRS')

axs_2[0,2].scatter(x = df['nhr'], y = df['total_updrs'])
axs_2[0,2].set_xlabel('Noise to Harmonic Ratio')
axs_2[0,2].set_ylabel('total_UPDRS')

axs_2[1,0].scatter(x = df['hnr'], y = df['total_updrs'])
axs_2[1,0].set_xlabel('Harmonic to Noise Ratio')
axs_2[1,0].set_ylabel('total_UPDRS')

axs_2[1,1].scatter(x = df['rpde'], y = df['total_updrs'])
axs_2[1,1].set_xlabel('Recurrance Period Density')
axs_2[1,1].set_ylabel('total_UPDRS')

axs_2[1,2].scatter(x = df['dfa'], y = df['total_updrs'])
axs_2[1,2].set_xlabel('Detrended Fluctuation Analysis')
axs_2[1,2].set_ylabel('total_UPDRS')

axs_2[2,0].scatter(x = df['ppe'], y = df['total_updrs'])
axs_2[2,0].set_xlabel('Pitch Period Entropy')
axs_2[2,0].set_ylabel('total_UPDRS')

axs_2[2,1].scatter(x = df['motor_updrs'], y = df['total_updrs'])
axs_2[2,1].set_xlabel('Motor UPDRS')
axs_2[2,1].set_ylabel('total_UPDRS')

df_total = df.drop(['subject#'], axis=1)
corr = df_total.corr()

fig_3, ax = plt.subplots()
fig_3.tight_layout()

ax = sns.heatmap(corr, vmax = 1, vmin = -1, center = 0, 
                  cmap = sns.diverging_palette(20, 220, n = 200), 
                  square = False, annot = True)

ax.set_xticklabels(ax.get_xticklabels(), rotation =45, horizontalalignment = 'right')

#Basic Model for total UPDRS Prediction without total UPDRS

print('\n----------Basic Model for total UPDRS Prediction with motor UPDRS----------\n')

X_total_ols = df_total.iloc[: , :-1].values
y_total_ols = df_total.iloc[: , -1].values

X_total_ols = pd.DataFrame(X_total_ols, columns=df_total.columns[:-1])

X_total_ols = sm.add_constant(X_total_ols)
model_total_ols = sm.OLS(y_total_ols,X_total_ols).fit()
pred_total_ols = model_total_ols.predict(X_total_ols)
model_details = model_total_ols.summary()
print(model_details)

plt.show()

#Log Transformation

print('\n-----------Log Transformation for total UPDRS Prediction with motor UPDRS----------\n')

df_total_LT = df_total.copy()

columns_to_log = ['age', 'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
                  'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 
                  'shimmer(dda)', 'nhr','hnr']

df_total_LT[['LAge', 'LJitter(%)', 'LAbs Jitter', 'LJitter RAP', 'LJitter PPQ5', 'LJitter DDP',
             'LShimmer(%)', 'LAbs Shimmer', 'LShimmer APQ3', 'LShimmer APQ5', 'LShimmer APQ11', 
             'LShimmer DDA', 'LNHR', 'LHNR']] = df_total_LT[columns_to_log].apply(lambda x: np.log(x))

df_total_LT = df_total_LT.drop(['age', 'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 
                    'jitter(ddp)', 'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 
                    'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)', 'nhr', 'hnr'],
                    axis = 1)

df_total_LT = df_total_LT[['LAge', 'sex', 'test_time','LJitter(%)','LAbs Jitter',	
        'LJitter RAP', 'LJitter PPQ5', 'LJitter DDP', 'LShimmer(%)', 
        'LAbs Shimmer', 'LShimmer APQ3', 'LShimmer APQ5', 'LShimmer APQ11', 
        'LShimmer DDA', 'LNHR', 'LHNR', 'rpde', 'dfa', 'ppe', 'motor_updrs', 'total_updrs']]

X_total_ols = df_total_LT.iloc[: , :-1].values
y_total_ols = df_total_LT.iloc[: , -1].values

X_total_ols = pd.DataFrame(X_total_ols, columns=df_total_LT.columns[:-1])

X_total_ols = sm.add_constant(X_total_ols)
model_total_ols = sm.OLS(y_total_ols,X_total_ols).fit()
pred_total_ols = model_total_ols.predict(X_total_ols)
model_details = model_total_ols.summary()
print(model_details)

corr = df_total_LT.corr()

fig_4, ax = plt.subplots()
fig_4.tight_layout()

ax = sns.heatmap(corr, vmax = 1, vmin = -1, center = 0, 
                  cmap = sns.diverging_palette(20, 220, n = 200), 
                  square = False, annot = True)

ax.set_xticklabels(ax.get_xticklabels(), rotation =45, horizontalalignment = 'right')

# plt.show()

#Colinearity Analysis

print('\n-----------Colinearity Analysis for total UPDRS Prediction with motor UPDRS----------\n')

df_total_CL = df_total.copy()

df_total_CL = df_total_CL.drop(['jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)', 'shimmer(%)' , 
                                'shimmer(abs)', 'shimmer(apq3)', 'shimmer(dda)', 'nhr', ],
                    axis = 1)

X_total_ols = df_total_CL.iloc[: , :-1].values
y_total_ols = df_total_CL.iloc[: , -1].values

X_total_ols = pd.DataFrame(X_total_ols, columns=df_total_CL.columns[:-1])

X_total_ols = sm.add_constant(X_total_ols)
model_total_ols = sm.OLS(y_total_ols,X_total_ols).fit()
pred_total_ols = model_total_ols.predict(X_total_ols)
model_details = model_total_ols.summary()
print(model_details)

# print(df_total)
# print(df_total_CL)

#Standardization

print('\n-----------Z-Standardization for total UPDRS Prediction with motor UPDRS----------\n')

X_total_ols = df_total.iloc[: , :-1].values
y_total_ols = df_total.iloc[: , -1].values

X_total_ols = pd.DataFrame(X_total_ols, columns=df_total.columns[:-1])

X_total_ols = sm.add_constant(X_total_ols)

scaler = StandardScaler()

X_total_ols = X_total_ols.drop(['const'], axis = 1)

std_X_total = scaler.fit_transform(X_total_ols)

std_X_total_df = pd.DataFrame(std_X_total, index=X_total_ols.index, columns=X_total_ols.columns)

std_X_total_df = sm.add_constant(std_X_total_df)

# print(std_X_total_df)

model_total_ols = sm.OLS(y_total_ols,std_X_total_df).fit()
pred_total_ols = model_total_ols.predict(std_X_total_df)
model_details = model_total_ols.summary()
print(model_details)

#Gaussian Transformation

print('\n-----------Gaussian Transformation for total UPDRS Prediction with motor UPDRS----------\n')

X_total_ols = df_total.iloc[: , :-1].values
y_total_ols = df_total.iloc[: , -1].values

X_total_ols = pd.DataFrame(X_total_ols, columns=df_total.columns[:-1])

X_total_ols = sm.add_constant(X_total_ols)

scaler = PowerTransformer()

X_total_ols = X_total_ols.drop(['const'], axis = 1)

GT_X_total = scaler.fit_transform(X_total_ols)

GT_X_total_df = pd.DataFrame(GT_X_total, index=X_total_ols.index, columns=X_total_ols.columns)

GT_X_total_df = sm.add_constant(GT_X_total_df)

# print(std_X_total_df)

model_total_ols = sm.OLS(y_total_ols,GT_X_total_df).fit()
pred_total_ols = model_total_ols.predict(GT_X_total_df)
model_details = model_total_ols.summary()
print(model_details)