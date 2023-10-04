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
print('\n------------ Motor UPDRS Prediction -------------\n')

X_motor = df.iloc[:, 1:-2].values
y_motor = df.iloc[:, -2:-1].values

X_motor_train, X_motor_test, y_motor_train, y_motor_test = train_test_split(X_motor, y_motor, train_size = 0.6, random_state = 0)
model_motor = LinearRegression()
model_motor.fit(X_motor_train, y_motor_train)

print('Intercept: ', model_motor.intercept_)
print('Coefficient: ', model_motor.coef_)

y_motor_pred = model_motor.predict(X_motor_test)
# # print(y_motor_pred)

y_motor_test = y_motor_test.reshape(-1)
y_motor_pred = y_motor_pred.reshape(-1)

df_motor_pred = pd.DataFrame({'Actual': y_motor_test, 'Predicted': y_motor_pred})
print(df_motor_pred)

mae_motor = metrics.mean_absolute_error(y_motor_test, y_motor_pred)
mse_motor = metrics.mean_squared_error(y_motor_test, y_motor_pred)
rmse_motor = math.sqrt(mse_motor)
y_motor_max = y_motor_test.max()
y_motor_min = y_motor_test.min()
rmse_norm_motor = rmse_motor/(y_motor_max-y_motor_min)
r_2_motor = metrics.r2_score(y_motor_test, y_motor_pred)
Adj_r_2_motor = 1-(1-r_2_motor)*(len(y_motor)-1)/(len(y_motor)-X_motor.shape[1]-1)

print('MLR Performance for Motor UPDRS Prediction\n')
print('MAE for Motor UPDRS Prediction Model: ', mae_motor)
print('MSE for Motor UPDRS Prediction Model: ', mse_motor)
print('RMSE for Motor UPDRS Prediction Model: ', rmse_motor)
print('Normalised RMSE for Motor UPDRS Prediction Model: ', rmse_norm_motor)
print('R-Sqaured for Motor UPDRS Prediction Model: ', r_2_motor)
print('Adjusted R-Sqaured for Motor UPDRS Prediction Model: ', Adj_r_2_motor)

y_base = np.mean(y_motor_train)
y_pred_base = [y_base] * len(y_motor_test)
print('Y mean: ', y_base)

df_pred_base = pd.DataFrame({'Actual': y_motor_test, 'Predicted': y_pred_base})
print(df_pred_base)

mae_base = metrics.mean_absolute_error(y_motor_test , y_pred_base)
mse_base = metrics.mean_squared_error(y_motor_test, y_pred_base)
rmse_base = math.sqrt(mse_base)
y_max = y_motor_test.max()
y_min = y_motor_test.min()
rmse_norm_base = rmse_base/(y_max - y_min)
r_2_base = metrics.r2_score(y_motor_test, y_pred_base)

print('MAE Base: ', mae_base)
print('MSE Base; ', mse_base)
print('RMSE Base: ', rmse_base)
print('RMSE Base (Normalised): ', rmse_norm_base)
print('R-squared Base: ', r_2_base)

fig, axs = plt.subplots(4,3)
fig.tight_layout()

axs[0,0].scatter(x = df['age'], y = df['motor_updrs'])
axs[0,0].set_xlabel('Age')
axs[0,0].set_ylabel('Motor_UPDRS')

axs[0,1].scatter(x = df['sex'], y = df['motor_updrs'])
axs[0,1].set_xlabel('Sex')
axs[0,1].set_ylabel('Motor_UPDRS')

axs[0,2].scatter(x = df['test_time'], y = df['motor_updrs'])
axs[0,2].set_xlabel('Test Time')
axs[0,2].set_ylabel('Motor_UPDRS')

axs[1,0].scatter(x = df['jitter(%)'], y = df['motor_updrs'])
axs[1,0].set_xlabel('Jitter %')
axs[1,0].set_ylabel('Motor_UPDRS')

axs[1,1].scatter(x = df['jitter(abs)'], y = df['motor_updrs'])
axs[1,1].set_xlabel('Absolute Jitter')
axs[1,1].set_ylabel('Motor_UPDRS')

axs[1,2].scatter(x = df['jitter(rap)'], y = df['motor_updrs'])
axs[1,2].set_xlabel('Jitter - RAP')
axs[1,2].set_ylabel('Motor_UPDRS')

axs[2,0].scatter(x = df['jitter(ppq5)'], y = df['motor_updrs'])
axs[2,0].set_xlabel('Jitter - PPQ5')
axs[2,0].set_ylabel('Motor_UPDRS')

axs[2,1].scatter(x = df['jitter(ddp)'], y = df['motor_updrs'])
axs[2,1].set_xlabel('Jitter - DDP')
axs[2,1].set_ylabel('Motor_UPDRS')

axs[2,2].scatter(x = df['shimmer(%)'], y = df['motor_updrs'])
axs[2,2].set_xlabel('Shimmer %')
axs[2,2].set_ylabel('Motor_UPDRS')

axs[3,0].scatter(x = df['shimmer(abs)'], y = df['motor_updrs'])
axs[3,0].set_xlabel('Absolute Shimmer')
axs[3,0].set_ylabel('Motor_UPDRS')

axs[3,1].scatter(x = df['shimmer(apq3)'], y = df['motor_updrs'])
axs[3,1].set_xlabel('Shimmer - APQ3')
axs[3,1].set_ylabel('Motor_UPDRS')

axs[3,2].scatter(x = df['shimmer(apq5)'], y = df['motor_updrs'])
axs[3,2].set_xlabel('Shimmer - APQ5')
axs[3,2].set_ylabel('Motor_UPDRS')


fig_2, axs_2 = plt.subplots(3,3)
fig_2.tight_layout()

axs_2[0,0].scatter(x = df['shimmer(apq11)'], y = df['motor_updrs'])
axs_2[0,0].set_xlabel('Shimmer - APQ11')
axs_2[0,0].set_ylabel('Motor_UPDRS')

axs_2[0,1].scatter(x = df['shimmer(dda)'], y = df['motor_updrs'])
axs_2[0,1].set_xlabel('Shimmer DDA')
axs_2[0,1].set_ylabel('Motor_UPDRS')

axs_2[0,2].scatter(x = df['nhr'], y = df['motor_updrs'])
axs_2[0,2].set_xlabel('Noise to Harmonic Ratio')
axs_2[0,2].set_ylabel('Motor_UPDRS')

axs_2[1,0].scatter(x = df['hnr'], y = df['motor_updrs'])
axs_2[1,0].set_xlabel('Harmonic to Noise Ratio')
axs_2[1,0].set_ylabel('Motor_UPDRS')

axs_2[1,1].scatter(x = df['rpde'], y = df['motor_updrs'])
axs_2[1,1].set_xlabel('Recurrance Period Density')
axs_2[1,1].set_ylabel('Motor_UPDRS')

axs_2[1,2].scatter(x = df['dfa'], y = df['motor_updrs'])
axs_2[1,2].set_xlabel('Detrended Fluctuation Analysis')
axs_2[1,2].set_ylabel('Motor_UPDRS')

axs_2[2,0].scatter(x = df['ppe'], y = df['motor_updrs'])
axs_2[2,0].set_xlabel('Pitch Period Entropy')
axs_2[2,0].set_ylabel('Motor_UPDRS')

df_motor = df.drop(['total_updrs', 'subject#'], axis=1)
corr = df_motor.corr()

fig_3, ax = plt.subplots()
fig_3.tight_layout()

ax = sns.heatmap(corr, vmax = 1, vmin = -1, center = 0, 
                  cmap = sns.diverging_palette(20, 220, n = 200), 
                  square = False, annot = True)

ax.set_xticklabels(ax.get_xticklabels(), rotation =45, horizontalalignment = 'right')

plt.show()

#Basic Model for Motor UPDRS Prediction without Total UPDRS

print('\n----------Basic Model for Motor UPDRS Prediction without Total UPDRS----------\n')

X_motor_ols = df_motor.iloc[: , :-1].values
y_motor_ols = df_motor.iloc[: , -1].values

X_motor_ols = pd.DataFrame(X_motor_ols, columns=df_motor.columns[:-1])

X_motor_ols = sm.add_constant(X_motor_ols)
model_motor_ols = sm.OLS(y_motor_ols,X_motor_ols).fit()
pred_motor_ols = model_motor_ols.predict(X_motor_ols)
model_details = model_motor_ols.summary()
print(model_details)

# plt.show()

#Log Transformation

print('\n-----------Log Transformation for Motor UPDRS Prediction without Total UPDRS----------\n')

df_motor_LT = df_motor.copy()

columns_to_log = ['age', 'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
                  'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 
                  'shimmer(dda)', 'nhr', 'hnr']

df_motor_LT[['LAge', 'LJitter(%)', 'LAbs Jitter', 'LJitter RAP', 'LJitter PPQ5', 'LJitter DDP',
             'LShimmer(%)', 'LAbs Shimmer', 'LShimmer APQ3', 'LShimmer APQ5', 'LShimmer APQ11', 
             'LShimmer DDA', 'LNHR', 'LHNR']] = df_motor_LT[columns_to_log].apply(lambda x: np.log(x))

df_motor_LT = df_motor_LT.drop(['age', 'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 
                    'jitter(ddp)', 'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 
                    'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)', 'nhr','hnr'],
                    axis = 1)

df_motor_LT = df_motor_LT[['LAge', 'sex', 'test_time','LJitter(%)','LAbs Jitter',	
        'LJitter RAP', 'LJitter PPQ5', 'LJitter DDP', 'LShimmer(%)', 
        'LAbs Shimmer', 'LShimmer APQ3', 'LShimmer APQ5', 'LShimmer APQ11', 
        'LShimmer DDA', 'LNHR', 'LHNR', 'rpde', 'dfa', 'ppe', 'motor_updrs']]
# print(df_motor)

X_motor_ols = df_motor_LT.iloc[: , :-1].values
y_motor_ols = df_motor_LT.iloc[: , -1].values

X_motor_ols = pd.DataFrame(X_motor_ols, columns=df_motor_LT.columns[:-1])

X_motor_ols = sm.add_constant(X_motor_ols)
model_motor_ols = sm.OLS(y_motor_ols,X_motor_ols).fit()
pred_motor_ols = model_motor_ols.predict(X_motor_ols)
model_details = model_motor_ols.summary()
print(model_details)

#Colinearity Analysis

print('\n-----------Colinearity Analysis for Motor UPDRS Prediction without Total UPDRS----------\n')

df_motor_CL = df_motor.copy()

df_motor_CL = df_motor_CL.drop(['jitter(%)', 'jitter(rap)', 'jitter(ddp)', 
                    'shimmer(abs)', 'shimmer(apq3)', 'shimmer(dda)', 'rpde'],
                    axis = 1)

X_motor_ols = df_motor_CL.iloc[: , :-1].values
y_motor_ols = df_motor_CL.iloc[: , -1].values

X_motor_ols = pd.DataFrame(X_motor_ols, columns=df_motor_CL.columns[:-1])

X_motor_ols = sm.add_constant(X_motor_ols)
model_motor_ols = sm.OLS(y_motor_ols,X_motor_ols).fit()
pred_motor_ols = model_motor_ols.predict(X_motor_ols)
model_details = model_motor_ols.summary()
print(model_details)

# print(df_motor)
# print(df_motor_CL)

#Standardization

print('\n-----------Z-Standardization for Motor UPDRS Prediction without Total UPDRS----------\n')

X_motor_ols = df_motor.iloc[: , :-1].values
y_motor_ols = df_motor.iloc[: , -1].values

X_motor_ols = pd.DataFrame(X_motor_ols, columns=df_motor.columns[:-1])

X_motor_ols = sm.add_constant(X_motor_ols)

scaler = StandardScaler()

X_motor_ols = X_motor_ols.drop(['const'], axis = 1)

std_X_motor = scaler.fit_transform(X_motor_ols)

std_X_motor_df = pd.DataFrame(std_X_motor, index=X_motor_ols.index, columns=X_motor_ols.columns)

std_X_motor_df = sm.add_constant(std_X_motor_df)

# print(std_X_motor_df)

model_motor_ols = sm.OLS(y_motor_ols,std_X_motor_df).fit()
pred_motor_ols = model_motor_ols.predict(std_X_motor_df)
model_details = model_motor_ols.summary()
print(model_details)

#Gaussian Transformation

print('\n-----------Gaussian Transformation for Motor UPDRS Prediction without Total UPDRS----------\n')

X_motor_ols = df_motor.iloc[: , :-1].values
y_motor_ols = df_motor.iloc[: , -1].values

X_motor_ols = pd.DataFrame(X_motor_ols, columns=df_motor.columns[:-1])

X_motor_ols = sm.add_constant(X_motor_ols)

scaler = PowerTransformer()

X_motor_ols = X_motor_ols.drop(['const'], axis = 1)

GT_X_motor = scaler.fit_transform(X_motor_ols)

GT_X_motor_df = pd.DataFrame(GT_X_motor, index=X_motor_ols.index, columns=X_motor_ols.columns)

GT_X_motor_df = sm.add_constant(GT_X_motor_df)

# print(std_X_motor_df)

model_motor_ols = sm.OLS(y_motor_ols,GT_X_motor_df).fit()
pred_motor_ols = model_motor_ols.predict(GT_X_motor_df)
model_details = model_motor_ols.summary()
print(model_details)