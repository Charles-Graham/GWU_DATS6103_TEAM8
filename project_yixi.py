# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%% [markdown]

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
### Preprocess
#Combine original train and test datasets

train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)

airline = pd.concat([train,test],ignore_index=True)


#Check any existing null values
for i in airline:
  print(i + ' has ' + str(airline[i].isnull().sum()) + ' nulls')

#Found 310 nulls in arrival delay in minutes column
#Drop the NAs
airline = airline[airline['Arrival Delay in Minutes'].isnull() == False]

#df without NAs
#Since the number of records is 103594 while NAs only 310, so not a big concern if dropping them
#Add new column of Total Delay Minutes, and switch the column order with satisfaction

def dfChkBasics(dframe, valCnt = False): 
  cnt = 1
  print('\ndataframe Basic Check function -')
  
  try:
    print(f'\n{cnt}: info(): ')
    cnt+=1
    print(dframe.info())
  except: pass

  print(f'\n{cnt}: describe(): ')
  cnt+=1
  print(dframe.describe())

  print(f'\n{cnt}: dtypes: ')
  cnt+=1
  print(dframe.dtypes)

  try:
    print(f'\n{cnt}: columns: ')
    cnt+=1
    print(dframe.columns)
  except: pass

  print(f'\n{cnt}: head() -- ')
  cnt+=1
  print(dframe.head())

  print(f'\n{cnt}: shape: ')
  cnt+=1
  print(dframe.shape)

  if (valCnt):
    print('\nValue Counts for each feature -')
    for colname in dframe.columns :
      print(f'\n{cnt}: {colname} value_counts(): ')
      print(dframe[colname].value_counts())
      cnt +=1


airline['Total Delay in Minutes'] = airline['Departure Delay in Minutes'] + airline['Arrival Delay in Minutes']
temp = airline['satisfaction']
airline = airline.drop(columns = ['satisfaction'])
airline['Satisfaction'] = temp
# %% YIXI part
# change columns name for model
airlineQ3 = airline.rename({'Departure Delay in Minutes': 'ddim', 'Arrival Delay in Minutes': 'adim', 'Total Delay in Minutes':'tdim'}, axis=1)
airlineQ3["Satisfaction"] = np.where(airlineQ3['Satisfaction'] == 'neutral or dissatisfied', 0, 1)
# reindex
#airlineQ3.reset_index(drop=True, inplace=True)
#%%
# Is there a relationship between total delay time and departure/arrival convenience satisfaction?
import statsmodels.api as sm 
from statsmodels.formula.api import glm
#%% model of Satisfaction ~ ddim + adim
modelDelayLogitFit = glm(formula='Satisfaction ~ ddim + adim', data=airlineQ3, family=sm.families.Binomial()).fit()
print( modelDelayLogitFit.summary())
# Since the p-value is extremely small, Departure Delay in Minutes and Arrival Delay in Minutes have strong relationship with Satisfaction
modelPredicitonOfDelay = pd.DataFrame( columns=['logit_ddimAdim'], data= modelDelayLogitFit.predict(airlineQ3)) 
print(dfChkBasics(modelPredicitonOfDelay))
#%%
# Confusion matrix
# Define cut-off value
cut_off = 0.4
# Compute class predictions
modelPredicitonOfDelay['logit_ddimAdim_result'] = np.where(modelPredicitonOfDelay['logit_ddimAdim'] > cut_off, 1, 0)
# %%
# Make a cross table
print(pd.crosstab(airlineQ3.Satisfaction, modelPredicitonOfDelay.logit_ddimAdim_result,
rownames=['Actual'], colnames=['Predicted'], margins = True))


#%% model of Satisfaction ~ tdim
modelDelayTdimLogitFit = glm(formula='Satisfaction ~ tdim', data=airlineQ3, family=sm.families.Binomial()).fit()
print( modelDelayTdimLogitFit.summary())
# Since the p-value is extremely small, Departure Delay in Minutes and Arrival Delay in Minutes have strong relationship with Satisfaction
modelPredicitonOfDelay['logit_tdim'] = modelDelayTdimLogitFit.predict(airlineQ3)
print(dfChkBasics(modelPredicitonOfDelay))
# Confusion matrix
# Define cut-off value
cut_off = 0.4
# Compute class predictions
modelPredicitonOfDelay['logit_tdim_result'] = np.where(modelPredicitonOfDelay['logit_tdim'] > cut_off, 1, 0)
# %%
# Make a cross table
print(pd.crosstab(airlineQ3['Satisfaction'], modelPredicitonOfDelay['logit_tdim_result'],
rownames=['Actual'], colnames=['Predicted'], margins = True))
# %%
