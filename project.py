# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%% [markdown]

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

#%%

### Preprocess

#%%
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)

airline = pd.concat([train,test])
dfChkBasics(airline)
#%%

#Check any existing null values

for i in airline:
  print(i + ' has ' + str(airline[i].isnull().sum()) + ' nulls')
#%%

#Found 310 nulls in arrival delay in minutes column

#Drop the NAs

airline = airline[airline['Arrival Delay in Minutes'].isnull() == False]
airline.to_csv('airline.csv')
#%%
#df without NAs

#Since the number of records is 103594 while NAs only 310, so not a big concern if dropping them

dfChkBasics(airline)








#EDA

#Model
# %%
