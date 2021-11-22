# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%% [markdown]

#%%
from re import X
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

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
#%%
#Flight distance and types travel for cutomer types. 
sns.boxplot(x="Type of Travel", y="Flight Distance",hue="Customer Type",data=airline)

#%%
#Flight distance and types travel for cutomer types. 
# sns.boxplot(x="Type of Travel", y="Flight Distance",hue="Customer Type",data=airline)
# plt.scatter(y= "Arrival Delay in Minutes",x ="Flight Distance" ,data=airline)
# plt.hist([airline["Flight Distance"],airline["Arrival Delay in Minutes"]], alpha=0.5, label=['World 1','World 2'],edgecolor='black', linewidth=1)
sns.scatterplot(airline["Departure Delay in Minutes"],airline["Arrival Delay in Minutes"])
#%%
plt.scatter(x="Departure Delay in Minutes",y="Flight Distance",data=airline)
plt.xlabel("Depature Delay in Minutes")
plt.ylabel("Flight Distance")
plt.scatter(x="Arrival Delay in Minutes",y="Flight Distance",data=airline,edgecolors= "red",alpha=0.2)
plt.xlabel("Depature Delay in Minutes")
plt.ylabel("Flight Distance")
plt.legend(["Departure Delay","Arrival Delay"])
plt.show


#%%
airline["avg_rating_score"] = airline[["Inflight wifi service","Departure/Arrival time convenient","Ease of Online booking","Gate location","Food and drink","Online boarding","Seat comfort","Inflight entertainment"	,"On-board service","Leg room service","Baggage handling",	"Checkin service","Inflight service","Cleanliness"]].mean(axis=1)
sns.violinplot(x='Class',y='avg_rating_score',hue='Gender',split= True,data= airline,saturation=1,palette= "Set1",order=["Eco","Eco Plus","Business"])
plt.title("Average rating by gender and class")
plt.show()

#Model
#%%
# Which rating scores has the strongest correlation with satisfaction?
airline.satisfaction = pd.Categorical(airline.satisfaction,["neutral or dissatisfied","satisfied"],ordered=True)
airline.satisfaction = airline.satisfaction.cat.codes
#%%
df = airline[["satisfaction","Inflight wifi service","Departure/Arrival time convenient","Ease of Online booking","Gate location","Food and drink","Online boarding","Seat comfort","Inflight entertainment"	,"On-board service","Leg room service","Baggage handling",	"Checkin service","Inflight service","Cleanliness"]]
df.corr()
sns.heatmap(df.corr())
# From the heat map it is evident that the Online boarding rating has comparatively the strongest correlation with satisfaction compared to the rest of the variables. 
# %%
