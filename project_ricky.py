# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%% [markdown]

#%%

### Preprocess
#Combine original train and test datasets

train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)

airline = pd.concat([train,test])


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

#airline.to_csv('airline.csv')
#%%
#Summary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot

#Check the basic info of the dataset, see the data columns, their types, and the shape
airline = pd.read_csv('airline.csv', index_col=0)
dfChkBasics(airline)

# %%
#pivot table of customer types by types of travels valued by average flight distance
airline.pivot_table(index='Customer Type', columns='Type of Travel' , values='Flight Distance', aggfunc = np.mean)
# %%
#pivot table of customer types by types of travels valued by average total delay minutes
airline.pivot_table(index='Customer Type', columns='Type of Travel' , values='Total Delay in Minutes', aggfunc = np.mean)
# %%
#pivot table of customer types by satisfaction valued by average flight distance
airline.pivot_table(index='Customer Type', columns='Satisfaction' , values='Flight Distance', aggfunc = np.mean)
# %%
#pivot table of customer types by satisfaction valued by average total delay minutes
airline.pivot_table(index='Customer Type', columns='Satisfaction' , values='Total Delay in Minutes', aggfunc = np.mean)


#%%
#Piechart of Satisfaction
sat = [1,2]
labels = ['Neutral/Dissatisfied', 'Satisfied']

fig, axs = plt.subplots() 
sns.set_style("whitegrid")
axs.pie(sat, labels = labels, startangle = 90, shadow = True, autopct='%1.2f%%')
axs.set_title('Piechart of Satisfaction')

#%%
#youngest age 7 and oldest 85
#Divide the Age Range

airline['Age_Range'] = ''
airline.loc[airline['Age'] <= 20, 'Age_Range'] = '<20'
airline.loc[(airline['Age'] > 20) & (airline['Age'] <= 40), 'Age_Range'] = '21-40'
airline.loc[(airline['Age'] > 40) & (airline['Age'] <= 60), 'Age_Range'] = '41-60'
airline.loc[airline['Age'] > 60, 'Age_Range'] = '>60'

#%%
#Barplot shows customers' age ranges
fig, axs = plt.subplots() 
sns.set_style(style="whitegrid")
sns.countplot(x="Age_Range", data=airline)
plt.title('Customers in Different Ranges of Ages')
plt.show()

#%%
#Violinplot of flight distance by class splitted by satisfaction

fig, axs = plt.subplots() 
sns.set_style(style="whitegrid")
sns.violinplot(x="Class", y="Flight Distance", hue="Satisfaction",
                    data=airline, palette = 'Set1', split=True)
plt.title('Violinplot of Flight Distances in Each Class splited by Satisfaction')

#%%
#a plot of customer count by class and type of travel (could be a stackplot)

df = airline.groupby(['Class', 'Type of Travel']).size().reset_index().pivot(columns = 'Class', index = 'Type of Travel', values = 0)
df.plot(kind='bar', stacked=True)
plt.xticks(rotation = 360)
plt.title('Customers by Type of Travel Stacked by Classes')

#%%
#Normality Visual for quantitative columns
columns = ['Age','Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Total Delay in Minutes']
def normal_visual(df, column):
  sns.distplot(df[column])
  title = 'Distplot of ' + column
  plt.title(title)

  qqplot(df[column], line = 's')
  title = 'Q-Q plot of ' + column
  plt.title(title)
  plt.show()

for i in columns:
  normal_visual(airline, i)

# %%

# Encode the satisfication and class
# eco 1, eco plus 2, bus 3
# satisfied 1 neutral-unsatisfied 0

airline['Class_Number'] = 0
airline.loc[airline['Class'] == 'Eco', 'Class_Number'] = 1
airline.loc[airline['Class'] == 'Eco Plus', 'Class_Number'] = 2
airline.loc[airline['Class'] == 'Business', 'Class_Number'] = 3

airline = pd.get_dummies(airline, columns=["Satisfaction"])

# use Class_Number and Satisfaction_satisfied

# %%
# question 2 (modeling etc) 

# Logit Regression?
# evaluation confusion matrix

# Classification Tree?
# evaluate confusion matrix
# classification report code from tree.py

# Random Forest?
# Roc Auc test?

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

xairline = airline['Class']
yairline = airline['Satisfaction']
X_train, X_test, y_train, y_test= train_test_split(xairline, yairline, test_size=0.3, stratify=yairline, random_state=1)


# %%
