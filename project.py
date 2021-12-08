# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%[markdown]
# # **Airline Satisfaction Investigation**
# By: TEAM 8 ~ Atharva Haldankar, Charles Graham, Ruiqi Li, Yixi Liang

#%%
# Import all required packages/modules
from re import X
import numpy as np
from numpy.core.fromnumeric import mean, std, var
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.genmod.families import family

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve

#%%[markdown]
## **Preprocessing**
#%%
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
#Since the number of records is 103594 while NAs only 310, it's not a big concern to drop them
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

airline.to_csv('airline.csv')
dfChkBasics(airline)
#%%[markdown]
# ## **EDA**

#%%
#Check the basic info of the dataset, see the data columns, their types, and the shape
airline = pd.read_csv('airline.csv', index_col=0)

#%%
# pivot table of customer types by types of travels valued by average flight distance
airline.pivot_table(index='Customer Type', columns='Type of Travel' , values='Flight Distance', aggfunc = np.mean)

#%%
# pivot table of customer types by types of travels valued by average total delay minutes
airline.pivot_table(index='Customer Type', columns='Type of Travel' , values='Total Delay in Minutes', aggfunc = np.mean)

#%%
# pivot table of customer types by satisfaction valued by average flight distance
airline.pivot_table(index='Customer Type', columns='Satisfaction' , values='Flight Distance', aggfunc = np.mean)

#%%
# pivot table of customer types by satisfaction valued by average total delay minutes
airline.pivot_table(index='Customer Type', columns='Satisfaction' , values='Total Delay in Minutes', aggfunc = np.mean)


#%%
# Piechart of Satisfaction
sat = [1,2]
labels = ['Neutral/Dissatisfied', 'Satisfied']

fig, axs = plt.subplots() 
sns.set_style("whitegrid")
axs.pie(sat, labels = labels, startangle = 90, shadow = False, autopct='%1.2f%%')
axs.set_title('Piechart of Satisfaction')

#%%
# youngest age 7 and oldest 85
# Divide the Age Range

airline['Age_Range'] = ''
airline.loc[airline['Age'] <= 20, 'Age_Range'] = '<20'
airline.loc[(airline['Age'] > 20) & (airline['Age'] <= 40), 'Age_Range'] = '21-40'
airline.loc[(airline['Age'] > 40) & (airline['Age'] <= 60), 'Age_Range'] = '41-60'
airline.loc[airline['Age'] > 60, 'Age_Range'] = '>60'

# Barplot shows customers' age ranges
fig, axs = plt.subplots() 
sns.set_style(style="whitegrid")
sns.countplot(x="Age_Range", data=airline, order=airline['Age_Range'].value_counts().index)
plt.title('Customers in Different Ranges of Ages')
plt.show()

#%%
# a plot of customer count by class and type of travel (could be a stackplot)

df = airline.groupby(['Class', 'Type of Travel']).size().reset_index().pivot(columns = 'Class', index = 'Type of Travel', values = 0)
df.plot(kind='bar', stacked=True)
plt.xticks(rotation = 360)
plt.title('Customers by Type of Travel Stacked by Classes')

#%%
# Normality Visual for quantitative columns
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

#%%
#Flight distance and types travel for cutomer types. 
sns.boxplot(x="Type of Travel", y="Flight Distance",hue="Customer Type",data=airline)
plt.title("Cutomer's purpose of travel vs distance of the flight and their loyalty")
#%%
#Flight distance and types travel for cutomer types. 
# sns.boxplot(x="Type of Travel", y="Flight Distance",hue="Customer Type",data=airline)
# plt.scatter(y= "Arrival Delay in Minutes",x ="Flight Distance" ,data=airline)
# plt.hist([airline["Flight Distance"],airline["Arrival Delay in Minutes"]], alpha=0.5, label=['World 1','World 2'],edgecolor='black', linewidth=1)
sns.scatterplot(airline["Departure Delay in Minutes"],airline["Arrival Delay in Minutes"])
plt.title("Correlation between delay during the departure and the delay during arrival")
#%%
plt.scatter(x="Departure Delay in Minutes",y="Flight Distance",data=airline)
plt.xlabel("Delay in Minutes")
plt.ylabel("Flight Distance")
plt.scatter(x="Arrival Delay in Minutes",y="Flight Distance",data=airline,edgecolors= "red",alpha=0.25)
plt.xlabel("Delay in Minutes")
plt.ylabel("Flight Distance")
plt.legend(["Departure Delay","Arrival Delay"])
plt.show
plt.title("Does flight distance have an effect on the delay during departure and arrival?")

#%%
airline["avg_rating_score"] = airline[["Inflight wifi service","Departure/Arrival time convenient","Ease of Online booking","Gate location","Food and drink","Online boarding","Seat comfort","Inflight entertainment"	,"On-board service","Leg room service","Baggage handling",	"Checkin service","Inflight service","Cleanliness"]].mean(axis=1)
sns.violinplot(x='Class',y='avg_rating_score',hue='Gender',split= True,data= airline,saturation=1,palette= "Set1",order=["Eco","Eco Plus","Business"])
plt.title("Average rating by gender and class")
plt.show()










# Models
# %%
