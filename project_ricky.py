# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%% [markdown]

#%%

### Preprocess
#Combine original train and test datasets

from numpy.core.fromnumeric import mean, std
import pandas as pd

train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)

airline = pd.concat([train,test], ignore_index = True)


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

#%%
airline.to_csv('airline.csv')
#%%
dfChkBasics(airline)
#%%
#Summary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot

#Check the basic info of the dataset, see the data columns, their types, and the shape
airline = pd.read_csv('airline.csv', index_col=0)
airline.shape

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
sns.countplot(x="Age_Range", data=airline, order=airline['Age_Range'].value_counts().index)
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

# Does a passenger tend to be satisfied with his or her trip 
# based on the passenger class?

import statsmodels.api as sm
from statsmodels.formula.api import glm
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Logit Regression?
# evaluation confusion matrix

#not train test split logistic

Satisfaction_Class_Model = glm(formula = 'Satisfaction_satisfied ~ C(Class_Number)', data = airline, family=sm.families.Binomial())

Satisfaction_Class_Model_Fit = Satisfaction_Class_Model.fit()
print(Satisfaction_Class_Model_Fit.summary())

airline_predictions = pd.DataFrame( columns=['sm_logit_pred'], data= Satisfaction_Class_Model_Fit.predict(airline)) 
airline_predictions['actual'] = airline['Satisfaction_satisfied']
airline_predictions
#%%
cut_off = 0.5
# Compute class predictions
airline_predictions['satisfaction_div'] = np.where(airline_predictions['sm_logit_pred'] > cut_off, 1, 0)
#print(airline_predictions.satisfaction_div.head())

# Make a cross table
print(pd.crosstab(airline.Satisfaction_satisfied, airline_predictions.satisfaction_div,
rownames=['Actual'], colnames=['Predicted'],
margins = True))

#0.75 accuracy

#%%
print('stats olm 0.75 accuracy')

#%%
# Classification Tree?
# evaluate confusion matrix
# classification report code from tree.py

# Random Forest?
# Roc Auc test?

#%%
airline1 = airline.copy()
airline1 = airline1.rename(columns={"Flight Distance": "Flight_Distance", "Inflight wifi service": "Inflight_wifi_service",
'Departure/Arrival time convenient': 'Departure_Arrival_time_convenient', 'Ease of Online booking': 'Ease_of_Online_booking',
'Gate location': 'Gate_location', 'Food and drink': 'Food_and_drink', 'Online boarding': 'Online_boarding',
'Seat comfort': 'Seat_comfort', 'Inflight entertainment': 'Inflight_entertainment', 'On-board service': 'On_board_service',
'Leg room service': 'Leg_room_service', 'Baggage handling':'Baggage_handling', 'Checkin service': 'Checkin_service',
'Inflight service': 'Inflight_service', 'Total Delay in Minutes': 'Total_Delay_in_Minutes'})

#%%
Satisfaction_Class_Model2 = glm(formula = 'Satisfaction_satisfied ~ Inflight_wifi_service + Departure_Arrival_time_convenient + Ease_of_Online_booking + Gate_location + Food_and_drink + Online_boarding + Seat_comfort + Inflight_entertainment + On_board_service + Leg_room_service + Checkin_service + Inflight_service + C(Class_Number)', data = airline1, family=sm.families.Binomial())

Satisfaction_Class_Model_Fit2 = Satisfaction_Class_Model2.fit()
print(Satisfaction_Class_Model_Fit2.summary())

airline_predictions2 = pd.DataFrame( columns=['sm_logit_pred'], data= Satisfaction_Class_Model_Fit2.predict(airline1)) 
airline_predictions2['actual'] = airline['Satisfaction_satisfied']
airline_predictions2
#%%
cut_off = 0.5
# Compute class predictions
airline_predictions2['satisfaction_div'] = np.where(airline_predictions2['sm_logit_pred'] > cut_off, 1, 0)
#print(airline_predictions.satisfaction_div.head())

# Make a cross table
print(pd.crosstab(airline1.Satisfaction_satisfied, airline_predictions2.satisfaction_div,
rownames=['Actual'], colnames=['Predicted'],
margins = True))

#0.84 accuracy if add up the rating scores
#%%

#Forest using only Class

y = airline['Satisfaction_satisfied']
X = airline['Class_Number']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True)

X_train = np.array(X_train.values.tolist()).reshape(-1,1)
y_train = np.ravel(y_train)

X_test = np.array(X_test.values.tolist()).reshape(-1,1)
y_test = np.ravel(y_test)

forest = RandomForestClassifier(n_estimators = 1000, max_depth = 5)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#%%
class_number = np.array(airline1['Class_Number'].values.tolist()).reshape(-1,1)

airline_predictions3 = pd.DataFrame( columns=['forest_pred'], data=forest.predict(class_number)) 
airline_predictions3['actual'] = airline1['Satisfaction_satisfied']
airline_predictions3

#%%
#ROC AUC Eval

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = forest.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('Random Forest: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

#%%

#Forest using all rating scores

y = airline['Satisfaction_satisfied']
X = airline[['Class_Number','Inflight wifi service','Departure/Arrival time convenient', 'Ease of Online booking',
'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
'Inflight entertainment', 'On-board service', 'Leg room service',
'Baggage handling', 'Checkin service', 'Inflight service',
'Cleanliness']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

#%%
forest1 = RandomForestClassifier(n_estimators = 1000, max_depth = 5, max_features='auto', n_jobs=1)
forest1.fit(X_train, y_train)
y_pred = forest1.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
#%%
columns = airline[['Class_Number','Inflight wifi service','Departure/Arrival time convenient', 'Ease of Online booking',
'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
'Inflight entertainment', 'On-board service', 'Leg room service',
'Baggage handling', 'Checkin service', 'Inflight service',
'Cleanliness']]

airline_predictions4 = pd.DataFrame(columns=['forest_pred'], data=forest1.predict(columns)) 
airline_predictions4['actual'] = airline['Satisfaction_satisfied']
airline_predictions4

#%%
#ROC AUC Eval

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = forest1.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('Random Forest: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Random Forest')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()






























#%%
column_names = ['Class_Number','Inflight wifi service','Departure/Arrival time convenient', 'Ease of Online booking',
'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
'Inflight entertainment', 'On-board service', 'Leg room service',
'Baggage handling', 'Checkin service', 'Inflight service',
'Cleanliness']
feature_importance_score = pd.Series(forest1.feature_importances_,index=column_names).sort_values(ascending=False)
feature_importance_score

#%%
#Forest using some important features

y = airline['Satisfaction_satisfied']
X = airline[['Online boarding', 'Class_Number', 'Inflight wifi service',
'Inflight entertainment', 'Seat comfort', 'Leg room service']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

forest2 = RandomForestClassifier(n_estimators = 1000, max_depth = 5, max_features='auto', n_jobs=1)
forest2.fit(X_train, y_train)
y_pred = forest2.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred))


#%%
RandomForestClassifier(n_estimators=1000, max_depth=5, max_features='auto', n_jobs=1)
# %%
