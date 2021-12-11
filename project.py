# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%[markdown]
# # **Airline Satisfaction Investigation**
# By: TEAM 8 ~ Atharva Haldankar, Charles Graham, Ruiqi Li, Yixi Liang

#%%
# Import all required packages/modules
try:
  import dataframe_image as dfi
except ModuleNotFoundError:
  !pip install dataframe_image
except ImportError:
  !pip install dataframe_image

import timeit
from re import X
import numpy as np
from numpy.core.fromnumeric import mean, std, var
import pandas as pd
from pandas.plotting import scatter_matrix
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm
from statsmodels.formula.api import glm

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix, classification_report, r2_score

#%%[markdown]
## **Preprocessing**
#%%
#Combine original train and test datasets

train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)

train.rename(columns={"satisfaction": "Satisfaction"}, inplace=True)
test.rename(columns={"satisfaction": "Satisfaction"}, inplace=True)

airline = pd.concat([train,test], ignore_index=True)


#Check any existing null values
for i in airline:
  print(i + ' has ' + str(airline[i].isnull().sum()) + ' nulls')

#Found 310 nulls in arrival delay in minutes column
#Drop the NAs
airline = airline[airline['Arrival Delay in Minutes'].isnull() == False]

#df without NAs
#Since the number of records is 103594 while NAs only 310, it's not a big concern to drop them
#Add new column of Total Delay Minutes, and switch the column order with Satisfaction

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
temp = airline['Satisfaction']
airline = airline.drop(columns = ['Satisfaction'])
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
# pivot table of customer types by Satisfaction valued by average flight distance
airline.pivot_table(index='Customer Type', columns='Satisfaction' , values='Flight Distance', aggfunc = np.mean)

#%%
# pivot table of customer types by Satisfaction valued by average total delay minutes
airline.pivot_table(index='Customer Type', columns='Satisfaction' , values='Total Delay in Minutes', aggfunc = np.mean)


#%%
# Piechart of Satisfaction
sat = [1,2]
labels = ['Neutral/Dissatisfied', 'Satisfied']

fig, axs = plt.subplots() 
sns.set_style("whitegrid")
axs.pie(sat, labels = labels, startangle = 90, shadow = True, autopct='%1.2f%%')
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

#%%[markdown]
# ## **Modeling**

#%%[markdown]
# ### **Q1:** Which rating scores have the strongest correlations with Satisfaction?

#%%
# airline.Satisfaction = pd.Categorical(airline.Satisfaction,["neutral or dissatisfied","satisfied"],ordered=True)
# airline.Satisfaction = airline.Satisfaction.cat.codes
#%%
df = airline[["Satisfaction","Inflight wifi service","Departure/Arrival time convenient","Ease of Online booking","Gate location","Food and drink","Online boarding","Seat comfort","Inflight entertainment"	,"On-board service","Leg room service","Baggage handling",	"Checkin service","Inflight service","Cleanliness"]]
df.Satisfaction = pd.Categorical(df.Satisfaction,["neutral or dissatisfied","satisfied"],ordered=True)
df.Satisfaction = df.Satisfaction.cat.codes
print(st.shapiro(df)) 
# Since the data is not normal we chose the sperman's test
cort = pd.DataFrame(df.corr(method="spearman"))
print(cort)
#%%
fig, ax = plt.subplots(figsize=(15,15)) 
sns.heatmap(df.corr(method="spearman"),annot=True,fmt = ".2g",ax=ax)
print("From the heat map it is evident that the Online boarding rating has comparatively the strongest correlation with Satisfaction compared to the rest of the variables.")
# %%
df1 = airline.copy()
df1.columns = df1.columns.str.replace(" ","_")
df1.columns = df1.columns.str.replace("-","_")
df1.columns = df1.columns.str.replace("/","_")
df1.var()
satisLogit = glm(formula='Satisfaction ~ Inflight_wifi_service + Departure_Arrival_time_convenient + Ease_of_Online_booking + Gate_location + Food_and_drink + Online_boarding + Seat_comfort + Inflight_entertainment + On_board_service + Leg_room_service + Baggage_handling + Inflight_service +Checkin_service + Cleanliness', data=df1,family=sm.families.Binomial())
satisLogitfit = satisLogit.fit()
print(satisLogitfit.summary())
np.exp(satisLogitfit.params)
np.exp(satisLogitfit.conf_int())
df1["SatisfiedLogit"] = satisLogitfit.predict(df1)
cut_off = 0.55
df1['Statisfied_prediction'] = np.where(df1['SatisfiedLogit'] > cut_off, 1, 0)
print(
  "The regression results show us that all variables are significant (i.e. p-value < alpha = 0.05) "
  "except 'Inflight Service' (p-value > alpha = 0.05)."
)

#%%[markdown]
# ### **Q2:** Does passenger class influence the passenger satisfaction and how accurate can it be for prediction?

#%%
airline['Class_Number'] = 0
airline.loc[airline['Class'] == 'Eco', 'Class_Number'] = 1
airline.loc[airline['Class'] == 'Eco Plus', 'Class_Number'] = 2
airline.loc[airline['Class'] == 'Business', 'Class_Number'] = 3

airline = pd.get_dummies(airline, columns=["Satisfaction"])
airline['Satisfaction'] = temp

# use Class_Number and Satisfaction_satisfied


# Logit Regression?

# not train test split logistic

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
# print(airline_predictions.satisfaction_div.head())

# evaluate confusion matrix
# Make a cross table
print(pd.crosstab(airline.Satisfaction_satisfied, airline_predictions.satisfaction_div,
rownames=['Actual'], colnames=['Predicted'],
margins = True))

print('stats olm 0.75 accuracy')

#%%
# olm with rating scores extra

airline1 = airline.copy()
airline1 = airline1.rename(columns={"Flight Distance": "Flight_Distance", "Inflight wifi service": "Inflight_wifi_service",
'Departure/Arrival time convenient': 'Departure_Arrival_time_convenient', 'Ease of Online booking': 'Ease_of_Online_booking',
'Gate location': 'Gate_location', 'Food and drink': 'Food_and_drink', 'Online boarding': 'Online_boarding',
'Seat comfort': 'Seat_comfort', 'Inflight entertainment': 'Inflight_entertainment', 'On-board service': 'On_board_service',
'Leg room service': 'Leg_room_service', 'Baggage handling':'Baggage_handling', 'Checkin service': 'Checkin_service',
'Inflight service': 'Inflight_service', 'Total Delay in Minutes': 'Total_Delay_in_Minutes'})

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

# Make a cross table
print(pd.crosstab(airline1.Satisfaction_satisfied, airline_predictions2.satisfaction_div,
rownames=['Actual'], colnames=['Predicted'],
margins = True))

print('stats olm with rating scores 0.84 accuracy')
#%%

# Forest using only Class

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
# predictions vs actuals
class_number = np.array(airline1['Class_Number'].values.tolist()).reshape(-1,1)

airline_predictions3 = pd.DataFrame( columns=['forest_pred'], data=forest.predict(class_number)) 
airline_predictions3['actual'] = airline1['Satisfaction_satisfied']
airline_predictions3

#%%
# ROC AUC Eval

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

# Forest using all rating scores

y = airline['Satisfaction_satisfied']
X = airline[['Class_Number','Inflight wifi service','Departure/Arrival time convenient', 'Ease of Online booking',
'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
'Inflight entertainment', 'On-board service', 'Leg room service',
'Baggage handling', 'Checkin service', 'Inflight service',
'Cleanliness']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

forest1 = RandomForestClassifier(n_estimators = 1000, max_depth = 5, max_features='auto', n_jobs=1)
forest1.fit(X_train, y_train)
y_pred = forest1.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
#%%
# predictions vs actuals
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

#%%[markdown]
# ### **Q3:** How does the departure/arrival delay time affect the satisfaction?

#%%
# change columns name for model
airlineQ3 = airline.rename({'Departure Delay in Minutes': 'ddim', 'Arrival Delay in Minutes': 'adim', 
                            'Total Delay in Minutes':'tdim', 'Type of Travel':'tot',
                            'Departure/Arrival time convenient' : 'datc', 'Customer Type' : 'ct'}, axis=1)
airlineQ3["Satisfaction"] = np.where(airlineQ3['Satisfaction'] == 'neutral or dissatisfied', 0, 1)
airlineQ3["tot"] = np.where(airlineQ3['tot'] == 'Business travel', 0, 1)
airline['Gender'] = np.where(airlineQ3['Gender'] == 'Male', 0, 1)
airlineQ3['ct'] = np.where(airlineQ3['ct'] == 'Loyal Customer', 0, 1)
airlineQ3['Class'] = airlineQ3['Class'].map(lambda x : 0 if x == 'Business' else 1 if x == 'Eco' else 2)
#%%
# heatmap
plt.figure(figsize=(20,15))
sns.heatmap(airlineQ3.corr(),annot=True,cmap='YlGnBu')
plt.title("correlation plot of all variables")
plt.tight_layout
# reindex
#airlineQ3.reset_index(drop=True, inplace=True)
#%%
#####################################################################
#
# How does the time affect the airline satisfaction?
#
#####################################################################


#####################################################################
# Anova of ddim + adim + datc
#####################################################################
#ax = sns.qqplo(x='ddim', data=airlineQ3)
#ax = sns.boxplot(x="adim", data=airlineQ3, color='#7d0013')
anovaData = airlineQ3[['adim','datc']]
CategoryGroupLists=anovaData.groupby('datc')['adim'].apply(list)
AnovaResults = st.f_oneway(*CategoryGroupLists)
print('P-Value for Anova is: ', AnovaResults[1])
print(AnovaResults)
#%%
sns.scatterplot(x="Departure Delay in Minutes", y="Arrival Delay in Minutes", hue='Satisfaction', data=airline)
plt.title("Scatterplot of Departure Delay in Minutes and Arrival Delay in Minutes")
plt.xlabel("Departure Delay in Minutes")
plt.ylabel("Arrival Delay in Minutes")
#%%
ax = sns.countplot(data=airline, x="Departure/Arrival time convenient", hue="Satisfaction")
for p in ax.patches:
        ax.annotate('{}'.format(p.get_height()), (p.get_x(), p.get_height()+60))
plt.title("Countplot of Departure/Arrival time convenient")

# %%
# This function is to show the result of confusion matrix
def showCrossTable(df, modelLogitFit, cut_off):
  modelPrediciton = pd.DataFrame(data= modelLogitFit.predict(df)) 
  # print(dfChkBasics(modelPrediciton))
  # Confusion matrix
  # Compute class predictions
  modelPrediciton[0] = np.where(modelPrediciton[0] > cut_off, 1, 0)
  crossTable = pd.crosstab(df['Satisfaction'], modelPrediciton[0],
  rownames=['Actual'], colnames=['Predicted'], margins = True)
  print(crossTable)
  TP = crossTable.iloc[1,1]
  TN = crossTable.iloc[0,0]
  Total = crossTable.iloc[2,2]
  FP = crossTable.iloc[0,1]
  FN = crossTable.iloc[1,0]
  print(f'Accuracy = (TP + TN) / Total = {(TP + TN) / Total}')
  print(f'Precision = TP / (TP + FP) = {TP / (TP + FP)}')
  print(f'Recall rate = TP / (TP + FN) = {TP / (TP + FN)}')
  print(f'Specificity = TN / (TN + FP) = {TN / (TN + FP)}')
  print(f'F1 score = TP / (TP + (FP + FN)/2) = {TP / (TP + (FP + FN)/2)}')
#%%
#####################################################################
# Logistic Regression model of Satisfaction ~ ddim + adim
#####################################################################
modelDelayLogitFitOrigin = glm(formula='Satisfaction ~ ddim + adim', data=airlineQ3, family=sm.families.Binomial()).fit()
print( modelDelayLogitFitOrigin.summary())
#%% [markdown]
# Since the p-value is extremely small, Departure Delay in Minutes and Arrival Delay in Minutes have strong relationship with Satisfaction
showCrossTable(airlineQ3,modelDelayLogitFitOrigin,0.4)
#%% [markdown]
# We can see this model is not good enough, the accuracy only have 0.46.\
# And from the correaltion plot, we can see that Departure Delay in Minutes and Arrival Delay in Minutes have high correlation, so I change them to Total Delay in Minutes (tdim)\
#%% 
#####################################################################
# Logistic Regression model of Satisfaction ~ tdim
#####################################################################
modelDelayTdimLogitFit = glm(formula='Satisfaction ~ tdim', data=airlineQ3, family=sm.families.Binomial()).fit()
print( modelDelayTdimLogitFit.summary())
showCrossTable(airlineQ3,modelDelayTdimLogitFit,0.4)
#%% [markdown]
# Let us add some time related variables to improve the model.
#%%
#####################################################################
# Logistic Regression model of Satisfaction ~ tdim + C(tot) + datc + C(Class)
#####################################################################
modelDelayLogitFit = glm(formula='Satisfaction ~ tdim + C(tot) + datc + C(Class)', data=airlineQ3, family=sm.families.Binomial()).fit()
print( modelDelayLogitFit.summary())
showCrossTable(airlineQ3,modelDelayLogitFit,0.5)
# showCrossTable(airlineQ3,modelDelayLogitFit,0.73)
#%% [marldown]
# In this model I add some time related variables such as Type of travel(tot), Departure/Arrival time convenient(datc), and Class.\
# We can see this model is much better. 
#%%
# sklearn method
xSatisfaction = airlineQ3[['tot', 'datc','tdim', 'Class']]
ySatisfaction = airlineQ3['Satisfaction']
#%%
# just see how the data looks like
sns.set()
sns.pairplot(xSatisfaction)
plt.show()
#%%
# scatter_matrix(xpizza, alpha = 0.2, figsize = (7, 7), diagonal = 'hist')
scatter_matrix(xSatisfaction, alpha = 0.2, figsize = (7, 7), diagonal = 'kde')
# plt.title("pandas scatter matrix plot")
plt.show()
#%%
# Plot the histogram thanks to the distplot function
sns.scatterplot(data=airlineQ3, x="adim", y="ddim", hue="tot")
plt.title("relation between departure delay in minutes (ddim) and arrival delay in minutes (adim)")
#%%
# split data set into train and test
X_train, X_test, y_train, y_test = train_test_split(xSatisfaction, ySatisfaction, random_state=1 )
satisfactionLogit = LogisticRegression()  # instantiate
satisfactionLogit.fit(X_train, y_train)
print('Logit model accuracy (with the test set):', satisfactionLogit.score(X_test, y_test))
print('Logit model accuracy (with the train set):', satisfactionLogit.score(X_train, y_train))
lr_cv_acc = cross_val_score(satisfactionLogit, xSatisfaction, ySatisfaction, cv=10, n_jobs = -1)
print(f'\nLogisticRegression CV accuracy score: {lr_cv_acc}\n')
#Confusion matrix in scikit-learn
y_pred = satisfactionLogit.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"r2_score: {r2_score(y_test, y_pred)}")
#%%
#####################################################################
# Receiver Operator Characteristics (ROC)
# Area Under the Curve (AUC)
#####################################################################
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = satisfactionLogit.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("AUC/ROC of Satisfaction ~ tdim + C(tot) + datc + C(Class)")
# show the legend
plt.legend()
# show the plot
plt.show()
#%% [markdown]
# From AUC/ROC = 0.812 we can see this model is not a bad model.
# %%
#####################################################################
# K-Nearest-Neighbor KNN 
#####################################################################
# number of neighbors
mrroger = 7
# KNN algorithm
knn = KNeighborsClassifier(n_neighbors=mrroger) # instantiate with n value given
knn.fit(xSatisfaction,ySatisfaction)
# y_pred = knn.predict(xSatisfaction)
# y_pred = knn.predict_proba(xSatisfaction)
print(f'knn use whole data set score: {knn.score(xSatisfaction,ySatisfaction)}')
##################################################
#%%
# 2-KNN algorithm
# The better way
# from sklearn.neighbors import KNeighborsClassifier
mrroger = 7
knn_split = KNeighborsClassifier(n_neighbors=mrroger) # instantiate with n value given
knn_split.fit(X_train,y_train)
print(f'knn train score:  {knn.score(X_train,y_train)}')
print(f'knn test score:  {knn.score(X_test,y_test)}')
print(confusion_matrix(y_test, knn.predict(X_test)))
print(classification_report(y_test, knn.predict(X_test)))
##################################################
#%%
# x-KNN algorithm
# The best way
def knnBest(num, knnDf):
  mrroger = num
  knn_cv = KNeighborsClassifier(n_neighbors=mrroger) # instantiate with n value given
  knn_cv.fit(X_train, y_train)
  cv_results = cross_val_score(knn_cv, xSatisfaction, ySatisfaction, cv=10, n_jobs = -1)
  # print(cv_results)
  knn_cv_mean_score = np.mean(cv_results)
  knn_cv_train_score = knn_cv.score(X_train,y_train)
  knn_cv_test_score = knn_cv.score(X_test,y_test)
  print(f'knn_cv mean score:  {knn_cv_mean_score}')
  print(f'knn_cv train score:  {knn_cv_train_score}')
  print(f'knn_cv test score:  {knn_cv_test_score}')
  print(confusion_matrix(y_test, knn_cv.predict(X_test)))
  print(classification_report(y_test, knn_cv.predict(X_test)))
  knnDf = knnDf.append({'knn_num':num,
                        'knn_cv_mean_score': knn_cv_mean_score, 
                        'knn_cv_train_score' : knn_cv_train_score,
                        'knn_cv_test_score' : knn_cv_test_score}, ignore_index=True)
  return knnDf
colName = ['knn_num','knn_cv_mean_score','knn_cv_train_score','knn_cv_test_score']
knnDf = pd.DataFrame(columns=colName)
# knnDf.set_index('knn_num', inplace=True)
for i in range(3,20):
  if i%2 == 1:
    knnDf = knnBest(i,knnDf)
print(knnDf)
#%%
plt.plot("knn_num","knn_cv_mean_score", data=knnDf, marker='o', label='knn_cv_mean_score')
plt.plot("knn_num",'knn_cv_train_score', data=knnDf, marker='o', label='knn_train_score')
plt.plot("knn_num",'knn_cv_test_score', data=knnDf, marker='o',  label='knn_test_score')
plt.title("Line plot of knn_cv")
plt.xlabel("KNN-K value")
plt.ylabel("accuracy")
plt.legend()
plt.show() 
##################################################
#%%
# 4-KNN algorithm
# Scale first? better or not?
# Re-do our darta with scale on X
xsSatisfaction = pd.DataFrame( scale(xSatisfaction), columns=xSatisfaction.columns )  
# Note that scale( ) coerce the object from pd.dataframe to np.array  
# Need to reconstruct the pandas df with column names
ysSatisfaction = ySatisfaction.copy()  # no need to scale y, but make a true copy / deep copy to be safe
#%%
knn_scv = KNeighborsClassifier(n_neighbors=9) # instantiate with n value given
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(xsSatisfaction, ySatisfaction, random_state=1 )
knn_scv.fit(X_train_s, y_train_s)
scv_results = cross_val_score(knn_scv, xsSatisfaction, ysSatisfaction, cv=5, n_jobs = -1)
print(scv_results) 
print(f'knn_cv mean score:  {np.mean(scv_results)}') 
print(f'knn_cv train score:  {knn_scv.score(X_train_s,y_train_s)}')
print(f'knn_cv test score:  {knn_scv.score(X_test_s,y_test_s)}')
print(confusion_matrix(y_test_s, knn_scv.predict(X_test_s)))
print(classification_report(y_test_s, knn_scv.predict(X_test_s)))
#%%
#####################################################################
# K-means 
#####################################################################
km_xSatisfaction = KMeans( n_clusters=2, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0 )
y_km = km_xSatisfaction.fit_predict(xSatisfaction)
# plot
# plot the 3 clusters
index1 = 1
index2 = 2
plt.scatter( xSatisfaction[y_km==0].iloc[:,index1], xSatisfaction[y_km==0].iloc[:,index2], s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1' )
plt.scatter( xSatisfaction[y_km==1].iloc[:,index1], xSatisfaction[y_km==1].iloc[:,index2], s=50, c='orange', marker='o', edgecolor='black', label='cluster 2' )
#plt.scatter( xSatisfaction[y_km==2].iloc[:,index1], xSatisfaction[y_km==2].iloc[:,index2], s=50, c='lightblue', marker='v', edgecolor='black', label='cluster 3' )
# plot the centroids
plt.scatter( km_xSatisfaction.cluster_centers_[:, index1], km_xSatisfaction.cluster_centers_[:, index2], s=250, marker='*', c='red', edgecolor='black', label='centroids' )
plt.legend(scatterpoints=1)
plt.xlabel(str(index1) + " : " + xSatisfaction.columns[index1])
plt.ylabel(str(index2) + " : " + xSatisfaction.columns[index2])
plt.grid()
plt.show()
#%%
# * SVC(): you can try adjusting the gamma level between 'auto', 'scale', 0.1, 5, etc, and see if it makes any difference 
# * SVC(kernel="linear"): having a linear kernel should be the same as the next one, but the different implementation usually gives different results 
# * LinearSVC() 
# * LogisticRegression()
# * KNeighborsClassifier(): you can try different k values and find a comfortable choice 
# * DecisionTreeClassifier(): try 'gini', 'entropy', and various max_depth  

#%% SVC
# svc = SVC()
svc = SVC()
svc.fit(X_train,y_train)
print(f'svc train score:  {svc.score(X_train,y_train)}')
print(f'svc test score:  {svc.score(X_test,y_test)}')
print(confusion_matrix(y_test, svc.predict(X_test)))
print(classification_report(y_test, svc.predict(X_test)))
# svc train score:  0.7439839365700458
# svc test score:  0.747714073891017
# [[13942  4303]
#  [ 3864 10263]]
#               precision    recall  f1-score   support

#            0       0.78      0.76      0.77     18245
#            1       0.70      0.73      0.72     14127

#     accuracy                           0.75     32372
#    macro avg       0.74      0.75      0.74     32372
# weighted avg       0.75      0.75      0.75     32372
# 58m 26.3s
#%% SVC kernel="linear"
svcKernelLinear = SVC(kernel="linear")
svcKernelLinear.fit(X_train, y_train)
print(f'svcKernelLinear train score:  {svcKernelLinear.score(X_train,y_train)}')
print(f'svcKernelLinear test score:  {svcKernelLinear.score(X_test,y_test)}')
print(confusion_matrix(y_test, svcKernelLinear.predict(X_test)))
print(classification_report(y_test, svcKernelLinear.predict(X_test)))
# svcKernelLinear train score:  0.7508829737939556
# svcKernelLinear test score:  0.7541393797108612
# [[13520  4725]
#  [ 3234 10893]]
#               precision    recall  f1-score   support

#            0       0.81      0.74      0.77     18245
#            1       0.70      0.77      0.73     14127

#     accuracy                           0.75     32372
#    macro avg       0.75      0.76      0.75     32372
# weighted avg       0.76      0.75      0.76     32372
# 30m 37.8s
#%% LinearSVC()
linearSVC = LinearSVC()
linearSVC.fit(X_train, y_train)
print(f'LinearSVC train score:  {linearSVC.score(X_train,y_train)}')
print(f'LinearSVC test score:  {linearSVC.score(X_test,y_test)}')
print(confusion_matrix(y_test, linearSVC.predict(X_test)))
print(classification_report(y_test, linearSVC.predict(X_test)))
# LinearSVC train score:  0.7593677598723163
# LinearSVC test score:  0.7622945755591252
# [[14124  4121]
#  [ 3574 10553]]
#               precision    recall  f1-score   support

#            0       0.80      0.77      0.79     18245
#            1       0.72      0.75      0.73     14127

#     accuracy                           0.76     32372
#    macro avg       0.76      0.76      0.76     32372
# weighted avg       0.76      0.76      0.76     32372
#10.3s
#%% LogisticRegression
# Apply logistic regression and print scores
lr = LogisticRegression()
lr.fit(X_train,y_train)
print(f'lr train score:  {lr.score(X_train,y_train)}')
print(f'lr test score:  {lr.score(X_test,y_test)}')
print(confusion_matrix(y_test, lr.predict(X_test)))
print(classification_report(y_test, lr.predict(X_test)))
# lr train score:  0.7656077845852854
# lr test score:  0.7687507722723341
# [[14152  4093]
#  [ 3393 10734]]
#               precision    recall  f1-score   support

#            0       0.81      0.78      0.79     18245
#            1       0.72      0.76      0.74     14127

#     accuracy                           0.77     32372
#    macro avg       0.77      0.77      0.77     32372
# weighted avg       0.77      0.77      0.77     32372
# 0.4s

#%%
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)
print(f'knn train score:  {knn.score(X_train,y_train)}')
print(f'knn test score:  {knn.score(X_test,y_test)}')
print(confusion_matrix(y_test, knn.predict(X_test)))
print(classification_report(y_test, knn.predict(X_test)))
# knn train score:  0.761591927096741
# knn test score:  0.7372111701470406
# [[14119  4126]
#  [ 4381  9746]]
#               precision    recall  f1-score   support

#            0       0.76      0.77      0.77     18245
#            1       0.70      0.69      0.70     14127

#     accuracy                           0.74     32372
#    macro avg       0.73      0.73      0.73     32372
# weighted avg       0.74      0.74      0.74     32372
# 22.6s

#%% DecisionTreeClassifier
# Instantiate dtree
dtree_digits = DecisionTreeClassifier(max_depth=6, criterion="entropy", random_state=1)
# Fit dt to the training set
dtree_digits.fit(X_train,y_train)
print(f'decisionTreeClassifier train score:  {dtree_digits.score(X_train,y_train)}')
print(f'decisionTreeClassifier test score:  {dtree_digits.score(X_test,y_test)}')
print(confusion_matrix(y_test, dtree_digits.predict(X_test)))
print(classification_report(y_test, dtree_digits.predict(X_test)))
# decisionTreeClassifier train score:  0.7740101940997786
# decisionTreeClassifier test score:  0.7645805016681082
# [[13739  4506]
#  [ 3115 11012]]
#               precision    recall  f1-score   support

#            0       0.82      0.75      0.78     18245
#            1       0.71      0.78      0.74     14127

#     accuracy                           0.76     32372
#    macro avg       0.76      0.77      0.76     32372
# weighted avg       0.77      0.76      0.77     32372
#0.2s

print("\nReady to continue.")
#%%
#####################################################################
# compare time (Too slow)
#####################################################################
def compareCountTimeInDifferentModel(model, compareTimeList):
    def countTime(model):
        global result
        model_cv_acc = cross_val_score(model, X_train, y_train, cv= 10, scoring='accuracy', n_jobs = -1)
        result = model_cv_acc
    meanTime = timeit.timeit(lambda: countTime(model), number = 7)/7
    # meanTime = timeit.timeit(lambda: countTime(model), number = 1)
    compareTimeList.append(meanTime)
    print(f"Execution time is: {meanTime}")
    print(f'\n{model} CV accuracy score: {result}\n')
    return compareTimeList
#%%
# modelList = [svc,svcKernelLinear,linearSVC,lr,knn,dtree_digits]
modelList = [linearSVC,lr,knn,dtree_digits]
compareTimeList =[]
#%% 
for i in modelList:
    compareTimeList = compareCountTimeInDifferentModel(i,compareTimeList)
# %%
# colName = ["svc","svcKernelLinear","linearSVC","lr","knn","dtree_digits"]
colName = ["linearSVC","lr","knn","dtree_digits"]
finalResult = pd.DataFrame([compareTimeList],columns=colName)
finalResult

#%%[markdown]
# ### **Q4:** Is there a difference in satisfaction for older passengers (50+) when comparing short and long distance flights?

#%%
# Prepare data frames for older passengers (We will define this as 50 and over)
oldairline = airline[airline["Age"] >= 50]; oldairline.reset_index(inplace=True)

# Rename columns of interest
oldairline.rename(
  {
  "Flight Distance": "fdist"
  },
  axis=1,
  inplace=True
)

oldairline["Satisfaction"] = np.where(oldairline["Satisfaction"] == "satisfied", 1, 0)
train["Satisfaction"] = np.where(train["Satisfaction"] == "satisfied", 1, 0)
test["Satisfaction"] = np.where(test["Satisfaction"] == "satisfied", 1, 0)

# %% [markdown]
# # Defining short, medium, and long flights
# According to [Wikipedia](https://en.wikipedia.org/wiki/Flight_length) ... there are generally three categories of commerical flight lengths:
# * short-haul: approx. 700 or less miles
# * medium-haul: approx. between 700 and 2400 miles
# * long-haul: approx. 2400 or more miles 

# %%
# Count plot, keeping above definition in mind
oldairline["fdist_group"]= pd.cut(oldairline.fdist, [0, 700, 2400, 5000], labels=["short-haul", "medium-haul", "long-haul"])
bydistance = oldairline.groupby(["fdist_group"]).Satisfaction.value_counts(normalize=True)

print(oldairline["fdist_group"].value_counts())

ax = bydistance.unstack().plot(kind="bar")
for p in ax.patches:
    ax.annotate(str(round(p.get_height(), 2)), (p.get_x() * 1.005, p.get_height() * 1.005))
plt.title("Satisfaction proportions by flight distance groups")

plt.ylabel("proportion")
plt.xticks(rotation=45)
plt.show()

# %%
# LOGISITC REGRESSION w/ statsmodel

# Describe and fit model
logitmodel_old = glm(formula="Satisfaction ~ fdist", data=oldairline, family=sm.families.Binomial()).fit()
print(logitmodel_old.summary())
# Predict
predictmodel_old = pd.DataFrame(columns=["logit_fdist"], data= logitmodel_old.predict(oldairline))
dfChkBasics(predictmodel_old)

# Confusion matrix
# Define cut off
cutoff = 0.5
# Model predictions
predictmodel_old["logit_fdist_result"] = np.where(predictmodel_old["logit_fdist"] > cutoff, 1, 0)
# Make a cross table
crosstable = pd.crosstab(oldairline["Satisfaction"], predictmodel_old["logit_fdist_result"],
rownames=["Actual"], colnames=["Predicted"], margins = True)

print(crosstable)
TP = crosstable.iloc[1,1]
TN = crosstable.iloc[0,0]
Total = crosstable.iloc[2,2]
FP = crosstable.iloc[0,1]
FN = crosstable.iloc[1,0]
print(f'Accuracy = (TP + TN) / Total = {(TP + TN) / Total}')
print(f'Precision = TP / (TP + FP) = {TP / (TP + FP)}')
print(f'Recall rate = TP / (TP + FN) = {TP / (TP + FN)}')

# %%
# LOGISITC REGRESSION w/ scikit-learn

# Prepare data, fit model
old_test = test[test["Age"] >= 50]; old_test.reset_index(inplace=True)
old_train = train[train["Age"] >= 50]; old_train.reset_index(inplace=True)

old_train_x = old_train[["Flight Distance", "Seat comfort", "Inflight entertainment"]]
old_train_y = old_train["Satisfaction"]
old_test_x = old_test[["Flight Distance", "Seat comfort", "Inflight entertainment"]]
old_test_y = old_test["Satisfaction"]

logitmodel2 = LogisticRegression()
logitmodel2.fit(old_train_x, old_train_y)

# Print score and classification report
print(f"Intercept: {logitmodel2.intercept_}")
for i, column in enumerate(["Flight Distance", "Seat comfort", "Inflight entertainment"]):
  print(f"Coefficient ({column}): {logitmodel2.coef_[0][i]}")
print("Logit model accuracy (with the test set):", logitmodel2.score(old_test_x, old_test_y))
print("Logit model accuracy (with the train set):", logitmodel2.score(old_train_x, old_train_y))

y_predict = logitmodel2.predict(old_test_x)
print(classification_report(old_test_y, y_predict))

# Receiver Operator Characteristics (ROC)
# Area Under the Curve (AUC)
from sklearn.metrics import roc_auc_score, roc_curve

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(old_test_y))]
# predict probabilities
lr_probs = logitmodel2.predict_proba(old_test_x)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(old_test_y, ns_probs)
lr_auc = roc_auc_score(old_test_y, lr_probs)
# summarize scores
print("No Skill: ROC AUC=%.3f" % (ns_auc))
print("Logistic: ROC AUC=%.3f" % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(old_test_y, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(old_test_y, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle="--", label="No Skill")
plt.plot(lr_fpr, lr_tpr, marker=".", label="Logistic")
# axis labels
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# show the legend
plt.legend()
# show the plot
plt.show()

#%%