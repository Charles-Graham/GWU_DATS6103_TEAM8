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
#sns.heatmap(airlineQ3.loc[:, ~airlineQ3.columns.isin(['id','Age'])], annot=True)
#corr_matrix=airlineQ3.loc[:,['Gender','tot','ct']]
#sns.heatmap(airlineQ3.loc[:,['Gender','tot','ct']], annot=True)
# reindex
#airlineQ3.reset_index(drop=True, inplace=True)
#%%
#####################################################################
#
# How does the time affect the airline satisfaction?
#
#####################################################################
import statsmodels.api as sm 
from statsmodels.formula.api import glm
#%% 
#####################################################################
# Anova of ddim + adim + datc
#####################################################################
#ax = sns.qqplo(x='ddim', data=airlineQ3)
#ax = sns.boxplot(x="adim", data=airlineQ3, color='#7d0013')
from scipy.stats import f_oneway
anovaData = airlineQ3[['adim','datc']]
CategoryGroupLists=anovaData.groupby('datc')['adim'].apply(list)
AnovaResults = f_oneway(*CategoryGroupLists)
print('P-Value for Anova is: ', AnovaResults[1])
print(AnovaResults)
#%%
sns.scatterplot(x="Departure Delay in Minutes", y="Arrival Delay in Minutes", hue='Satisfaction', data=airline)
plt.title("Scatterplot of Departure Delay in Minutes and Arrival Delay in Minutes")
plt.xlabel("Departure Delay in Minutes")
plt.ylabel("Arrival Delay in Minutes")


#%%
sns.countplot(data=airline, x="Departure/Arrival time convenient", hue="Satisfaction")
plt.title("Countplot of Departure/Arrival time convenient")














#%%
#####################################################################
# Logistic Regression model of Satisfaction ~ ddim + adim
#####################################################################
modelDelayLogitFit = glm(formula='Satisfaction ~ ddim + adim + C(tot) + C(datc) + C(Class)', data=airlineQ3, family=sm.families.Binomial()).fit()
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

#%% [markdown]
# cut-off_0.4\
# Accuracy    = (TP + TN) / Total = (52231 + 7364) / 129487 = 0.4602392518167847\
# Precision   = TP / (TP + FP) = 52231 / (52231 + 65861) = 0.4422907563594486\
# Recall rate = TP / (TP + FN) = 52231 / (52231 + 4031) = 0.9283530624577868
#%%
#%% 
#####################################################################
# Logistic Regression model of Satisfaction ~ tdim
#####################################################################
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
#%% [markdown]
# cut-off_0.4\
# Accuracy    = (TP + TN) / Total = (52892 + 6047) / 129487 = 0.45517310618054324\
# Precision   = TP / (TP + FP) = 52892 / (52892 + 67178) = 0.4405097026734405\
# Recall rate = TP / (TP + FN) = 52892 / (52892 + 3370) = 0.9401016671998862

#%%
xSatisfaction = airlineQ3[['tot', 'datc','adim', 'ddim', 'Class']]
ySatisfaction = airlineQ3['Satisfaction']
# %%
sns.set()
sns.pairplot(xSatisfaction)
plt.show()
#%%
from pandas.plotting import scatter_matrix
# scatter_matrix(xpizza, alpha = 0.2, figsize = (7, 7), diagonal = 'hist')
scatter_matrix(xSatisfaction, alpha = 0.2, figsize = (7, 7), diagonal = 'kde')
# plt.title("pandas scatter matrix plot")
plt.show()
#%%
import seaborn as sns
# Plot the histogram thanks to the distplot function
sns.scatterplot(data=airlineQ3, x="adim", y="ddim", hue="tot")

#%%
from sklearn.model_selection import train_test_split
x_trainSatisfaction, x_testSatisfaction, y_trainSatisfaction, y_testSatisfaction = train_test_split(xSatisfaction, ySatisfaction, random_state=1 )
from sklearn.linear_model import LogisticRegression
satisfactionLogit = LogisticRegression()  # instantiate
satisfactionLogit.fit(x_trainSatisfaction, y_trainSatisfaction)
print('Logit model accuracy (with the test set):', satisfactionLogit.score(x_testSatisfaction, y_testSatisfaction))
print('Logit model accuracy (with the train set):', satisfactionLogit.score(x_trainSatisfaction, y_trainSatisfaction))
#Confusion matrix in scikit-learn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
y_pred = satisfactionLogit.predict(x_testSatisfaction)
print(confusion_matrix(y_testSatisfaction, y_pred))
print(classification_report(y_testSatisfaction, y_pred))
#%%
#####################################################################
# Receiver Operator Characteristics (ROC)
# Area Under the Curve (AUC)
#####################################################################
from sklearn.metrics import roc_auc_score, roc_curve

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_testSatisfaction))]
# predict probabilities
lr_probs = satisfactionLogit.predict_proba(x_testSatisfaction)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_testSatisfaction, ns_probs)
lr_auc = roc_auc_score(y_testSatisfaction, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_testSatisfaction, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_testSatisfaction, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC AUC of Satisfaction ~ ddim + adim + C(tot) + C(datc) + C(Class)")
# show the legend
plt.legend()
# show the plot
plt.show()

# %%
#####################################################################
# K-Nearest-Neighbor KNN 
#####################################################################
# number of neighbors
mrroger = 7
# KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=mrroger) # instantiate with n value given
knn.fit(xSatisfaction,ySatisfaction)
y_pred = knn.predict(xSatisfaction)
y_pred = knn.predict_proba(xSatisfaction)
print(y_pred)
print(knn.score(xSatisfaction,ySatisfaction))
##################################################
#%%
# 2-KNN algorithm
# The better way
# from sklearn.neighbors import KNeighborsClassifier
mrroger = 4
knn_split = KNeighborsClassifier(n_neighbors=mrroger) # instantiate with n value given
knn_split.fit(x_trainSatisfaction,y_trainSatisfaction)
ytest_pred = knn_split.predict(x_testSatisfaction)
ytest_pred
print(knn_split.score(x_testSatisfaction,y_testSatisfaction))
##################################################
#%%
# 3-KNN algorithm
# The best way
mrroger = 3
from sklearn.neighbors import KNeighborsClassifier
knn_cv = KNeighborsClassifier(n_neighbors=mrroger) # instantiate with n value given

from sklearn.model_selection import cross_val_score
cv_results = cross_val_score(knn_cv, xSatisfaction, ySatisfaction, cv=10)
print(cv_results) 
print(np.mean(cv_results)) 
##################################################
#%%
# 4-KNN algorithm
# Scale first? better or not?
# Re-do our darta with scale on X
from sklearn.preprocessing import scale
xsSatisfaction = pd.DataFrame( scale(xSatisfaction), columns=xSatisfaction.columns )  # reminder: xadmit = dfadmit[['gre', 'gpa', 'rank']]
# Note that scale( ) coerce the object from pd.dataframe to np.array  
# Need to reconstruct the pandas df with column names
# xsadmit.rank = xadmit.rank
ysSatisfaction = ySatisfaction.copy()  # no need to scale y, but make a true copy / deep copy to be safe
#%%
# from sklearn.neighbors import KNeighborsClassifier
knn_scv = KNeighborsClassifier(n_neighbors=mrroger) # instantiate with n value given
# from sklearn.model_selection import cross_val_score
scv_results = cross_val_score(knn_scv, xsSatisfaction, ysSatisfaction, cv=5)
print(scv_results) 
print(np.mean(scv_results)) 
#%%
#####################################################################
# K-means 
#####################################################################
from sklearn.cluster import KMeans
km_xSatisfaction = KMeans( n_clusters=2, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0 )
y_km = km_xSatisfaction.fit_predict(xSatisfaction)
# plot
# plot the 3 clusters
index1 = 0
index2 = 1

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
# %%
