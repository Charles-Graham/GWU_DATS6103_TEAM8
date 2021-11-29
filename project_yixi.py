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
# Logistic Regression model of Satisfaction ~ ddim + adim + C(tot) + C(datc) + C(Class)
#####################################################################
import statsmodels.api as sm 
from statsmodels.formula.api import glm
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
crossTable = pd.crosstab(airlineQ3['Satisfaction'], modelPredicitonOfDelay['logit_ddimAdim_result'],
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
crossTable = pd.crosstab(airlineQ3['Satisfaction'], modelPredicitonOfDelay['logit_tdim_result'],
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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
#%%
#####################################################################
# Receiver Operator Characteristics (ROC)
# Area Under the Curve (AUC)
#####################################################################
from sklearn.metrics import roc_auc_score, roc_curve

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
plt.title("AUC/ROC of Satisfaction ~ ddim + adim + C(tot) + C(datc) + C(Class)")
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
# 7-KNN algorithm
# The best way
def knnBest(num, knnDf):
  mrroger = num
  from sklearn.neighbors import KNeighborsClassifier
  knn_cv = KNeighborsClassifier(n_neighbors=mrroger) # instantiate with n value given
  knn_cv.fit(X_train, y_train)
  from sklearn.model_selection import cross_val_score
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
  knnDf = knnBest(i,knnDf)
  

print(knnDf) 
#%%
# knnDf.set_index('index', inplace=True)
#%%
plt.plot("knn_num","knn_cv_mean_score", data=knnDf, marker='o', label='knn_cv_mean_score')
plt.plot("knn_num",'knn_cv_train_score', data=knnDf, marker='o', label='knn_cv_train_score')
plt.plot("knn_num",'knn_cv_test_score', data=knnDf, marker='o',  label='knn_cv_test_score')
plt.title("Line plot of knn_cv")
plt.xlabel("KNN-K value")
plt.ylabel("score")
plt.legend()
plt.show() 
  
  

##################################################
#%%
# 4-KNN algorithm
# Scale first? better or not?
# Re-do our darta with scale on X
from sklearn.preprocessing import scale
xsSatisfaction = pd.DataFrame( scale(xSatisfaction), columns=xSatisfaction.columns )  
# Note that scale( ) coerce the object from pd.dataframe to np.array  
# Need to reconstruct the pandas df with column names
ysSatisfaction = ySatisfaction.copy()  # no need to scale y, but make a true copy / deep copy to be safe
#%%
knn_scv = KNeighborsClassifier(n_neighbors=mrroger) # instantiate with n value given
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
from sklearn.cluster import KMeans
km_xSatisfaction = KMeans( n_clusters=2, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0 )
y_km = km_xSatisfaction.fit_predict(xSatisfaction)
# plot
# plot the 3 clusters
index1 = 2
index2 = 3

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

#%%
#####################################################################
# Receiver Operator Characteristics (ROC)
# Area Under the Curve (AUC)
#####################################################################
def rocAuc(model):
  # generate a no skill prediction (majority class)
  ns_probs = [0 for _ in range(len(y_test))]
  # predict probabilities
  lr_probs = model.predict_proba(X_test)
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
  plt.title("ROC AUC of Satisfaction ~ ddim + adim + C(tot) + C(datc) + C(Class)")
  # show the legend
  plt.legend()
  # show the plot
  plt.show()


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
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
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print(f'knn train score:  {knn.score(X_train,y_train)}')
print(f'knn test score:  {knn.score(X_test,y_test)}')
print(confusion_matrix(y_test, knn.predict(X_test)))
print(classification_report(y_test, knn.predict(X_test)))
# knn train score:  0.7563095299387325
# knn test score:  0.7051155319411837
# [[13917  4328]
#  [ 5218  8909]]
#               precision    recall  f1-score   support

#            0       0.73      0.76      0.74     18245
#            1       0.67      0.63      0.65     14127

#     accuracy                           0.71     32372
#    macro avg       0.70      0.70      0.70     32372
# weighted avg       0.70      0.71      0.70     32372
# 22.6s

#%% DecisionTreeClassifier
# Instantiate dtree
dtree_digits = DecisionTreeClassifier(max_depth=12, criterion="entropy", random_state=1)
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
import timeit
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
modelList = [lr,knn,dtree_digits]
compareTimeList =[]
#%% 
for i in modelList:
    compareTimeList = compareCountTimeInDifferentModel(i,compareTimeList)
# %%
# colName = ["svc","svcKernelLinear","linearSVC","lr","knn","dtree_digits"]
colName = ["lr","knn","dtree_digits"]
finalResult = pd.DataFrame([compareTimeList],columns=colName)
finalResult
#%%