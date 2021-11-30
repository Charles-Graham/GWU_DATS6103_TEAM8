# To add a new cell, type "#%%"
# To add a new markdown cell, type "#%% [markdown]"

#%% [markdown]

#%%

### Preprocess
#Combine original train and test datasets

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)

airline = pd.concat([train,test])


#Check any existing null values
for i in airline:
  print(i + " has " + str(airline[i].isnull().sum()) + " nulls")

#Found 310 nulls in arrival delay in minutes column
#Drop the NAs
airline = airline[airline["Arrival Delay in Minutes"].isnull() == False]

#df without NAs
#Since the number of records is 103594 while NAs only 310, so not a big concern if dropping them
#Add new column of Total Delay Minutes, and switch the column order with satisfaction

def dfChkBasics(dframe, valCnt = False): 
  cnt = 1
  print("\ndataframe Basic Check function -")
  
  try:
    print(f"\n{cnt}: info(): ")
    cnt+=1
    print(dframe.info())
  except: pass

  print(f"\n{cnt}: describe(): ")
  cnt+=1
  print(dframe.describe())

  print(f"\n{cnt}: dtypes: ")
  cnt+=1
  print(dframe.dtypes)

  try:
    print(f"\n{cnt}: columns: ")
    cnt+=1
    print(dframe.columns)
  except: pass

  print(f"\n{cnt}: head() -- ")
  cnt+=1
  print(dframe.head())

  print(f"\n{cnt}: shape: ")
  cnt+=1
  print(dframe.shape)

  if (valCnt):
    print("\nValue Counts for each feature -")
    for colname in dframe.columns :
      print(f"\n{cnt}: {colname} value_counts(): ")
      print(dframe[colname].value_counts())
      cnt +=1


airline["Total Delay in Minutes"] = airline["Departure Delay in Minutes"] + airline["Arrival Delay in Minutes"]
temp = airline["satisfaction"]
airline = airline.drop(columns = ["satisfaction"])
airline["Satisfaction"] = temp

# airline.to_csv("airline.csv")

# %%
# Charles ~ 4) Is there a difference in satisfaction for older passengers (50+) when comparing short and long distance flights?

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
train["satisfaction"] = np.where(train["satisfaction"] == "satisfied", 1, 0)
test["satisfaction"] = np.where(test["satisfaction"] == "satisfied", 1, 0)

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
import statsmodels.api as sm 
from statsmodels.formula.api import glm

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# LOGISITC REGRESSION w/ scikit-learn

# Prepare data, fit model
old_test = test[test["Age"] >= 50]; old_test.reset_index(inplace=True)
old_train = train[train["Age"] >= 50]; old_train.reset_index(inplace=True)

old_train_x = old_train[["Flight Distance", "Seat comfort", "Inflight entertainment"]]
old_train_y = old_train["satisfaction"]
old_test_x = old_test[["Flight Distance", "Seat comfort", "Inflight entertainment"]]
old_test_y = old_test["satisfaction"]

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

# %%
