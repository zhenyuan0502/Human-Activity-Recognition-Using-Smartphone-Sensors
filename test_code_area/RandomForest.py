import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns
import pandas_profiling as prof

from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/samsungData.txt', sep='|')
randomState = 42
ntree = 25

train = data.sample(frac=0.7,
                    random_state=randomState)

test = data[~data.index.isin(train.index)]

# Base RF - with all variables
def varPlot(X, model, plotSize=(15, 6), xticks=False):
    model_vars = pd.DataFrame({
        'variable': X.columns,
        'importance': model.feature_importances_})

    model_vars.sort_values(by='importance',
                           ascending=False,
                           inplace=True)

    plt.figure(figsize=plotSize)
    sns.set_style('whitegrid')

    sns.barplot(y='importance',
                x='variable',
                data=model_vars,
                palette=sns.color_palette("Blues_r", n_colors=X.shape[1]))

    if xticks == True:
        plt.xticks([])
    else:
        plt.xticks(rotation=90)

    plt.xlabel('Variable')
    plt.ylabel('Variable Importance')
    plt.title('Random Forest : Averaged variable importance over ' +
              str(ntree) + ' trees')
    plt.show()
    return 0


# Build model containing all 561 variable
X = train[train.columns[:-2]]
Y = train.activity
randomState = 42

model0 = rfc(n_estimators=ntree,
             random_state=randomState,
             n_jobs=4,
             warm_start=True,
             oob_score=True)
model0 = model0.fit(X, Y)
model0.oob_score_

# Variable importance:all 561 variables
varPlot(X=X, model=model0, plotSize=(10, 4), xticks=True)

# Sorted dataframe for variable & their importance score
model_vars0 = pd.DataFrame({'variable': X.columns,
                            'importance': model0.feature_importances_})

model_vars0.sort_values(by='importance',
                        ascending=False,
                        inplace=True)

# top n variables
n = 25
cols_model = [col for col in model_vars0.variable[:n].values]

# Create OOB accuracy table for using top 1 to n variables at a time
oobAccuracy = {}

for cols in range(n):
    X = train[[col for col in model_vars0['variable'][:cols + 1].values]]
    Y = train.activity

    model1 = rfc(n_estimators=ntree,
                 random_state=randomState,
                 n_jobs=4,
                 warm_start=False,
                 oob_score=True)

    model1 = model1.fit(X, Y)
    accuracy = accuracy_score(Y, model1.predict(X))

    oobAccuracy[cols + 1] = [cols + 1, model1.oob_score_, accuracy]

accuracyTable = pd.DataFrame.from_dict(oobAccuracy).transpose()
accuracyTable.columns = ['variables', 'oobAccuracy', 'accuracy']


# Plot OOB accuracy vs number of variables
plt.figure(figsize=(10, 5))
plt.scatter(x=accuracyTable.variables,
                y=100 * accuracyTable.oobAccuracy)

plt.xlim(0, n + 1)
plt.ylim(0, 100)
plt.minorticks_on()

plt.xlabel('Number of variables')
plt.ylabel('Accuracy(%)')
plt.title('OOB Accuracy vs Number of variables')
plt.show()


# Variable importance plot for top n variables
varPlot(X, model1)

# Selected variables (5)from top 15
n_used = 4
cols_model = [col for col in model_vars0.variable[:n_used].values] + \
    [model_vars0.variable[6]]

X = train[cols_model]
Y = train.activity


# For the selected variables, we determine optimum number of trees
# Create a loop for 5 to 150 trees with steps of 5
# Fit a model during each iteration
# Store OOB score for each iteration

ntree_determination = {}

for ntree in range(5, 150, 5):
    model = rfc(n_estimators=ntree,
                random_state=randomState,
                n_jobs=4,
                warm_start=False,
                oob_score=True)
    model = model.fit(X, Y)
    ntree_determination[ntree] = model.oob_score_

ntree_determination = pd.DataFrame.from_dict(
    ntree_determination, orient='index')
ntree_determination['ntree'] = ntree_determination.index
ntree_determination.columns = ['oobScore', 'ntree']

# Plot number of trees vs OOB accuracy
plt.figure(figsize=(6, 4), )

plt.scatter(x='ntree',
                y='oobScore',
                s=35, c='red', alpha=0.65,
                data=ntree_determination)
plt.xlabel('Number of trees used')
plt.ylabel('OOB Accuracy')
plt.show()


# For our final iteration, we choose 50 trees & 5 variables
model2 = rfc(n_estimators=50,
             random_state=randomState,
             n_jobs=4,
             warm_start=False,
             oob_score=True)
model2 = model2.fit(X, Y)

# Variable importance plot for the final model
varPlot(X, model2, plotSize=(6, 4))

# Accuracy metrics for the final model
train_actual = Y
train_pred = model2.predict(X)

# Confusion matrix for training data
confusion_matrix(train_actual, train_pred)

sns.heatmap(data=confusion_matrix(train_actual, train_pred),
            fmt='.0f',
            annot=True,
            xticklabels=np.unique(train_actual),
            yticklabels=np.unique(train_actual))

# Training accuracy
accuracy_score(train_actual, train_pred)

# OOB accuracy
model2.oob_score_

# Test set results
test_actual = test.activity
test_pred = model2.predict(test[X.columns])

# Confusion matrix for test data
confusion_matrix(test_actual,test_pred)
sns.heatmap(data=confusion_matrix(test_actual,test_pred),
            fmt='.0f',
            annot=True,
            xticklabels=np.unique(test_actual),
            yticklabels=np.unique(test_actual))

# Accuracy on test data
accuracy_score(test_actual,test_pred)

# Distribution of final variables for each category
plt.figure(figsize=(6,6))
sns.set_style('whitegrid')

sns.pairplot(data=train[[col for col in X.columns]+['activity']],
             hue='activity',
             palette='Set2',
             diag_kind='kde')