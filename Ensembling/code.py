# --------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Code starts here
#loading the dataset
df = pd.read_csv(path)

#looking at the data
print(df.head(5))

#splitting data
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=4)

#initializing MinMaxScaler
scaler = MinMaxScaler()

#fitting and transforming
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Code ends here


# --------------
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

#initializinf Logistic Regression model
lr = LogisticRegression()

#fitting and predicting X_test
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

#calculating ROC AUC Score
roc_score = roc_auc_score(y_test, y_pred)
print('roc auc score: ',roc_score)


# --------------
from sklearn.tree import DecisionTreeClassifier

#initializing Decision Tree
dt = DecisionTreeClassifier(random_state=4)

#fitting
dt.fit(X_train,y_train)

#predicting
y_pred = dt.predict(X_test)

#calculating ROC AUC Score
roc_score = roc_auc_score(y_test,y_pred)
print('roc auc score: ',roc_score)





# --------------
from sklearn.ensemble import RandomForestClassifier


# Code strats here
#initializing Decision Tree
rfc = RandomForestClassifier(random_state=4)

#fitting
rfc.fit(X_train,y_train)

#predicting
y_pred = rfc.predict(X_test)

#calculating ROC AUC Score
roc_score = roc_auc_score(y_test,y_pred)
print('roc auc score: ',roc_score)


# Code ends here


# --------------
# Import Bagging Classifier
from sklearn.ensemble import BaggingClassifier


# Code starts here
#initializing bagging classifier
bagging_clf = BaggingClassifier(base_estimator= DecisionTreeClassifier(), n_estimators=100 , max_samples=100, random_state=0,)

#fitting
bagging_clf.fit(X_train,y_train)

#accuracy
score_bagging = bagging_clf.score(X_test,y_test)
print('accuracy: ',score_bagging)

# Code ends here


# --------------
# Import libraries
from sklearn.ensemble import VotingClassifier

# Various models
clf_1 = LogisticRegression()
clf_2 = DecisionTreeClassifier(random_state=4)
clf_3 = RandomForestClassifier(random_state=4)

model_list = [('lr',clf_1),('DT',clf_2),('RF',clf_3)]


# Code starts here

#initializing voting classifier
voting_clf_hard = VotingClassifier(estimators=model_list,voting='hard')

#fitting
voting_clf_hard.fit(X_train,y_train)

#accuracy
hard_voting_score = voting_clf_hard.score(X_test,y_test)
print('accuracy: ',hard_voting_score)

# Code ends here


