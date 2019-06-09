# --------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# path- variable storing file path

#Code starts here

df = pd.read_csv(path)
df.head(5)

X = df.drop('Price',axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)

corr = X_train.corr()
print('correlation between features stored in X_train: ',corr)


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Code starts here
# instantiate and fit model
regressor = LinearRegression().fit(X_train, y_train)

# make predictions and calculate r2

y_pred = regressor.predict(X_test)

r2 = r2_score(y_test, y_pred)
print('r2 score: ',r2)


# --------------
from sklearn.linear_model import Lasso

# Code starts here

# instantiate lasso model
lasso = Lasso()

# fit and predict and calculate r2
lasso.fit(X_train,y_train)
lasso_pred = lasso.predict(X_test)

r2_lasso = r2_score(y_test,lasso_pred)
print('r2 score by lasso: ', r2_lasso)



# --------------
from sklearn.linear_model import Ridge

# Code starts here
# instantiate lasso model
ridge= Ridge()

# fit and predict
ridge.fit(X_train,y_train)
ridge_pred = ridge.predict(X_test)

# calculate R2
r2_ridge = r2_score(y_test, ridge_pred)
print('r2 score: ',r2_ridge)

# Code ends here


# --------------
from sklearn.model_selection import cross_val_score
#Code starts here

#instantiate model
regressor = LinearRegression()

#calculating cross_val_score 
score = cross_val_score(regressor,X_train,y_train,cv=10)

#calculating mean of score
mean_score = np.mean(score)
print('mean score: ',mean_score)



# --------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Code starts here

#initialize a pipeline for polynomial features
model = make_pipeline(PolynomialFeatures(2),LinearRegression())

#fit model
model.fit(X_train,y_train)

#make prediction
y_pred = model.predict(X_test)

#calculate r2 score
r2_poly = r2_score(y_test,y_pred)
print('r2 score by polynimial regressor: ',r2_poly)



