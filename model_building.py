# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 14:14:47 2021

@author: HB6
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

df = pd.read_csv('eda_data.csv')

# choose relevant columns
df.columns
df_model = df[['avg_salary', 'Rating', 'Size','Type of ownership','Industry', 'Sector', 
               'Revenue','num_comp','hourly', 'employer_provided','job_state', 
               'same_state', 'age', 'python_yn', 'spark', 'aws', 'excel', 
               'job_simp', 'seniority', 'desc_len']]

# get dummy data (when we have categorical variables we need to make dummy data (or variables))
# example : we have job simplified where the values are in one of the categories (data sc/ data analist/ mle/manager/director)
# >> make for each category a new column where the value = 1 if the job has that attribute (else 0)
df_dum = pd.get_dummies(df_model)

# train test split
from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary', axis=1)
y = df_dum['avg_salary'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# multiple linear regression
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()
# interpretation of results : 
# R-squared = 0.708 >> the model explains 70% of variation in data sc salaries (r-squared : statistical measure of how close the data are to the fitted regression line)
# P>|t| >> we are interested in the values < 0.05 (significant to our model) : num_com >> coeff = 2.2503 >> for each additional competitor, we add around 2000$ to the salary

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))
# neg_mean_absolute_error = shows how far from avg we are off of the general pred
# -20.766855128917243>> we are off of about 20K$

# lasso regression (bc the dataset will be larger with the dummy vars>> normalize that with lasso reg)
# lm_l = Lasso() 
lm_l = Lasso(alpha=.13) # replaced "lm_l = Lasso()" (after finding the best alpha)
lm_l.fit(X_train,y_train) # added afer finding the best alpha
np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))
    
plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]
# >> an alpha=0.13 returns the least error>> error = -19. 

# random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3))
# better results >> error = -15

# tune models GridsearchCV 
# GridsearchCV >> you put  all the params >> runs all the models and returns the one with the best results
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}
# number of scenarios : 30(nb_est)*2(criterion)*3(features)
gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(X_train,y_train)

gs.best_score_
gs.best_estimator_

# test ensembles
# use these diffrent models to predict the test/set data and see if we have the same results

tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)
# compare them to the y_test

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_lm)
mean_absolute_error(y_test,tpred_lml)
mean_absolute_error(y_test,tpred_rf)
# random forest is the best

# try to combine the best 2 (lm&rf) and see if the results improve 
mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2)
# Nope >>> we can improve it by running it through a regression model >> get weights >> exp 90% of rf and 10%of lm would give better results

# productionize
import pickle
pickl = {'model': gs.best_estimator_}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

model.predict(X_test.iloc[1,:].values.reshape(1,-1))
# model.predict(np.array(list(X_test.iloc[1,:])).reshape(1,-1))[0]

# sample data to test the model on (FLASK step)
list(X_test.iloc[1,:])

