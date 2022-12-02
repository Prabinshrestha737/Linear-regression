# importing modules and packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing

# importing data
df = pd.read_csv('Comp1801CourseworkData.csv')

# print(df.head())

sns.scatterplot(x='Age',
                y='Salary', data=df)


# creating feature variables
X = df.drop('Age',axis= 1)
y = df['Salary']
print(X)
print(y)

# creating train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

model = LinearRegression()
model.fit(X_train,y_train)
# making predictions
predictions = model.predict(X_test)
# model evaluation
print('mean_squared_error : ', mean_squared_error(y_test, predictions))
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))