import pandas as pd 
df = pd.read_csv('Comp1801CourseworkData.csv')


df['Sex'] = df['Sex'].astype('category')
df['Sex'] = df['Sex'].cat.codes

df['Education'] = df['Education'].astype('category')
df['Education'] = df['Education'].cat.codes

df['WorkType'] = df['WorkType'].astype('category')
df['WorkType'] = df['WorkType'].cat.codes

df['Region'] = df['Region'].astype('category')
df['Region'] = df['Region'].cat.codes


X = df.drop(columns='Salary')

y = df['Salary']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = (train_test_split(
    X, y, test_size=0.3, random_state=0))

print(y_train)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr)

c = lr.intercept_
print(c)

m = lr.coef_
print(m)

y_pred_train = lr.predict(X_train)
print(y_pred_train)

import matplotlib.pyplot as plt
plt.scatter(y_train, y_pred_train)
plt.xlabel("Actual charges")
plt.ylabel("Predicted charges")
plt.show()

from sklearn.metrics import r2_score

r2_score(y_train, y_pred_train)

print(r2_score(y_train, y_pred_train))