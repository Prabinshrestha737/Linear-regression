# Import necessary modules
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np 


df = pd.read_csv('Comp1801CourseworkData.csv')
df.head()

df['Salary'] = np.where( df['Salary']> 35000, 1, 0)

df['Sex'] = df['Sex'].astype('category')
df['Sex'] = df['Sex'].cat.codes

df['Education'] = df['Education'].astype('category')
df['Education'] = df['Education'].cat.codes

df['WorkType'] = df['WorkType'].astype('category')
df['WorkType'] = df['WorkType'].cat.codes

df['Region'] = df['Region'].astype('category')
df['Region'] = df['Region'].cat.codes

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.1)
X = train.iloc[:,:8]
y = train["Salary"]

# Create an instance of the MLPClassifier
clf = MLPClassifier()

# Train the model on the data
clf.fit(X, y)

# Use the trained model to make predictions
predictions = clf.predict(X)
