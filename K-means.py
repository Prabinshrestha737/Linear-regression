import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import copy

'''First step: read the data from the frame or file, Divide the data into the cluster k and initialize 
random centroids to each cluster from the given data. 
Assign color to the centroids colmap.
'''
df = pd.read_csv('Comp1801CourseworkData.csv')
salary = df['Age']
salary.values.tolist()

y = df['Salary']

'''First step: read the data from the frame or file, Divide the data into the cluster k and initialize 
random centroids to each cluster from the given data. 
Assign color to the centroids colmap.
'''

df = pd.DataFrame({
    'x': salary,
    'y': y,

})

np.random.seed(10)
k =2
#centroids[i] = [x, y]

centroids = {
    i+1: [np.random.randint(0, 20), np.random.randint(0, 100)]
    for i in range(k)
}

colmap = {1: 'r', 2:'g'}
review = {1: 'Below', 2: 'Above'}


#Assignment state 

''' Calculate the distance from each points to the cluter's centroid and take the mean value. 
Reapeat the step until the mean value becomes stable.'''

def assignment(df, centroids):
    
    for i in centroids.keys():
        #sqrt((x1-x2)^2-(y1-y2)^2)
        df['distance_from_{}'.format(i)] = (
        np.sqrt(
            (df['x']  - centroids[i][0]) ** 2 
            + (df['y'] - centroids[i][1]) ** 2
            )
        )

        
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    df['Perform'] = df['closest'].map(lambda x: review[x])
    
    return df

df = assignment(df, centroids)


#update_stage

old_centroids = copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
        
    return k 

centroids = update(centroids)
        

## Repeat Assignment stage 

df = assignment(df, centroids)


while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    
    
    if closest_centroids.equals(df['closest']):
        break

print(df)      

#plot the final result.

fig = plt.figure(figsize=(2, 2))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=1, edgecolor='k')

for i in centroids.keys():
    a = plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 2)
plt.ylim(0, 2)
plt.show()