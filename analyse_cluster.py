# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:48:47 2020

@author: rashmibh
"""

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = pd.read_csv('Fat_Supply_Quantity_Data.csv',usecols=[0,24,26])

dataset=dataset.replace("<2.5", 0)

dataset = dataset.dropna()
dataset = dataset.reset_index(drop=True)

X = dataset.iloc[:, [1,2]].values

scalerX = StandardScaler().fit(X)
X_scaled = scalerX.transform(X)

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
#plt.show()
figure = plt.gcf()  # get current figure
figure.set_size_inches(16, 9) # set figure's size manually to your full screen (32x18)
plt.savefig("Elbow.png", bbox_inches='tight') # bbox_inches removes extra white spaces

plt.clf()


num_opt_clusters=3

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = num_opt_clusters, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X_scaled)


original_len=dataset.shape[0]
for i in range(0,original_len):
    dataset.loc[i,"Cluster"]=y_kmeans[i]
 
dataset.to_csv('Cluster Assignment Result.csv') 


plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.title('Cluster prediction')
plt.xlabel('Obesity')
plt.ylabel('Confirmed')
plt.show()

axes = plt.gca()
axes.set_xlim([min(X[:,0])-.05,max(X[:,0])+.05])
axes.set_ylim([min(X[:,1])-.005,max(X[:,1])+.005])
axes.spines["bottom"].set_color("purple")
axes.spines["left"].set_color("purple")
axes.tick_params(axis='x', colors='purple')
axes.tick_params(axis='y', colors='purple')
    
for i in range (0,X.shape[0]):
    if y_kmeans[i] == 1 :
        plt.scatter(X[i,0], X[i,1], s = 100, c = 'green')
    elif y_kmeans[i] == 2 :
        plt.scatter(X[i,0], X[i,1], s = 100, c = 'red')
    elif y_kmeans[i] == 0 :
        plt.scatter(X[i,0], X[i,1], s = 100, c = 'blue')
   # plt.annotate(dataset.iloc[i, 0], (X[i,0], X[i,1]))
    if i%2 == 1:
        plt.annotate(dataset.iloc[i, 0], (X[i,0], X[i,1]), fontsize=14,rotation=35,va='bottom')
    else:
        plt.annotate(dataset.iloc[i, 0], (X[i,0], X[i,1]), fontsize=14,rotation=-35,va='top')

plt.title('Cluster Analysis',fontsize=20, fontweight='bold',c = 'purple')
plt.xlabel('Obesity Percentage',fontsize=16, fontweight='bold',c = 'purple')
plt.ylabel('Confirmed Cases Percentage',fontsize=16, fontweight='bold',c = 'purple')
#plt.show()

figure = plt.gcf()  # get current figure
figure.set_size_inches(32, 18) # set figure's size manually to your full screen (32x18)
plt.savefig("Cluster.png", bbox_inches='tight') # bbox_inches removes extra white spaces
plt.clf()


for j in range(0,num_opt_clusters):
    if j==1:
        colour = 'green'
    elif j==2:
        colour = 'red'
    elif j==0:
        colour = 'blue'
    else:
        print("Error:")
    
    axes = plt.gca()
    axes.set_xlim([min(X[y_kmeans==j,0])-.05,max(X[y_kmeans==j,0])+.05])
    axes.set_ylim([min(X[y_kmeans==j,1])-.005,max(X[y_kmeans==j,1])+.005])
    axes.spines["bottom"].set_color("purple")
    axes.spines["left"].set_color("purple")
    axes.tick_params(axis='x', colors='purple')
    axes.tick_params(axis='y', colors='purple')

    for i in range (0,X.shape[0]):
        if y_kmeans[i] == j :
            plt.scatter(X[i,0], X[i,1], s = 150, c = colour)
            if i%2 == 1:
                plt.annotate(dataset.iloc[i, 0], (X[i,0], X[i,1]), fontsize=14,rotation=-45,va='top')
            else:
                plt.annotate(dataset.iloc[i, 0], (X[i,0], X[i,1]), fontsize=14,rotation=+45,va='bottom')
        
    plt.title('Cluster_' + str(j) + ' Detail',fontsize=20, fontweight='bold',c = 'purple')
    plt.xlabel('Obesity Percentage',fontsize=16, fontweight='bold',c = 'purple')
    plt.ylabel('Confirmed Cases percentage',fontsize=16, fontweight='bold',c = 'purple')
    #plt.show()
    
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(32, 18) # set figure's size manually to your full screen (32x18)
    plt.savefig("Cluster"+str(j)+".png", bbox_inches='tight') # bbox_inches removes extra white spaces
    plt.clf()

