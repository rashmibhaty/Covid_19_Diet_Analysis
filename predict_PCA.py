# PCA

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = pd.read_csv('Protein_Supply_Quantity_Data.csv')
dataset=dataset.replace("<2.5", 0)

dataset = dataset.dropna()
dataset = dataset.reset_index(drop=True)

X_ori = dataset.iloc[:, 1:25].values
y_ori = dataset.iloc[:, 26].values

column_names=dataset.columns[1:25]

X=X_ori
y=y_ori

X = StandardScaler().fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train_ori = X_train
X_test_ori = X_test


# Applying PCA
from sklearn.decomposition import PCA
for i in range (10,1,-1):
    pca = PCA(n_components = i)
    X_train = pca.fit_transform(X_train_ori)
    X_test = pca.transform(X_test_ori)
    explained_variance = pca.explained_variance_ratio_
    print(explained_variance)
    n_pcs= pca.components_.shape[0]
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    most_important_names = [column_names[most_important[i]] for i in range(n_pcs)]
    print(most_important)
    print(most_important_names)
    print("------")
    

X_new = X_ori[:,most_important[0]]
y_new = y_ori

axes = plt.gca()
axes.spines["bottom"].set_color("purple")
axes.spines["left"].set_color("purple")
axes.tick_params(axis='x', colors='purple')
axes.tick_params(axis='y', colors='purple')

for i in range (0,X.shape[0]):
    plt.scatter(X_new[i], y_new[i], s = 100, c = 'red')
    if i%2 == 1:
        plt.annotate(dataset.iloc[i, 0], (X_new[i], y_new[i]), fontsize=16,rotation=-45,va='top')
    else:
        plt.annotate(dataset.iloc[i, 0], (X_new[i], y_new[i]), fontsize=16,rotation=+45,va='bottom')

plt.title(column_names[most_important[0]] +' Consumed vs Confirmed cases',fontsize=20, fontweight='bold',c = 'purple')
plt.xlabel(column_names[most_important[0]] + ' Consumed Percentage',fontsize=16, fontweight='bold',c = 'purple')
plt.ylabel('Confirmed Cases Percentage',fontsize=16, fontweight='bold',c = 'purple')
#plt.show()

figure = plt.gcf()  # get current figure
figure.set_size_inches(32, 18) # set figure's size manually to your full screen (32x18)
plt.savefig(column_names[most_important[0]] +".png", bbox_inches='tight') # bbox_inches removes extra white spaces
plt.clf()