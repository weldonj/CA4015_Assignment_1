#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from scipy.spatial.distance import cdist


# In[4]:


from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


df_vpp = pd.read_csv("./data/parameters_igt_vpp.csv")


# In[8]:


df_vpp.describe()


# In[15]:


df_pvl_delta = pd.read_csv("./data/parameter_igt_pvl_delta.csv")


# In[16]:


df_pvl_delta.head()


# In[17]:


df_pvl_delta.drop(columns=['SubID'], inplace=True)

df_clustering = df_pvl_delta.drop(columns=['group'])
df_clustering.head()


# In[29]:


# Preprocessing the data to make it visualizable 
  
# Scaling the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clustering)

# Normalizing the Data
X_normalized = normalize(X_scaled)
  
# Converting the numpy array into a pandas DataFrame 
X_normalized = pd.DataFrame(X_normalized) 
  
# Reducing the dimensions of the data 
pca = PCA(n_components = 2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']

X_principal.head(2)


# In[45]:


# List of Silhouette Scores for rbf method
s_scores_rbf = []

# Finding the optimal number of cluster for the rbf basis appraoch
for k in range(2, 11):
    
    # Building the clustering model 
    spectral_model_rbf = SpectralClustering(n_clusters = k, affinity ='rbf')

    # Training the model and Storing the predicted cluster labels
    labels_rbf = spectral_model_rbf.fit_predict(X_principal)
    
    # Evaluating the performance 
    s_scores_rbf.append(silhouette_score(df_clustering, labels_rbf))

print(s_scores_rbf)

plt.plot(range(2,11), s_scores_rbf, 'bx-')
plt.xlabel('# Clusters')
plt.ylabel('Silhouette Scores')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.show()


# In[38]:


# Visualizing the clustering 
plt.scatter(X_principal['P1'], X_principal['P2'],  
           c = SpectralClustering(n_clusters = 3, affinity ='rbf').fit_predict(X_principal), cmap =plt.cm.winter) 
plt.show()


# In[24]:


# Building the clustering model 
spectral_model_nn = SpectralClustering(n_clusters = 2, affinity ='nearest_neighbors') 
  
# Training the model and Storing the predicted cluster labels 
labels_nn = spectral_model_nn.fit_predict(X_principal)


# In[42]:


# List of Silhouette Scores for nn method
s_scores_nn= []

# Finding the optimal number of cluster for the rbf basis appraoch
for k in range(2, 11):
    
    # Building the clustering model 
    spectral_model_nn = SpectralClustering(n_clusters = k, affinity ='nearest_neighbors')

    # Training the model and Storing the predicted cluster labels
    labels_nn = spectral_model_rbf.fit_predict(X_principal)
    
    # Evaluating the performance
    s_scores_nn.append(silhouette_score(df_clustering, labels_nn))

print(s_scores_nn)

plt.plot(range(2,11), s_scores_nn, 'bx-')
plt.xlabel('# Clusters')
plt.ylabel('Silhouette Scores')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.show()


# In[27]:


# List of different values of affinity 
affinity = ['rbf', 'nearest-neighbours'] 
  
# List of Silhouette Scores 
s_scores = [] 
  
# Evaluating the performance 
s_scores.append(silhouette_score(df_clustering, labels_rbf)) 
s_scores.append(silhouette_score(df_clustering, labels_nn)) 

# Plotting a Bar Graph to compare the models 
plt.bar(affinity, s_scores) 
plt.xlabel('Affinity') 
plt.ylabel('Silhouette Score') 
plt.title('Comparison of different Clustering Models')
plt.show()

print(s_scores)


# In[ ]:




