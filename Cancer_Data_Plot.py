#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[1]:


seed=42
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from imblearn.over_sampling import SMOTE 
np.random.seed(seed)


# 
# # Load data

# In[2]:


X, y = load_breast_cancer(return_X_y=True)
print('Original dataset shape {}'.format(Counter(y)))
#Samples per class	212(Malignant),357(Benign)


# # Preprocessing
# ## Spilt the Dataset

# In[3]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)


# ## Oversampling

# In[4]:


sm = SMOTE(ratio=1, random_state=seed, kind='borderline1')
x_train, y_train = sm.fit_sample(x_train, y_train)
x_test, y_test = sm.fit_sample(x_test, y_test)


# # Reduce Dimension

# In[5]:


pca = PCA(n_components=2, random_state=seed)
pca.fit(x_train, y_train)
x_train = pca.transform(x_train)
print ('After PCA', x_train.shape)
x_test = pca.transform(x_test)


# # Plot Graph

# In[6]:


plt.plot()
colors = ['red', 'green']
target_names=['Tumor Sample', 'Normal Sample']
lw = 0.5

for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(x_train[y_train == i, 0], x_train[y_train == i, 1], color=color, alpha=.5, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title("Cancer Class Distribution After PCA", fontsize='small')
plt.savefig('class_distribution_after_pca.eps', format='eps', dpi=2500)
plt.show()


# In[ ]:




