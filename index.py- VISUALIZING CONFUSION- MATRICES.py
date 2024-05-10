#!/usr/bin/env python
# coding: utf-8

# # Visualizing Confusion Matrices - Lab
# 
# ## Introduction
# 
# In this lab, you'll build upon the previous lesson on confusion matrices and visualize a confusion matrix using `matplotlib`. 
# 
# ## Objectives
# 
# In this lab you will:  
# 
# - Create a confusion matrix from scratch 
# - Create a confusion matrix using scikit-learn 
# - Craft functions that visualize confusion matrices 
# 
# ## Confusion matrices
# 
# Recall that the confusion matrix represents the counts (or normalized counts) of our True Positives, False Positives, True Negatives, and False Negatives. This can further be visualized when analyzing the effectiveness of our classification algorithm.   
#   
# Here's an example of how a confusion matrix is displayed:
# <img src="https://curriculum-content.s3.amazonaws.com/data-science/images/new_confusion_matrix_2.png" width="350">

# With that, let's look at some code for generating this kind of visual.

# ## Create our model
# As usual, we start by fitting a model to data by importing, normalizing, splitting into train and test sets and then calling your chosen algorithm. All you need to do is run the following cell. The code should be familiar to you. 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('heart.csv')

# Define appropriate X and y
X = df[df.columns[:-1]]
y = df.target

# Split the data into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Normalize the data
X_train = X_train.copy()
X_test = X_test.copy()

for col in X_train.columns:
    X_train[col] = (X_train[col] - min(X_train[col]))/ (max(X_train[col]) - min(X_train[col]))

for col in X_test.columns:
    X_test[col] = (X_test[col] - min(X_test[col]))/ (max(X_test[col]) - min(X_test[col]))    

# Fit a model
logreg = LogisticRegression(fit_intercept=False, C=1e12, solver='liblinear')
model_log = logreg.fit(X_train, y_train)

# Preview model params
print(model_log) 

# Predict
y_hat_test = logreg.predict(X_test)

print("")
# Data preview
df.head()


# ## Create the confusion matrix
# 
# To gain a better understanding of confusion matrices, complete the `conf_matrix()` function in the cell below.  This function should:
# 
# * Take in two arguments: 
#     * `y_true`, an array of labels
#     * `y_pred`, an array of model predictions
# * Return a confusion matrix in the form of a dictionary, where the keys are `'TP', 'TN', 'FP', 'FN'`  

# In[2]:


def conf_matrix_df(y_true, y_pred):
    TP = ((y_true == 1) & (y_pred == 1)).sum()
    TN = ((y_true == 0) & (y_pred == 0)).sum()
    FP = ((y_true == 0) & (y_pred == 1)).sum()
    FN = ((y_true == 1) & (y_pred == 0)).sum()

    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}

# Predict
y_hat_test = logreg.predict(X_test)

# Compute the confusion matrix
conf_matrix_result = conf_matrix_df(y_test, y_hat_test)
print(conf_matrix_result)


# ## Check your work with `sklearn`
# 
# To check your work, make use of the `confusion_matrix()` function found in `sklearn.metrics` and make sure that `sklearn`'s results match up with your own from above.
# 
# - Import the `confusion_matrix()` function
# - Use it to create a confusion matrix for `y_test` versus `y_hat_test`, as above 

# In[3]:


# Import confusion_matrix

from sklearn.metrics import confusion_matrix
# Print confusion matrix
cnf_matrix = confusion_matrix(y_test, y_hat_test)
print('Confusion Matrix:\n', cnf_matrix)


# ## Create a nice visual
# 
# Luckily, sklearn recently implemented a `ConfusionMatrixDisplay` function that you can use to create a nice visual of your confusion matrices. 
# 
# [Check out the documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html), then visualize the confusion matrix from your logistic regression model on your test data.

# In[4]:


# Import plot_confusion_matrix
from sklearn.metrics import plot_confusion_matrix


# In[5]:


# Visualize your confusion matrix
plot_confusion_matrix(model_log, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


# ## Summary
# 
# Well done! In this lab, you created a confusion matrix from scratch, then explored how to use a new function to visualize confusion matrices nicely!
