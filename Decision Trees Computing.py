# -*- coding: utf-8 -*-
"""
Data Analytics Computing Follow Along: Decision Trees for Regression and Classification
Spyder version 5.1.5
"""

# Import the required packages
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# We will perform regression and classification with decision trees this time
# Decision trees are rather straightforward, as we use them as mental models all the time
# pip install pandas
# pip install sklearn
# pip install seaborn

# The lines below import the dataset that we will use in this script
# The link in kaggle is in the description, this is a salaries dataset

# We will be using pandas.read_csv to import our data

income_df = pd.read_csv('C:/Videos/archive/salaries_clean.csv')

# We can get the shape of the dataframe with the shape method

income_df.shape

# The print function just shows in the console the text that you give it    
    
print("Dataset imported")

# Get relevant columns

income_df_sub = income_df[['employer_name', 'job_title', 'job_title_category',
                           'total_experience_years', 'annual_base_pay', 
                           'employer_experience_years']]

# We can see the null values below

income_df_sub.isna().sum()

# We need to drop the null values for sklearn

income_df_clean = income_df_sub.dropna()

# For the classification problem, a High Income variable will be created
# We will say that if the salary is more than 120000, then it will be high income

income_df_clean['High Income?'] = pd.cut(income_df_clean['annual_base_pay'], 
                                         bins = [-1, 120000, float('Inf')], 
                                         labels = [0, 1])

# See the columns and the first rows of the dataset

print(income_df_clean.columns)

print(income_df_clean.head())

# Use total_experience_years as our predictor
# High Income? will be our prediction for classification
# We want to know if the number of years with an employer can be used to predict if a position is high income

X = income_df_clean[['total_experience_years']]

y = income_df_clean['annual_base_pay']

# Show a scatterplot to see the points first

plt.scatter(X, y, color = "Blue", s = 10)

max(y)

# We will first remove the outlier

income_df_clean = income_df_clean[income_df_clean['annual_base_pay'] < 2000000]

# Repeat the previous steps after EDA 

X = income_df_clean[['total_experience_years']]

# We also need to change Y as some of its values may have been dropped
y = income_df_clean['annual_base_pay']

plt.scatter(X, y, color = "Blue", s = 10)
plt.xlabel("Total Years of Experience")
plt.ylabel("Annual Base Pay")
plt.title("Scatterplot on Annual Base Pay vs Total Years of Experience")

# First import the tree regressor

tree_reg = tree.DecisionTreeRegressor()
tree_reg.fit(X, y)
tree_regre_predictions = tree_reg.predict(X)

# Plot the three
tree.plot_tree(tree_reg)

# We can see that the graph is not really very insightful

tree_reg_2 = tree.DecisionTreeRegressor(criterion='squared_error', max_depth = 1)
tree_reg_2.fit(X, y)
tree_regre_predictions = tree_reg_2.predict(X)

# Plot the three
# Let's take a second to see the structure of the tree and how it works
tree.plot_tree(tree_reg_2, feature_names = X.columns,
               filled = True)

# Let's try to understand the tree before moving onto more complicated trees
# First, let's get the mean of each of the nodes
print(income_df_clean['annual_base_pay'].mean())
print(income_df_clean['annual_base_pay'].count())

# The first "box" describes the dataset before the split, we can use the split value
print(y[income_df_clean['total_experience_years'] < 6.75].mean())
print(y[income_df_clean['total_experience_years'] < 6.75].count())

# We can do the same with the other box
print(y[income_df_clean['total_experience_years'] > 6.75].mean())
print(y[income_df_clean['total_experience_years'] > 6.75].count())

# More formally, the boxes are known as the nodes of the tree
# Similarly to actual trees, if a node does not have additional nodes coming from it, it is a leaf
# Notice the max_depth parameter, now we can use a more robust approach
tree_reg_depth = tree.DecisionTreeRegressor(criterion='squared_error', max_depth = 2)
tree_reg_depth.fit(X, y)
# Plot the tree
tree.plot_tree(tree_reg_depth, feature_names = X.columns,
               filled = True)

# Notice that this is low resolution, we can make it legible 
# This is changing a parameter of the environment, the dots per inch of plots generated
plt.rcParams['figure.dpi'] = 600

# Instead of one split on our model, there are three of them
# We can also get the feature importance, although for now it will not be that important until we consider more variables
tree_reg_depth.feature_importances_

# Now, let's take advantage of the decison tree methodology to expand our X
# Until now, we have only used one X, now we can use multiple of them instead
# The model will now be trained both on the total years of experience and the years of experience with the employer
# In Linear Regression, the equivalent of doing this is Multiple Linear Regression

X_multi = income_df_clean[['total_experience_years', 'employer_experience_years']]

# Check X_multi in comparison to X
X.shape
X_multi.shape

# Fit a new model that uses more than one variable as the predictors
tree_reg_multi = tree.DecisionTreeRegressor(criterion='squared_error', max_depth = 2)
tree_reg_multi.fit(X_multi, y)

# Now let's plot the new tree
tree.plot_tree(tree_reg_multi, feature_names = X_multi.columns,
               filled = True)

# The tree has improved somewhat, we can refine it by allowing it to be more complex
complex_tree = tree.DecisionTreeRegressor(criterion='squared_error', max_depth = 3)
complex_tree.fit(X_multi, y)
tree.plot_tree(complex_tree, feature_names = X_multi.columns,
               filled = True)

# Notice how the plot is too small again, we can use the fontsize this time
plt.figure(figsize=(27,10))
tree.plot_tree(complex_tree, feature_names = X_multi.columns,
               filled = True, fontsize = 10)

# The tree starts to become too complex to just plot it, review the feature importance
complex_tree.feature_importances_

# We can have a more visual plot of the feature importances
feature_importances = pd.Series(complex_tree.feature_importances_, 
                                index=X_multi.columns)

# We can plot the importances
feature_importances.plot(kind='barh', color = ['Red', 'Blue'])

# These lines of code are for aesthetic purposes as usual
plt.ylabel('Variables')
plt.xlabel('Total Importance')
plt.title('Variable importance for tree model')

# From linear regression, we know we can predict a value outside of the original range
# Can we do the same with a decision tree?
max(X_multi['total_experience_years'])
max(X_multi['employer_experience_years'])

# Use the structure of the X_multi dataset

new_obs = X_multi[0:0]

# There are more elegant approaches to do this, but this works for demonstration purposes
new_obs.loc[0] = [40, 40]

new_obs.loc[1] = [50, 50]

new_obs.loc[2] = [60, 60]

new_obs.loc[3] = [70, 70]

new_obs.loc[4] = [80, 80]

complex_tree.predict(new_obs)

# Compare this to the highest predictions from the model

complex_predictions = complex_tree.predict(new_obs)

max(complex_predictions)

# The rule of the tree will always put these observations on the leaf with the appropiate rule on it

new_obs.loc[5] = [6.70, 0.3]

# See how we can follow the decisions of the three through the nodes
complex_predictions = complex_tree.predict(new_obs)

# The decision tree classifier has a very similar methodology

class_tree = tree.DecisionTreeClassifier(max_depth = 3)

# We will just use X_multi again
y_class = income_df_clean['High Income?']
class_tree.fit(X_multi, y_class)
plt.figure(figsize=(27,10))
tree.plot_tree(class_tree, feature_names = X_multi.columns,
               filled = True, fontsize = 10)
# The only difference is that instead of squared error, we have the gini index
# By default, gini is a measure of purity
# Just like as classification, we can get the variable importance
feature_importances_class = pd.Series(class_tree.feature_importances_, 
                                index=X_multi.columns)
feature_importances_class.plot(kind='barh', color = ['Red', 'Blue'])

# We can get the depth of the three in case we have forgotten about it
class_tree.get_depth()

# The number of leaves is also pretty useful
class_tree.get_n_leaves()

# Decision trees are non-parametric models, but we can still retrieve their rules
class_tree.get_params(deep = True)

# And just as with Logistic Regression, we can get a probability instead of a prediction
class_predictions = class_tree.predict(X_multi)
class_predictions_proba = class_tree.predict_proba(X_multi)

# Remember that the second one is the probability of being high income
class_predictions_proba[:, 1]

# We can get the score of the model
class_tree.score(X_multi, y_class)

# Compare with the simpler tree with multiple predictors
tree_reg_multi.score(X_multi, y)
complex_tree.score(X_multi, y)

# And finally, the classic Confusion Matrix
# 964 observations were true positives, 156 were true negatives
confusion_matrix(y_class, class_predictions)

# And the classification report

print(classification_report(y_class, class_predictions))

# We can also get a text version of the tree with export_text

class_tree_text = tree.export_text(class_tree)

# Decision trees are pretty useful, although they do not perform very well by themselves
# They are easy to understand, and can be improved 
# More advanced methods such as bagged trees, boosted trees, and the popular random forest algorithm

# Decision trees also have different regression plots than simple linear regression

# Predict values again
y_d = tree_reg_multi.predict(X_multi)
y_d_2 = complex_tree.predict(X_multi)

# Plot the results
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_multi['total_experience_years'], tree_regre_predictions, color="red", label="max_depth of 1", linewidth = 0.2)
plt.plot(X_multi['total_experience_years'], y_d, color="blue", label="max_depth of 2", linewidth = 0.2)
plt.plot(X_multi['total_experience_years'], y_d_2, color="green", label="max_depth of 3", linewidth = 0.2)
plt.xlabel("Total Years of Experience")
plt.ylabel("High Income?")
plt.title("Decision Tree Regression for Total Years of Experience")
plt.legend()
plt.xlim([0, 35])
plt.ylim([0, 220000])
plt.show()

# It does not look pretty for sure, but the most complex one is the green one
# Both the number of features used and the depth of the tree increase the complexity
# A decision boundary plot for decision tree classifiers is outside the scope of this video