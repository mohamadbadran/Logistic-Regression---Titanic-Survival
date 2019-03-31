#STEP #0 : LIBRARIES IMPORT

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#STEP #1 : IMPORT THE DATASET

# read the dataset using pandas dataframe
Dataset = pd.read_csv('Train_Titanic.csv')

#STEP #2 : EXPLORE/VISUALIZE THE DATASET
# Let's count the number of survivors and non-survivors
survived = Dataset [ Dataset ['Survived']==1]
not_survived = Dataset [Dataset ['Survived']==0]

# Count the survived and deceased 
print('Total = ', len(Dataset))
print('number of passengers who survived', len(survived))
print('number of passengers who did not survive', len(not_survived))
print('% Survived', 1.* len(survived)/len(Dataset) *100)
print('% Did not Survive', 1.* len(not_survived)/len(Dataset) *100)

# Bar Chart to indicate the number of people survived based on their class
# If you are a first class, you have a higher chance of survival
plt.figure(figsize = [2, 7])
plt.subplot(211)
sns.countplot(x ='Pclass', data = Dataset)
plt.subplot(212)
sns.countplot(x ='Pclass', hue = 'Survived', data = Dataset)

# Bar Chart to indicate the number of people survived based on their siblings status
# If you have 1 siblings (SibSp = 1), you have a higher chance of survival compared to being alone (SibSp = 0)
plt.figure(figsize = [2, 7])
plt.subplot(211)
sns.countplot(x ='SibSp', data = Dataset)
plt.subplot(212)
sns.countplot(x ='SibSp', hue = 'Survived', data = Dataset)

# Bar Chart to indicate the number of people survived based on their Parch status (how many parents onboard)
# If you have 1, 2, or 3 family members (Parch = 1,2), you have a higher chance of survival compared to being alone (Parch = 0)
plt.figure(figsize = [2, 7])
plt.subplot(211)
sns.countplot(x ='Parch', data = Dataset)
plt.subplot(212)
sns.countplot(x ='Parch', hue = 'Survived', data = Dataset)

# Bar Chart to indicate the number of people survived based on the port they embarked from
# Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
# If you embarked from port "C", you have a higher chance of survival compared to other ports!
plt.figure(figsize = [2, 7])
plt.subplot(211)
sns.countplot(x ='Embarked', data = Dataset)
plt.subplot(212)
sns.countplot(x ='Embarked', hue = 'Survived', data = Dataset)

# Bar Chart to indicate the number of people survived based on their sex
# If you are a female, you have a higher chance of survival compared to other ports!
plt.figure(figsize = [2, 7])
plt.subplot(211)
sns.countplot(x ='Sex', data = Dataset)
plt.subplot(212)
sns.countplot(x ='Sex', hue = 'Survived', data = Dataset)

# Bar Chart to indicate the number of people survived based on their age
# If you are a baby, you have a higher chance of survival!
plt.figure(figsize = [25,20])
sns.countplot(x = 'Age', hue = 'Survived', data = Dataset)

# Age Histogram 
Dataset['Age'].hist(bins = 40)

# Bar Chart to indicate the number of people survived based on their fare
# If you pay a higher fare, you have a higher chance of survival
plt.figure(figsize = [25,8])
sns.countplot(x = 'Fare', hue = 'Survived', data = Dataset)

#Fare Histogram
Dataset['Fare'].hist(bins = 40)

# STEP#3 : PREPARE THE DATA FOR TRAINING/DATA CLEANING

# Let's explore which dataset is missing
sns.heatmap(Dataset.isnull(), yticklabels = False, cbar = False , cmap = 'Blues')

# Let's drop the 'Name', 'Ticket', 'Embarked', 'Cabin','PassengerId' coloumns
Dataset.drop(['Name', 'Ticket', 'Embarked', 'Cabin','PassengerId'], axis = 1, inplace = True)

# Let's view the data one more time!
sns.heatmap(Dataset.isnull(), yticklabels = False , cbar = False , cmap = 'Blues')

# Let's get the average age for male (~29) and female (~25)
plt.figure(figsize = (10 , 7))  
sns.boxplot(x = 'Sex', y = 'Age', data = Dataset)

# Function to return our missing data 'NaN'
def Fill_Age(data) :
    sex = data[0]
    age = data[1]
    
    if pd.isnull(age) :
      if sex is 'male' :
            return 29
      else :
        return 25
    else :
        return age 
    
# Let's apply our function on our missing data 'NaN'
Dataset['Age'] = Dataset[['Sex', 'Age'] ].apply(Fill_Age , axis = 1)

# Let's view the data one more time!
sns.heatmap(Dataset.isnull(), yticklabels = False , cbar = False , cmap = 'Blues')

# Age histogram after filling the missing data
Dataset['Age'].hist(bins = 40)

# Encoding The Categorical Data (Sex)
# Avoiding Dummy Variable Trap
male = pd.get_dummies(Dataset['Sex'], drop_first = True)

# Let's drop the sex column
Dataset.drop(['Sex'], axis = 1, inplace = True)

# Now let's add the encoded column male again
Dataset = pd.concat([Dataset , male], axis = 1)

#Let's drop the target coloumn 'Survived' before we do train test split
X = Dataset.drop('Survived', axis = 1).values
y = Dataset['Survived'].values

#STEP #4 : MODEL TRAINING

# Let's import our model for train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#STEP #5 : MODEL TESTING

# Predicting the Test set results
y_predict = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot= True, fmt = 'd')

# Displaying our model results
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))
