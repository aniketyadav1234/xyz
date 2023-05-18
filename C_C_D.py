import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import logisticRegression 
from sklearn.metrics import accuracy_score

# loading the dataset to a pandas dataframe
credit_card_data = pd.read_csv('/content/creditcard.csv')

# firt 5  rows of the database
credit_card_data.head() #-->show us first 5 rows

credit_card_data.tall() #-->show us last 5 rows

# dataset informations
credit_card_data.info()

# checking the number of missing values in each column
credit_card_data.isnull.sum()

# distribution of legit transactions & fraudulent transections
credit_card_data['Class'].value_counts()
'''
this database is highly unbalanced

0----> normal transection

1--->fraudulent trasection

'''

# seperating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)

# statistical measures of the data
legit.amount.discribe()

fraud.amount.discribe()

# compare the values for both transections
credit_card_data.groupby('Class').mean()
'''
dealing with unbalaced data

'''

'''
under-sampling

build a sample dataset containing similar distribution of normal transactions fraudelent transactions

number of fraudelent transactions---> 492

'''
legit_sample = legit.sample(n=492)

'''
concatenating two dataframes

'''

new_dataset = pd.concat([legit_sample,fraud],axis=0)
'''
axis = 0 means it will add in rows 
axis = 1 means it will add in columns

'''
new_dataset.head()

new_dataset.tall()

new_dataset['Class'].value_counts()

new_dataset.groupby('Class').mean()
'''
splitting the data info features & targets
'''
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

print(X)
print(Y)
'''
split the data into training data & testing data
'''
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2) 

print(X.shape, X_train.shape, X_test.shape)
'''
logistic regression
'''
model = logisticRegression()
# training the logistic Regression model with Training data
model.fit(X_train, Y_train)

'''
model evaluation:

accuracy score
'''
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training Data : ', training_data_accuracy)
'''
Accuracy on Training Data :  0.9364675984752223
'''
# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
     

print('Accuracy score on Test Data : ', test_data_accuracy)
'''
Accuracy score on Test Data :  0.9289340101522843
'''