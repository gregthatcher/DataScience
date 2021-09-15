'''
Cleaning up Titanic Data
Ideas from : https://app.pluralsight.com/course-player?clipId=e4f0cd91-8578-432b-a8e8-96a96695f29d
'''

import pandas as pd
from sklearn import preprocessing

titanic_data = pd.read_csv("data/raw_data/Titanic/train.csv", quotechar='"')

print(titanic_data.columns)
print(titanic_data.head())

# Drop columns we don't need
# Note that I've seen clever stuff done on "Name"
# in the past, including extracting suffixes and prefixes
# in the names (e.g. "Reverend", "Mrs.", etc.)
titanic_data.drop(['PassengerId', 'Name', 'Ticket',
                  'Cabin'], 'columns', inplace=True)


print(titanic_data.info())
# SibSp -> number of siblings
# Parch -> number of Parents
print(titanic_data.columns)
print(titanic_data.head())

print("Types:", titanic_data.dtypes)
le = preprocessing.LabelEncoder()
# Do we _really_ need .astype(str)?
# This converts "male"/"female" to 1/0
titanic_data["Sex"] = le.fit_transform(titanic_data["Sex"].astype(str))

print("Types:", titanic_data.dtypes)
# SibSp -> number of siblings
# Parch -> number of Parents
print(titanic_data.columns)
print(titanic_data.head())

# Survived 1 - yes, 0 - 
# Embarked ->
#   C - Cherbourg
#   Q - Queenstown
#   S = Southampton
titanic_data = pd.get_dummies(titanic_data, columns=["Embarked"])
print("Types:", titanic_data.dtypes)
# SibSp -> number of siblings
# Parch -> number of Parents
print(titanic_data.columns)
#print(titanic_data.head())

# Let's check if there are any nulls
print("How many rows have nulls:")
print(titanic_data[titanic_data.isnull().any(axis=1)].shape[0])
# print(titanic_data.isnull())
print("How many rows are there:")
print(titanic_data.shape[0])

titanic_data = titanic_data.dropna()
print("How many rows after dropna:")
print(titanic_data.shape[0])


titanic_data.to_csv("data/preprocessed_data/titanic_train.csv", index=False)
