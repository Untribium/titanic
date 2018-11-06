import numpy as np
import pandas as pd
import math

# load data sets
train = pd.read_csv('./data/train.csv')
print('train:', train.shape)
test = pd.read_csv('./data/test.csv')
print('test: ', test.shape)

# combine sets into one (split later by Survived=NaN)
data = pd.concat([train, test], sort=True)
print('data: ', data.shape)

# remove rows with Embarked = NaN (lol, python)
data = data[~data['Embarked'].isnull()]

# print columns
print(data.columns.values)

# print first sample
print(data.iloc[0])

# extract title (format is LastName, Title FirstName MiddleName (MaidenName))
data['Title'] = [name.split(',')[1].split(' ')[1] for name in data['Name']]
data['NameLen'] = [len(name.replace(title, '')) for name, title in zip(data['Name'], data['Title'])]

# convert sex to category
# get list of unique values, then replace text with index of value in list
sexes = sorted(list(set(data['Sex'])))
data['Sex'] = [sexes.index(row) for row in data['Sex']]

# analogous
embarkeds = sorted(list(set(data['Embarked'])))
data['Embarked'] = [embarkeds.index(row) for row in data['Embarked']]

# analogous
titles = sorted(list(set(data['Title'])))
data['Title'] = [titles.index(row) for row in data['Title']]

# replace NaN age with mean
mean_age = data['Age'].mean()
data['Age'] = data['Age'].replace(float('nan'), mean_age)

# drop incomplete/unused features
data = data.drop(labels=['Cabin', 'Ticket', 'Name'], axis=1)

print(data.iloc[0])

# write data to features folder
data[~data['Survived'].isnull()].to_csv('./features/train.csv')
data[ data['Survived'].isnull()].to_csv('./features/test.csv')
