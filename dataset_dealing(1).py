import json

import pandas as pd

from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('./dataset/titanic.csv')
sex_le = LabelEncoder()
embarked_le = LabelEncoder()
df.dropna(inplace=True)
'''
将sex和embarked属性进行标签编码
sex={'male': 0 , 'female': 10}
Embarked={'S': 0, 'C': 1, 'Q': 2}

'''
df['Sex'] = sex_le.fit_transform(df['Sex'].values)
df['Embarked'] = sex_le.fit_transform(df['Embarked'].values)
# df.Sex = pd.get_dummies(df.Sex, prefix="Sex")
# df.Sex = pd.get_dummies(df.Embarked, prefix="Embarked")
df.to_csv('./dataset/titanicout.csv')

