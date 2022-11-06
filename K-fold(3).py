from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

train = pd.read_csv('./dataset/train.csv')
test = pd.read_csv('./dataset/test.csv')
# print(type(train))
# 提取训练集数据的特征（目标标签的特征）
X_train = train.drop('Survived', axis=1)
# 提取目标标签数据
y_train = train['Survived']
#创建一个管道（Pipeline）实例，里面包含标准化方法和随机森林模型
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=4))
# 创建一个用于得到不同训练集和测试集样本的索引的StratifiedKFold实例，折数为10
strtfdKFold = StratifiedKFold(n_splits=10)
#把特征和标签传递给StratifiedKFold实例
kfold = strtfdKFold.split(X_train, y_train)
#循环迭代，（K-1）份用于训练，1份用于验证，把每次模型的性能记录下来。
scores = []
for k, (train, test) in enumerate(kfold):
    pipeline.fit(X_train.iloc[train, :], y_train.iloc[train])
    score = pipeline.score(X_train.iloc[test, :], y_train.iloc[test])
    scores.append(score)
    print('Fold: %2d, Training/Test Split Distribution: %s, Accuracy: %.3f' % (k+1, np.bincount(y_train.iloc[train]), score))
print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))