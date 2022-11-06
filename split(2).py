from sklearn.model_selection import train_test_split
import pandas as pd

def train_test_val_split(data, ratio_train, ratio_test, ratio_val):
    train, middle = train_test_split(data, train_size=ratio_train, test_size=ratio_test + ratio_val)
    ratio = ratio_val/(1-ratio_train)
    test, validation = train_test_split(middle, test_size=ratio)
    return train, test, validation

if __name__ == '__main__':
    data = pd.read_csv('./dataset/titanicout.csv')
    train, test, validation = train_test_val_split(data, 0.6, 0.2, 0.2)
    train.to_csv('./dataset/train.csv', index=False)  # 数据存入csv,存储位置及文件名称
    test.to_csv('./dataset/test.csv', index=False)  # 数据存入csv,存储位置及文件名称
    validation.to_csv('./dataset/validation.csv', index=False)  # 数据存入csv,存储位置及文件名称

