import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit,KFold,GridSearchCV
from sklearn.metrics import accuracy_score,fbeta_score,f1_score
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


'''
adult_df = pd.read_csv(filepath_or_buffer="adult_cleaned.csv")
adult_df_dummies = pd.get_dummies(adult_df.drop('target', 1))
adult_x = adult_df_dummies.copy().values
adult_y = adult_df["target"].copy().values
print(adult_x.shape)
scaler = StandardScaler()
adult_train_x, adult_test_x, adult_train_y, adult_test_y = train_test_split(adult_x, adult_y, test_size=0.3, random_state=0)
adult_train_x, adult_val_x, adult_train_y, adult_val_y = train_test_split(adult_train_x, adult_train_y, test_size=0.2, random_state=0)
scaler = StandardScaler()
transformed_adult_train_x = scaler.fit_transform(adult_train_x)
transformed_adult_val_x = scaler.transform(adult_val_x)
transformed_adult_test_x = scaler.transform(adult_test_x)




train = pd.DataFrame(np.hstack((transformed_adult_train_x,np.atleast_2d(adult_train_y).T)))
test = pd.DataFrame(np.hstack((transformed_adult_test_x,np.atleast_2d(adult_test_y).T)))
val = pd.DataFrame(np.hstack((transformed_adult_val_x,np.atleast_2d(adult_val_y).T)))
print(f"train: {train.shape}")
print(f"test: {test.shape}")
print(f"val: {val.shape}")

train.to_csv('adult_train.csv',index=False,header=False)
test.to_csv('adult_test.csv',index=False,header=False)
val.to_csv('adult_val.csv',index=False,header=False)
'''

spam_df = pd.read_csv(filepath_or_buffer="spambase.csv")
spam_x = spam_df.drop('class', axis=1).copy().values
spam_y = spam_df['class'].copy().values
scaler = StandardScaler()
spam_train_x, spam_test_x, spam_train_y, spam_test_y = train_test_split(spam_x, spam_y, test_size=0.3, random_state=0)
spam_train_x, spam_val_x, spam_train_y, spam_val_y = train_test_split(spam_train_x, spam_train_y, test_size=0.2, random_state=0)
transformed_spam_train_x = scaler.fit_transform(spam_train_x)
transformed_spam_val_x = scaler.transform(spam_val_x)
transformed_spam_test_x = scaler.transform(spam_test_x)




train = pd.DataFrame(np.hstack((transformed_spam_train_x,np.atleast_2d(spam_train_y).T)))
test = pd.DataFrame(np.hstack((transformed_spam_test_x,np.atleast_2d(spam_test_y).T)))
val = pd.DataFrame(np.hstack((transformed_spam_val_x,np.atleast_2d(spam_val_y).T)))
print(f"train: {train.shape}")
print(f"test: {test.shape}")
print(f"val: {val.shape}")

train.to_csv('spam_train.csv',index=False,header=False)
test.to_csv('spam_test.csv',index=False,header=False)
val.to_csv('spam_val.csv',index=False,header=False)