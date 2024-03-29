import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import metrics
import pickle

d=DecisionTreeClassifier()

def left_merge_dataset(left_dframe, right_dframe, merge_column):
   return pd.merge(left_dframe, right_dframe, on=merge_column, how='left')

df = pd.read_csv('processed_dataset.csv')
dest = pd.read_csv('destination_normalize.csv')
merged_data_set = left_merge_dataset(df, dest, 'srch_destination_id')
print('merged:' + str(type(merged_data_set)))
print('df:' + str(type(df)))

merged_data_set = merged_data_set.dropna()
print('merged:' + str(type(merged_data_set)))
print('df:' + str(type(df)))

columns = ['date_time', 'srch_ci', 'srch_co','user_id','disc_orig_destination_distance','std_srch_children_cnt','std_srch_adults_cnt']

merged_data_set = merged_data_set.drop(columns=columns,axis=1)
print('merged:' + str(type(merged_data_set)))
y = merged_data_set['hotel_cluster']
merged_data_set = merged_data_set.drop(['hotel_cluster'],1)

X = merged_data_set
print('Going/ Decision into the classifier')

resultDT = d.fit(X,y)
s= pickle.dumps(resultDT)
pickle.dump(s, open('tree.sav', 'wb'))
