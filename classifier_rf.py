import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def left_merge_dataset(left_dframe, right_dframe, merge_column):
   return pd.merge(left_dframe, right_dframe, on=merge_column, how='left')

df = pd.read_csv('processed_dataset.csv')
dest = pd.read_csv('destination_normalize.csv')
merged_data_set = left_merge_dataset(df, dest, 'srch_destination_id')
print('merged:' + str(type(merged_data_set)))
print('df:' + str(type(df)))
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score, average='micro'),
           'recall' : make_scorer(recall_score, average='micro'), 
           'f1_score' : make_scorer(f1_score, average='micro')}

merged_data_set = merged_data_set.dropna()
print('merged:' + str(type(merged_data_set)))
print('df:' + str(type(df)))

columns = ['date_time', 'srch_ci', 'srch_co','user_id','disc_orig_destination_distance','std_srch_children_cnt','std_srch_adults_cnt']

merged_data_set = merged_data_set.drop(columns=columns,axis=1)
print('merged:' + str(type(merged_data_set)))
y = merged_data_set['hotel_cluster']
merged_data_set = merged_data_set.drop(['hotel_cluster'],1)

X = merged_data_set
print('Going into the classifier')

resultMNB = cross_validate(RandomForestClassifier(), X, y, cv=KFold(n_splits=5, shuffle=True),scoring=scoring)
print('Accuracy per fold =', resultMNB['test_accuracy'])
print('Mean Accuracy =', np.mean(resultMNB['test_accuracy']))
print('Mean Precision =', np.mean(resultMNB['test_precision']))
print('Mean Recall =', np.mean(resultMNB['test_recall']))
print('Mean F1 Score =', np.mean(resultMNB['test_f1_score']))