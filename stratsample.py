import pandas as pd

df = pd.read_csv('training.csv')

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2)

newsample = train.groupby('hotel_cluster').apply(lambda x: x.sample(1000)).reset_index(drop=True)

newsample.to_csv('newsample1.csv')