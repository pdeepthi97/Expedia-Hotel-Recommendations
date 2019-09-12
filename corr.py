prerna [10:34 PM]
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
expedia_df = pd.read_csv("newsample1.csv")


# Convert srch_ci column to Date(Y-M)
expedia_df['Date']  = expedia_df['srch_ci'].apply(lambda x: (str(x)[:7]) if x == x else np.nan)

hotel_country_piv       = pd.pivot_table(expedia_df,values='is_booking', index='Date', columns=['hotel_country'],aggfunc='sum')
hotel_country_piv       = hotel_country_piv.fillna(0)
hotel_country_piv.head()

# Plot correlation between range of hotel_country
country_ids = [1,5,7,8,47,50,182,185]

fig, (axis1) = plt.subplots(1,1,figsize=(15,5))

# using summation of booking values for each hotel_country
sns.heatmap(hotel_country_piv[country_ids].corr(),annot=True,linewidths=2,cmap="YlGnBu");