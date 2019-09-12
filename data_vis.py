import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
expedia_df = pd.read_csv(r"C:\Users\deept\PycharmProjects\DA_project\newsample1.csv")
fig, (axis1,axis2) = plt.subplots(2,1,figsize=(15,10))

bookings_df = expedia_df[expedia_df["is_booking"] == 1]


# Plot post_continent & hotel_continent

fig, ((axis1,axis2),(axis3,axis4)) = plt.subplots(2,2,figsize=(15,10))

# Plot frequency for each posa_continent
sns.countplot('posa_continent', data=expedia_df,order=[0,1,2,3,4],palette="Set3",ax=axis1)

# Plot frequency for each posa_continent decomposed by hotel_continent
sns.countplot('posa_continent', hue='hotel_continent',data=expedia_df,order=[0,1,2,3,4],palette="Set3",ax=axis2)

# Plot frequency for each hotel_continent
sns.countplot('hotel_continent', data=expedia_df,order=[0,2,3,4,5,6],palette="Set3",ax=axis3)

# Plot frequency for each hotel_continent decomposed by posa_continent
sns.countplot('hotel_continent', hue='posa_continent', data=expedia_df, order=[0,2,3,4,5,6],palette="Set3",ax=axis4)
# heatmap of the data imp4
fig, ax = plt.subplots()
fig.set_size_inches(15, 10)
sns.heatmap(expedia_df.corr(),cmap='coolwarm',ax=ax,annot=True,linewidths=2)

# Plot number of bookings over Date imp1
# Convert srch_ci column to Date(Y-M)
expedia_df['Date']  = expedia_df['srch_ci'].apply(lambda x: (str(x)[:7]) if x == x else np.nan)

date_bookings  = expedia_df.groupby('Date')["is_booking"].sum()
ax1 = date_bookings.plot(legend=True,marker='o',title="Total Bookings", figsize=(15,5))
ax1.set_xticks(range(len(date_bookings)))
xlabels = ax1.set_xticklabels(date_bookings.index.tolist(), rotation=90)

# correlation between hotel_country in number of bookings through 2013, 2014, & 2015 imp2
hotel_country_piv= pd.pivot_table(expedia_df,values='is_booking', index='Date', columns=['hotel_country'],aggfunc='sum')
hotel_country_piv= hotel_country_piv.fillna(0)
hotel_country_piv.head()

# Plot correlation between range of hotel_country imp3
country_ids = [1,5,7,8,47,50,182,185]

fig, (axis1) = plt.subplots(1,1,figsize=(15,5))
plt.show()