import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
expedia_df = pd.read_csv(r"C:\Users\deept\PycharmProjects\DA_project\newsample1.csv")
fig, (axis1,axis2) = plt.subplots(2,1,figsize=(15,10))

bookings_df = expedia_df[expedia_df["is_booking"] == 1]

# What are the  countries the customers travel most from?
sns.countplot('user_location_country',data=bookings_df.sort_values(by=['user_location_country']),ax=axis1,palette="Set3")

# What are the most countries the customer travel to?
sns.countplot('hotel_country',data=bookings_df.sort_values(by=['hotel_country']),ax=axis2,palette="Set3")
plt.show()