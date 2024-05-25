from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Assuming you have your train and test datasets loaded into train_df and test_df
train_df=pd.read_excel("train.xlsx")
test_df=pd.read_excel("test.xlsx")
# # Printing the data
# print(train_df.head())
# print(test_df.head())

#removing the target column
train_df=train_df.drop('target',axis=1)

# print(train_df.head())


# For example, scaling the features
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

# Choose number of clusters (k) - this can be tuned based on domain knowledge or using techniques like elbow method
k = 5

# Train K-Means model
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(train_scaled)

# Assign cluster labels to train and test data
train_clusters = kmeans.predict(train_scaled)
test_clusters = kmeans.predict(test_scaled)

# Now you have cluster labels for both train and test data
# You can use these clusters for further analysis or tasks like anomaly detection, segmentation, etc.
