import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

def main():
    st.title("KMeans Clustering App")
    st.write("Upload your train and test datasets.")

    train_file = st.file_uploader("Upload Train Dataset", type=["xlsx"])
    test_file = st.file_uploader("Upload Test Dataset", type=["xlsx"])

    if train_file is not None and test_file is not None:
        train_df = pd.read_excel(train_file)
        test_df = pd.read_excel(test_file)

        # Removing the target column
        if 'target' in train_df.columns:
            train_df = train_df.drop('target', axis=1)

        # Scaling the features
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df)
        test_scaled = scaler.transform(test_df)

        # Choose number of clusters (k)
        k = st.slider("Select number of clusters (k)", 2, 10, 5)

        # Train K-Means model
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(train_scaled)

        # Assign cluster labels to train and test data
        train_clusters = kmeans.predict(train_scaled)
        test_clusters = kmeans.predict(test_scaled)

        st.write("Train Cluster Labels:")
        st.write(train_clusters)

        st.write("Test Cluster Labels:")
        st.write(test_clusters)

        # Allow user to provide a data point
        st.write("Enter a data point (a row) to identify its cluster:")
        user_input = st.text_input("Enter comma-separated values")

        if user_input:
            # Convert user input to array
            user_data = [float(x.strip()) for x in user_input.split(',')]
            user_data_scaled = scaler.transform([user_data])
            
            # Predict cluster for user data
            user_cluster = kmeans.predict(user_data_scaled)[0]
            st.write(f"The provided data point belongs to cluster {user_cluster}")

            # Explain why the data point belongs to that cluster
            cluster_centers = kmeans.cluster_centers_
            distances = ((cluster_centers - user_data_scaled) ** 2).sum(axis=1)
            closest_cluster_index = distances.argmin()
            closest_cluster_center = cluster_centers[closest_cluster_index]
            st.write(f"The data point is closest to the cluster center of cluster {user_cluster}")
            st.write(f"Cluster Center Values: {closest_cluster_center}")

if __name__ == "__main__":
    main()
