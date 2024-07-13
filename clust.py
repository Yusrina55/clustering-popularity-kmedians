

# import library
import pandas as pd
import numpy as np
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
np.warnings = warnings

# Load dataset
dflama= pd.read_csv('/content/IMDb_All_Genres_etf_clean1.csv')
dflama.head()

# Cek nilai null di seluruh kolom
print(dflama.isnull().sum())

# Hitung jumlah nilai di setiap kolom
jumlah_nilai = dflama.count(axis=0)

# Tampilkan hasil
print(jumlah_nilai)

# MENGHILANGKAN DATA NULL Total Gross

def clean_gross(Total_Gross):

    if isinstance(Total_Gross, str):
        # Hapus dollar dan M, ubah ke float dengan dikali 1e6
        Total_Gross = Total_Gross.replace('$', '').replace('M', '')
        try:
            # Convert to float and scale by 1 million
            return float(Total_Gross) * 1e6
        except ValueError:
            # Handle non-numeric value
            return np.nan
    else:
        # Handle non-string values (sudah numeric)
        return Total_Gross  # Return nilai asli

# Load dataset
df = pd.read_csv('/content/IMDb_All_Genres_etf_clean1.csv')

# Clean total gross
df['Total_Gross'] = df['Total_Gross'].apply(clean_gross)

# Drop null values
df.dropna(subset=['Total_Gross'], inplace=True)

# Explore and analyze the cleaned DataFrame
print(df.head())
print(df['Total_Gross'].isnull().sum())  # Check for remaining null values

# handle missing value 'Rating'
df.dropna(subset=['Rating'], inplace=True)
print(df['Rating'].isnull().sum())

# Hitung jumlah nilai di setiap kolom
jumlah_nilai = df.count(axis=0)
print(jumlah_nilai)

# drop kolom yang tidak digunakan
df2 = df.drop(["Movie_Title", "Director", "Actors", "side_genre", "Runtime(Mins)", "Censor"], axis=1)
df2.head()

# Preprocess genre data
df2['main_genre'] = df2['main_genre'].apply(lambda x: x.split(','))

# One-hot encode genres
genre_df = df2['main_genre'].str.join('|').str.get_dummies()
print(genre_df)

# masukkan pada dtaframe
genre_df['Rating'] = df2['Rating']
genre_df['Total_Gross'] = df2['Total_Gross']

# Normalisasi minmax 'Rating' dan 'Total_Gross'
scaler = MinMaxScaler()
genre_df[['Rating', 'Total_Gross']] = scaler.fit_transform(genre_df[['Rating', 'Total_Gross']])

# Convert ke numpy
X = genre_df.values

import seaborn as sns

# Load dataset
file_path = '/content/IMDb_All_Genres_etf_clean1.csv'
data = pd.read_csv(file_path)

# penghitungan IQR
Q1 = data['Rating'].quantile(0.25)
Q3 = data['Rating'].quantile(0.75)
IQR = Q3 - Q1

outliers_data = data[(data['Rating'] < (Q1 - 1.5 * IQR)) | (data['Rating'] > (Q3 + 1.5 * IQR))]

# Check columns in the filtered data for debugging purposes
print(outliers_data.columns)

# Check if 'main_genre' is present in the DataFrame
if 'main_genre' not in outliers_data.columns:
    raise ValueError("Column 'main_genre' not found in the filtered data")

# Buat subplot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Boxplot untuk rating
sns.boxplot(
    x="main_genre",
    y="Rating",
    showmeans=True,
    data=outliers_data,
    ax=axes[0]
)
axes[0].set_xlabel("Main Genre")
axes[0].set_ylabel("Rating")
axes[0].set_title("Boxplot of Outliers (Rating)")

# Boxplot untuk total gross
sns.boxplot(
    x="main_genre",
    y="Total_Gross",
    showmeans=True,
    data=outliers_data,
    ax=axes[1]
)
axes[1].set_xlabel("Main Genre")
axes[1].set_ylabel("Total Gross")
axes[1].set_title("Boxplot of Outliers (Total Gross)")

plt.tight_layout()
plt.show()

# normalisasi
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(cleaned_genre_df[['Rating', 'Total_Gross']])

# banyak kluster
K = 3

# Inisialisasi dengan K-Means++
initial_centers = kmeans_plusplus_initializer(X_normalized, K).initialize()

# masuk klustering K-Medians (Melewati K-Means karena K-Medians adalah turunannya)
kmedians_instance = kmedians(X_normalized, initial_centers)
kmedians_instance.process()

# Get clusters and medians
clusters = kmedians_instance.get_clusters()
medians = np.array(kmedians_instance.get_medians())

# Assign clusters to each point
y_kmedians = np.zeros(X_normalized.shape[0])
for cluster_id, cluster in enumerate(clusters):
    for index in cluster:
        y_kmedians[index] = cluster_id

# Add cluster labels to the original dataframe
cleaned_df['Cluster'] = y_kmedians

# Analyze clusters and sort them by average rating and total gross
cluster_info = []
for i in range(K):
    cluster_genres = cleaned_genre_df.iloc[clusters[i], :-2].sum().sort_values(ascending=False)
    avg_rating = cleaned_df[cleaned_df['Cluster'] == i]['Rating'].mean()
    avg_gross = cleaned_df[cleaned_df['Cluster'] == i]['Total_Gross'].mean()
    cluster_info.append((i, avg_rating, avg_gross, cluster_genres))

# Sort clusters by average rating and total gross
sorted_clusters = sorted(cluster_info, key=lambda x: (x[1], x[2]), reverse=True)

# Create a mapping from old cluster labels to new sorted labels
old_to_new_cluster = {info[0]: idx for idx, info in enumerate(sorted_clusters)}

# Update cluster labels in the original dataframe
cleaned_df['Cluster'] = cleaned_df['Cluster'].map(old_to_new_cluster)

# Visualize the clustering result with updated labels
plt.figure(figsize=(12, 8))
ax = plt.gca()
ax.grid(True, zorder=1)

# Scatter plot for medians with updated labels
new_medians = np.array([medians[info[0]] for info in sorted_clusters])
ax.scatter(new_medians[:, 0], new_medians[:, 1], marker='+', c='red', zorder=4)

# Scatter plot for clusters with updated labels
colors = ['blue', 'green', 'purple']
for i in range(K):
    idx = np.where(cleaned_df['Cluster'] == i)
    ax.scatter(X_normalized[idx, 0], X_normalized[idx, 1], marker='.', zorder=3, c=colors[i])

plt.xlabel('Normalized Rating')
plt.ylabel('Normalized Total Gross')
plt.title('K-Medians Clustering of Movies by Rating and Total Gross')
plt.show()

# Print sorted cluster analysis
for idx, (original_cluster_id, avg_rating, avg_gross, cluster_genres) in enumerate(sorted_clusters):
    print(f"Cluster {idx+1}:")
    print(f"- Dominant Genres: {cluster_genres.head(3)}")
    print(f"- Average Rating: {avg_rating:.2f}")
    print(f"- Average Total Gross: ${avg_gross:,.2f}")

from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in range(2, 6):
    # Initialize initial cluster centers using k-means++ method
    initial_centers = kmeans_plusplus_initializer(X_normalized, k).initialize()

    # Create an instance of kmedians and run the clustering process
    kmedians_instance = kmedians(X_normalized, initial_centers)
    kmedians_instance.process()

    # Get clusters
    clusters = kmedians_instance.get_clusters()

    # Assign clusters to each point
    y_kmedians = np.zeros(X_normalized.shape[0])
    for cluster_id, cluster in enumerate(clusters):
        for index in cluster:
            y_kmedians[index] = cluster_id

    # Compute silhouette score
    silhouette = silhouette_score(X_normalized, y_kmedians)
    silhouette_scores.append(silhouette)
    print(f"For k={k}, Silhouette Score: {silhouette}")

# Plot silhouette scores
plt.figure()
plt.plot(range(2, 6), silhouette_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different Values of k')
plt.grid(True)
plt.show()