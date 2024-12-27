import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Generate sample customer data
np.random.seed(42)
n_customers = 1000

# Generate features
annual_income = np.concatenate([
    np.random.normal(45000, 10000, n_customers//3),  # Lower income
    np.random.normal(75000, 15000, n_customers//3),  # Middle income
    np.random.normal(120000, 25000, n_customers//3)  # Higher income
])

age = np.concatenate([
    np.random.normal(25, 5, n_customers//3),  # Younger
    np.random.normal(45, 10, n_customers//3),  # Middle-aged
    np.random.normal(65, 8, n_customers//3)    # Older
])

spending_score = np.concatenate([
    np.random.normal(30, 10, n_customers//3),  # Low spenders
    np.random.normal(60, 15, n_customers//3),  # Medium spenders
    np.random.normal(90, 20, n_customers//3)   # High spenders
])

purchase_frequency = np.concatenate([
    np.random.normal(2, 1, n_customers//3),    # Infrequent
    np.random.normal(8, 2, n_customers//3),    # Regular
    np.random.normal(15, 3, n_customers//3)    # Frequent
])

avg_order_value = np.concatenate([
    np.random.normal(50, 20, n_customers//3),   # Small orders
    np.random.normal(150, 40, n_customers//3),  # Medium orders
    np.random.normal(300, 80, n_customers//3)   # Large orders
])

# Create DataFrame
data = pd.DataFrame({
    'annual_income': annual_income,
    'age': age,
    'spending_score': spending_score,
    'purchase_frequency': purchase_frequency,
    'avg_order_value': avg_order_value
})

# Clean up any negative values
data = data.clip(lower=0)

print("Step 1: Data Overview")
print("\nSample of customer data:")
print(data.head())
print("\nData Description:")
print(data.describe())

# Scale the features
scaler = StandardScaler()
data_scaled = pd.DataFrame(
    scaler.fit_transform(data),
    columns=data.columns
)

print("\nStep 2: Finding Optimal Number of Clusters")
# Try different numbers of clusters
max_clusters = 10
silhouette_scores = []
inertias = []

for n_clusters in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data_scaled)
    
    silhouette_avg = silhouette_score(data_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    inertias.append(kmeans.inertia_)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    print(f"Inertia: {kmeans.inertia_:.3f}\n")

# Choose optimal number of clusters (using silhouette score)
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Optimal number of clusters based on Silhouette Score: {optimal_clusters}")

# Perform final clustering with optimal number of clusters
final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Cluster'] = final_kmeans.fit_predict(data_scaled)

print("\nStep 3: Analyzing Clusters")
# Calculate cluster characteristics
cluster_analysis = data.groupby('Cluster').mean()
print("\nCluster Centers (Mean Values):")
print(cluster_analysis)

# Calculate cluster sizes
cluster_sizes = data['Cluster'].value_counts().sort_index()
print("\nCluster Sizes:")
print(cluster_sizes)

# Function to interpret clusters
def interpret_cluster(cluster_stats):
    interpretations = []
    for cluster in cluster_stats.index:
        stats = cluster_stats.loc[cluster]
        
        # Initialize characteristics list
        characteristics = []
        
        # Income level
        if stats['annual_income'] < 50000:
            characteristics.append("low income")
        elif stats['annual_income'] < 90000:
            characteristics.append("middle income")
        else:
            characteristics.append("high income")
            
        # Age group
        if stats['age'] < 35:
            characteristics.append("young")
        elif stats['age'] < 55:
            characteristics.append("middle-aged")
        else:
            characteristics.append("senior")
            
        # Spending behavior
        if stats['spending_score'] < 40:
            characteristics.append("conservative spender")
        elif stats['spending_score'] < 70:
            characteristics.append("moderate spender")
        else:
            characteristics.append("big spender")
            
        # Purchase frequency
        if stats['purchase_frequency'] < 5:
            characteristics.append("infrequent buyer")
        elif stats['purchase_frequency'] < 12:
            characteristics.append("regular buyer")
        else:
            characteristics.append("frequent buyer")
            
        interpretation = f"Cluster {cluster}: " + ", ".join(characteristics)
        interpretations.append(interpretation)
    
    return interpretations

print("\nStep 4: Cluster Interpretations")
cluster_interpretations = interpret_cluster(cluster_analysis)
for interpretation in cluster_interpretations:
    print(interpretation)

print("\nStep 5: Customer Segmentation Example")
# Create sample customers for prediction
sample_customers = pd.DataFrame({
    'annual_income': [35000, 80000, 150000],
    'age': [25, 45, 65],
    'spending_score': [20, 60, 90],
    'purchase_frequency': [2, 8, 15],
    'avg_order_value': [50, 150, 300]
})

print("\nSample customer profiles:")
print(sample_customers)

# Scale the sample customers
sample_customers_scaled = scaler.transform(sample_customers)

# Predict clusters
sample_clusters = final_kmeans.predict(sample_customers_scaled)

print("\nPredicted segments for sample customers:")
for i, cluster in enumerate(sample_clusters):
    print(f"Customer {i+1} belongs to {cluster_interpretations[cluster]}")

# Calculate feature importance for each cluster
print("\nStep 6: Feature Importance Analysis")
feature_importance = pd.DataFrame(
    abs(final_kmeans.cluster_centers_),
    columns=data.columns[:-1]  # Exclude the Cluster column
)

print("\nFeature importance for each cluster:")
print(feature_importance)