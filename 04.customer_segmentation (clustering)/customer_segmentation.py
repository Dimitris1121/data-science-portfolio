import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Load Dataset
df = pd.read_csv(r'D:\Δουλειά\Python\Apr26 Refresh ML\04.Customer_Segmentation\online_retail.csv')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn info:")
print(df.info())


# ---- CLEANING ----
# Drop rows with no CustomerID
df = df.dropna(subset=['CustomerID'])

# Drop negative quantities (returns/cancellations)
df = df[df['Quantity'] > 0]

# Drop negative prices
df = df[df['UnitPrice'] > 0]

# Convert CustomerID to integer
df['CustomerID'] = df['CustomerID'].astype(int)

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print("Clean shape:", df.shape)
print("Unique customers:", df['CustomerID'].nunique())



# ---- RFM FEATURE ENGINEERING ----

# Reference date - day after last transaction
reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Calculate total spend per transaction
df['TotalSpend'] = df['Quantity'] * df['UnitPrice']

# Build RFM table - one row per customer
rfm = df.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('TotalSpend', 'sum')
).reset_index()

print("RFM Table:")
print(rfm.head(10))
print("\nRFM Statistics:")
print(rfm.describe())



# ---- SCALE THE DATA ----
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
rfm_scaled = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])

print("Scaled RFM:")
print(rfm_scaled.describe())




# ---- ELBOW METHOD ----
inertia = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o', color='blue')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method - Finding Optimal K')
plt.xticks(k_range)
plt.grid(True)
plt.show()


# ---- KMEANS WITH K=4 ----
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Check cluster sizes
print(rfm.head(10))
print("Cluster sizes:")
print(rfm['Cluster'].value_counts())

# Cluster profiles
print("\nCluster profiles:")
print(rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2))



# ---- VISUALIZE CLUSTERS ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Recency vs Frequency
axes[0].scatter(rfm['Recency'], rfm['Frequency'], 
                c=rfm['Cluster'], cmap='viridis', alpha=0.5)
axes[0].set_xlabel('Recency (days)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Recency vs Frequency')

# Recency vs Monetary
axes[1].scatter(rfm['Recency'], rfm['Monetary'], 
                c=rfm['Cluster'], cmap='viridis', alpha=0.5)
axes[1].set_xlabel('Recency (days)')
axes[1].set_ylabel('Monetary ($)')
axes[1].set_title('Recency vs Monetary')

# Frequency vs Monetary
axes[2].scatter(rfm['Frequency'], rfm['Monetary'], 
                c=rfm['Cluster'], cmap='viridis', alpha=0.5)
axes[2].set_xlabel('Frequency')
axes[2].set_ylabel('Monetary ($)')
axes[2].set_title('Frequency vs Monetary')

plt.suptitle('Customer Segments - RFM Clustering', fontsize=14)
plt.tight_layout()
plt.show()


# ---- SILHOUETTE SCORE ----
score = silhouette_score(rfm_scaled, rfm['Cluster'])
print(f"Silhouette Score: {score:.4f}")


# ---- ADD BUSINESS LABELS ----
cluster_labels = {
    0: 'Loyal Customers',
    1: 'Lost Customers',
    2: 'Champions',
    3: 'High Value Actives'
}

rfm['Segment'] = rfm['Cluster'].map(cluster_labels)

# Final segment profiles
print("---- CUSTOMER SEGMENTS ----")
segment_summary = rfm.groupby('Segment').agg(
    Customers=('CustomerID', 'count'),
    Avg_Recency=('Recency', 'mean'),
    Avg_Frequency=('Frequency', 'mean'),
    Avg_Monetary=('Monetary', 'mean')
).round(2)

print(segment_summary)

# Visualize segment sizes
plt.figure(figsize=(8, 6))
rfm['Segment'].value_counts().plot(
    kind='bar',
    color=['gold', 'red', 'green', 'blue'],
    edgecolor='black'
)
plt.title('Customer Segment Distribution')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save results
rfm.to_csv('customer_segments.csv', index=False)
print("\n✅ Segments saved to customer_segments.csv")