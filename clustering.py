import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage


def load_data() -> pd.DataFrame:
    df = pd.read_csv('Mall_Customers.csv')
    df.drop('CustomerID', axis=1, inplace=True)
    return df


def _assign_segment(income: float, spend: float) -> str:
    if income > 70 and spend > 65:
        return 'Premium Shoppers'
    elif income > 70 and spend <= 65:
        return 'High Income, Low Spenders'
    elif income <= 45 and spend > 65:
        return 'Budget Enthusiasts'
    elif income <= 45 and spend <= 45:
        return 'Low Income, Low Spenders'
    else:
        return 'Average Customers'


def run_kmeans(df: pd.DataFrame, k: int = 5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[['Annual Income (k$)', 'Spending Score (1-100)']])

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    result = df.copy()
    result['Cluster'] = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, kmeans.labels_)

    profile = result.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
    segment_map = {
        c: _assign_segment(r['Annual Income (k$)'], r['Spending Score (1-100)'])
        for c, r in profile.iterrows()
    }
    result['Segment'] = result['Cluster'].map(segment_map)
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)

    return result, segment_map, score, centroids


def compute_wcss(df: pd.DataFrame) -> list:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    wcss = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
    return wcss


def run_hierarchical(df: pd.DataFrame):
    le = LabelEncoder()
    df_hc = df.copy()
    df_hc['Genre'] = le.fit_transform(df_hc['Genre'])
    features = df_hc[['Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = MinMaxScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns
    )
    Z = linkage(features_scaled, method='ward')
    return Z, features_scaled
