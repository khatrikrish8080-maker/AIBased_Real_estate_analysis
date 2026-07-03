import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,DBSCAN

df=pd.read_csv('Delhi_v2.csv')

df['Balcony']=df['Balcony'].fillna(0)
df['parking']=df['parking'].fillna(0)
df['Lift']=df['Lift'].fillna(0)

features = ['price','area','Bedrooms','Bathrooms','Balcony','parking','Lift']

df_clean=df.dropna(subset=['price','area','Bedrooms','Bathrooms']).copy()

def run_ml_pipeline(k_clusters, eps, min_samples):
    
    x=df_clean[features]
   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)
    

    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    

    return df_clean, kmeans_labels, dbscan_labels, X_pca
