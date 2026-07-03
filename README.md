# Real Estate Market Segmentation & Outlier Detection Dashboard

An interactive, enterprise-grade data analytics application built with Python and Streamlit. This project leverages unsupervised machine learning to partition real estate markets, identify high-yield investment zones, and isolate structural anomalies. By combining partitioning algorithms (**K-Means**) and density-based clustering (**DBSCAN**), the dashboard provides real-time, interactive insights into multi-dimensional real estate datasets.

---

## 🚀 Core Product Capabilities

*   **Dual-Engine Clustering Strategy:** 
    *   *Macro Segmentation:* Uses K-Means to divide the housing market into structural pricing brackets (Budget, Mid-Tier, Luxury).
    *   *Micro Segmentation & Outlier Detection:* Uses DBSCAN to identify organic geographic/density pockets while filtering out lone listings, typos, or fake data.
*   **Advanced Dimensionality Reduction:** Utilizes Principal Component Analysis (PCA) to compress high-dimensional real estate features (e.g., Price, Square Footage, BHK, location metrics) into optimized 2D coordinates (`x_pca`, `y_pca`) for robust spatial visualization without losing variance.
*   **Live Hyperparameter Optimization:** Integrates real-time UI sliders in a Streamlit sidebar, allowing users to modify cluster limits ($k$), search radius ($\epsilon$), and density thresholds (`min_samples`) on the fly.
*   **Dynamic Operational Metrics:** Displays live KPI cards tracking anomaly counts, outlier percentages, and standard market segment allocations that update instantly as sliders change.

---

## 🛠️ Architecture & Tech Stack

*   **User Interface:** Streamlit (Layouts, Metrics, Dynamic State Controls)
*   **Data Pipeline & Parsing:** Pandas, NumPy, OpenPyXL
*   **Machine Learning Engine:** Scikit-Learn
    *   *Preprocessing:* `StandardScaler` (Z-score normalization)
    *   *Dimensionality Reduction:* `PCA`
    *   *Clustering Models:* `KMeans`, `DBSCAN`
*   **Data Visualization:** Plotly Express (Interactive 2D Scatter Plots), Matplotlib

---
## Scientific Methodology

 1. Feature Preprocessing & Scaling
Real estate parameters operate on vastly mismatched scalar dimensions (e.g., Price listed in millions versus BHK listed in single digits). To prevent the distance math from being dominated entirely by pricing columns, data is passed through a standard normal distribution scaler (StandardScaler), which centers the data to a mean of 0 and a standard deviation of 1.

2. Dimensionality Reduction (PCA)
To visually display multi-dimensional property rows on a 2D dashboard monitor, Principal Component Analysis calculates eigenvectors to rotate the data into maximized orthogonal directions of variance, projecting high-dimensional metrics into a clean (x, y) coordinate array.

3. K-Means vs. DBSCAN Hyperparameter Tuning
K-Means (k): Partitioning focuses on minimizing the within-cluster sum-of-squares (Inertia). Ideal for classifying broad market brackets.

DBSCAN (Radius / Min Samples): Scans dense neighborhoods. Points with fewer than the specified minimum sample threshold within a given radius fail the density-connection rule and are cleanly separated into a specialized Noise category (Cluster -1).

The 1-20 Slider Guardrail: The minimum samples slider is restricted to a maximum of 20 to strictly align with machine learning heuristics (2 * Dimensions). Pushing this value higher causes severe over-smoothing, which breaks up smaller, highly exclusive luxury real estate sectors and accidentally misclassifies normal niche properties as noise.
