## 🏡 NCR Real Estate Analytics & Price Prediction Suite

An enterprise-grade Data Science and Machine Learning ecosystem designed to analyze, cluster, segment, and estimate residential property prices across the National Capital Region (NCR).

This project bridges the gap between complex unsupervised market analysis and real-time supervised predictive inference. It uses K-Means and DBSCAN to isolate structural tiers and anomalies, projects high-dimensional metrics using PCA, and trains a robust Random Forest Regressor to power an interactive Streamlit deployment.

##🛠️ Architecture & Tech Stack

User Interface: Streamlit (Layouts, Metrics, Dynamic State Controls)

Data Pipeline & Parsing: Python, Pandas, NumPy

Machine Learning Engine: Scikit-Learn

Preprocessing: StandardScaler (Z-score normalization)

Dimensionality Reduction: PCA

Clustering Models: KMeans, DBSCAN

Regression Model: RandomForestRegressor

Data Visualization: Plotly Express (Interactive 2D Scatter Plots), Matplotlib

Model Serialization: Pickle

## 📂 Project Architecture
Plaintext
├── Delhi_v2.csv          # Raw scraped dataset containing listing and raw addresses
├── Delhi_processed.csv   # Unified dataset exported with engineered locality and price metrics
├── unsupervised.py       # Market segmentation, PCA dimensionality reduction, and outlier detection
├── supervised.py         # Primary ML pipeline (Feature engineering, training, evaluation)
├── app.py                # Live interactive Streamlit dashboard application
├── rf_model.pkl          # Saved weights of the trained Random Forest Regressor
├── scaler.pkl            # Trained StandardScaler mathematical parameters
└── README.md             # Project documentation

##🚀 Core Product Capabilities

1. Unsupervised Exploration Dashboard (unsupervised.py)
Dual-Engine Clustering Strategy:

Macro Segmentation: Uses K-Means to divide the housing market into structural pricing brackets (Budget, Mid-Tier, Luxury).

Micro Segmentation & Outlier Detection: Uses DBSCAN to identify organic geographic/density pockets while filtering out lone listings, typos, or fake data.

Advanced Dimensionality Reduction: Utilizes PCA to compress high-dimensional real estate features (Price, Square Footage, BHK, location metrics) into optimized 2D coordinates (x 
pca
​
 , y 
pca
​
 ) for robust spatial visualization without losing variance.

Live Hyperparameter Optimization: Integrates real-time UI sliders in a Streamlit sidebar, allowing users to modify cluster limits (k), search radius (ϵ), and density thresholds (min_samples) on the fly.

Dynamic Operational Metrics: Displays live KPI cards tracking anomaly counts, outlier percentages, and standard market segment allocations that update instantly as sliders change.

2. Supervised Training Pipeline (supervised.py)
Feature Extraction: Cleans unstructured address strings to extract precise neighborhood keys on-the-fly:
df['locality'] = df['Address'].str.split(',').str[0:3].str.join(' ')

Target Encoding: Computes localized median price indexes (Locality_prices) per neighborhood to capture micro-market spatial variance.

Data Persistence: Automatically snapshots the engineered data out to Delhi_processed.csv to cleanly feed the web application.

3. Production Predictive Interface (app.py)
Seamless UI Mapping: Automatically maps human-readable textual neighborhood selectors to their hidden background target-encoded Locality_prices.

Inference Guardrails: Transforms live user inputs through the pre-fit training scaler to eliminate deployment data drift before feeding the Random Forest Regressor.

##🔬 Scientific Methodology

Feature Preprocessing & Scaling
Real estate parameters operate on vastly mismatched scalar dimensions (e.g., Price listed in millions versus BHK listed in single digits). To prevent the distance math from being dominated entirely by pricing columns, data is passed through a standard normal distribution scaler (StandardScaler), which centers the data to a mean of 0 and a standard deviation of 1.

Dimensionality Reduction (PCA)
To visually display multi-dimensional property rows on a 2D dashboard monitor, Principal Component Analysis calculates eigenvectors to rotate the data into maximized orthogonal directions of variance, projecting high-dimensional metrics into a clean (x, y) coordinate array.

K-Means vs. DBSCAN Hyperparameter Tuning
K-Means (k): Partitioning focuses on minimizing the within-cluster sum-of-squares (Inertia). Ideal for classifying broad market brackets.

DBSCAN (Radius / Min Samples): Scans dense neighborhoods. Points with fewer than the specified minimum sample threshold within a given radius fail the density-connection rule and are cleanly separated into a specialized Noise category (Cluster -1).

The 1-20 Slider Guardrail: The minimum samples slider is restricted to a maximum of 20 to strictly align with machine learning heuristics (2×Dimensions). Pushing this value higher causes severe over-smoothing, which breaks up smaller, highly exclusive luxury real estate sectors and accidentally misclassifies normal niche properties as noise.

##🏁 Installation & Step-by-Step Execution

1. Environment Setup
Clone your workspace repository and ensure your Python virtual environment is fully active:

Bash
# Activate your local environment (Windows)
.\venv\Scripts\activate

# Install project dependencies
pip install pandas numpy scikit-learn streamlit openpyxl plotly matplotlib
2. Run the Data & Model Pipeline
Bash
# 1. Launch the interactive clustering and outlier analysis dashboard
python unsupervised.py

# 2. Run model training to build 'Delhi_processed.csv', 'rf_model.pkl', and 'scaler.pkl'
python supervised.py

# 3. Launch the live price prediction app
streamlit run app.py
This covers every square inch of your machine learning workflow. Ready to push this clean layout over to GitHub?
