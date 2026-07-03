import streamlit as st
import pandas as pd
import plotly.express as px

from model import run_ml_pipeline

st.set_page_config(
    page_title="Delhi Real Estate Clustering Dashboard",
    layout="wide"
)

st.title("🏙️ Delhi Real Estate Market Segmentation")
st.markdown("Explore structural market clusters using Market Segments and density-based anomalies using Deal and Anomaly Finder.")
st.markdown("---")

st.sidebar.header("🛠️ Model Hyperparameters")

st.sidebar.markdown("---")

st.sidebar.title("Adjust these sliders to re-cluster the real estate market in real time.")

st.sidebar.markdown("---")

k=st.sidebar.slider("Number of clusters(k) ",2,10,3,step=1)

eps=st.sidebar.slider("Radius",0.1,2.0,0.5,step=0.1)

min_samples=st.sidebar.slider("Minimun Properties per group",2,20,5,step=1)

df_display, kmeans_tags, dbscan_tags, X_pca=run_ml_pipeline(k,eps,min_samples)

df_display['x_pca']=X_pca[:,0]
df_display['y_pca']=X_pca[:,1]

df_display['Market Segment'] = [f"Segment {label + 1}" for label in kmeans_tags]

df_display['Market Status'] = [
    "🚨 Unusual Listing (Anomaly)" if tag == -1 else "✅ Standard Market Group" 
    for tag in dbscan_tags
]

all_features_hover = [
   'Address', 'price', 'area', 'Bedrooms', 'Bathrooms', 'Balcony', 'parking', 'Lift'
]


chart_col1, chart_col2 = st.columns(2)


with chart_col1:
    st.subheader("📊 Structural Market Tiers")
    st.markdown("Ordered groups categorized by full pricing, sizing, and amenity patterns.")
    
    fig_kmeans = px.scatter(
        df_display,
        x="x_pca",
        y="y_pca",
        color="Market Segment",
        hover_data=all_features_hover, 
        title="K-Means Market Clustering Map",
        template="plotly_dark"
    )
    fig_kmeans.update_traces(marker=dict(size=4, opacity=0.6))
    st.plotly_chart(fig_kmeans, use_container_width=True)



with chart_col2:
    st.subheader("🔍 Uncommon Properties & Deals Finder")
    st.markdown("Density scanning to isolate highly unusual listings from the core market data.")
    
    
    color_map = {
        "🚨 Unusual Listing (Anomaly)": "#FF4B4B",  
        "✅ Standard Market Group": "#00CC96"   
    }
    
    fig_dbscan = px.scatter(
        df_display,
        x="x_pca",
        y="y_pca",
        color="Market Status",
        color_discrete_map=color_map,
        hover_data=all_features_hover, 
        title="DBSCAN Outlier Map",
        template="plotly_dark"
    )
    fig_dbscan.update_traces(marker=dict(size=4, opacity=0.6))
    st.plotly_chart(fig_dbscan, use_container_width=True)

st.markdown("---") 


with st.expander("🧭 How to Navigate the Market Maps (Click to Expand)"):
    st.markdown("""
    To view the complex real estate market in a single glance, we combine all dimensions (Price, Area, Bedrooms, and Amenities) into a unified layout:
    * **➡️ Horizontal Axis (Size & Scale):** Moving left-to-right tracks properties growing from smaller apartments to sprawling multi-bedroom configurations.
    * **⬆️ Vertical Axis (Luxury & Premium Pricing):** Moving bottom-to-top tracks rising density pricing, premium brackets, and high-end feature combinations.
    
    *Properties sitting side-by-side in a tight cluster share highly similar architectural traits, sizes, and market values.*
    """)


with st.expander("📊 What do the Segment Numbers mean? (Click to Expand)"):
    st.markdown("""
    The K-Means algorithm groups similar properties together, but it assigns the names (**Segment 1, Segment 2, etc.**) completely at random. 
    
    * **The Numbers are Arbitrary:** "Segment 10" does not mean it is better or more expensive than "Segment 1". The numbers are just random IDs assigned by the AI during initialization.
    * **How to actually read the groups:** Look at the **color chunks** and where they sit on the map rather than the number in the legend:
        * Clusters sitting near the **bottom-left** are your entry-level, smaller, budget-friendly properties.
        * Clusters stretching toward the **far right** are larger, multi-room spaces.
        * Clusters climbing toward the **top** represent high-density premium and luxury price points.
    
    *Think of the numbers as simple category tags, while the position on the graph tells the real story.*
    """)

st.markdown("---")

st.subheader("🔍 Explore the Property Database")
st.markdown("Filter and search through the raw listings matching the clusters above.")

df_table=df_display.copy()

selected_segment=st.select_slider("Filter By AI Segment",options=sorted(df_table['Market Segment'].unique()))

df_table=df_table[df_table['Market Segment']== selected_segment]

search_query = st.text_input("Search by city or location",placeholder="...")

if search_query:
    df_table=df_table[df_table['Address'].str.contains(search_query,case=False,na=False)]


display_cols = ['Address', 'type_of_building', 'price', 'area', 'Bedrooms', 'Bathrooms', 'Market Status']


st.dataframe(
    df_table[[col for col in display_cols if col in df_table.columns]],
    use_container_width=True,
    hide_index=True,
    column_config={
        "area": "Area (Sq. Ft.)",
        "price": "Price (₹)",
        "type_of_building": "Building Type",
        "Market Status": "AI Verdict"
    }
)





