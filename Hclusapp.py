import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Netflix Content Clusterer")

# Load saved components
@st.cache_resource
def load_models():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('cluster_model.pkl')
    data = pd.read_csv('clustered_data.csv')
    return vectorizer, model, data

vectorizer, model, data = load_models()

st.title("🎬 Netflix Title Clustering App")
st.write("Enter a movie description or genre to find its cluster.")

# User Input
user_input = st.text_area("Description/Genre:", "A documentary about space exploration and NASA.")

if st.button("Analyze Cluster"):
    # 1. Vectorize
    vec_input = vectorizer.transform([user_input]).toarray()
    
    # 2. Predict Cluster
    cluster_id = model.predict(vec_input)[0]
    
    st.success(f"This content belongs to **Cluster {cluster_id}**")
    
    # 3. Show similar items from the same cluster
    st.subheader(f"Other titles in Cluster {cluster_id}:")
    results = data[data['cluster'] == cluster_id][['title', 'listed_in']].head(10)
    st.table(results)