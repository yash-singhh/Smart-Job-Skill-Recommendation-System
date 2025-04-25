import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\\Users\\Dell\\Desktop\\Frud_detection\\jobs.csv")

# Initialize TF-IDF globally so it's reused
tfidf = TfidfVectorizer(stop_words='english')

def build_similarity_matrix(jobs_df):
    tfidf_matrix = tfidf.fit_transform(jobs_df['skills'].fillna(''))
    return tfidf_matrix

def get_recommendations(user_input, tfidf_matrix, jobs_df, top_n=5):
    user_vec = tfidf.transform([user_input])  # Use the same fitted vectorizer
    similarity_scores = linear_kernel(user_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    return jobs_df.iloc[top_indices], similarity_scores[top_indices]

# Streamlit UI
st.title("🔍 Job Recommendation System")

jobs_df = load_data()
tfidf_matrix = build_similarity_matrix(jobs_df)

user_skills = st.text_input("Enter your skills (comma-separated)", "Python, ML, SQL")

if st.button("Find Jobs"):
    results, scores = get_recommendations(user_skills, tfidf_matrix, jobs_df)
    
    if results.empty:
        st.warning("No matching jobs found.")
    else:
        for idx, row in results.iterrows():
            st.subheader(f"{row['job_title']} at {row['company']}")
            st.write(f"📍 Location: {row['location']}")
            st.write(f"🛠️ Skills: {row['skills']}")
            st.write(f"📝 Description: {row['description']}")
            st.markdown("---")
