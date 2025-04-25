import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\\Users\\Dell\\Desktop\\Frud_detection\\jobs.csv")

# Initialize TF-IDF globally
tfidf = TfidfVectorizer(stop_words='english')

def build_tfidf_matrix(column_data):
    return tfidf.fit_transform(column_data.fillna(''))

def recommend_jobs(user_input, tfidf_matrix, jobs_df, top_n=5):
    user_vec = tfidf.transform([user_input])
    similarity_scores = linear_kernel(user_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    return jobs_df.iloc[top_indices], similarity_scores[top_indices]

def recommend_skills(job_role_input, tfidf_matrix, jobs_df, top_n=5):
    role_vec = tfidf.transform([job_role_input])
    similarity_scores = linear_kernel(role_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    return jobs_df.iloc[top_indices], similarity_scores[top_indices]

# Streamlit UI
st.title("🔍 Smart Job & Skill Recommendation System")

# Load and preprocess data
jobs_df = load_data()

# Tabs for two different modes
tab1, tab2 = st.tabs(["🎯 Find Jobs by Skills", "📘 Discover Skills by Job Role"])

with tab1:
    st.subheader("🎯 Job Recommendations Based on Your Skills")
    skill_input = st.text_input("Enter your skills (comma-separated)", "Python, ML, SQL")
    tfidf_matrix = build_tfidf_matrix(jobs_df['skills'])

    if st.button("Find Jobs"):
        results, _ = recommend_jobs(skill_input, tfidf_matrix, jobs_df)

        if results.empty:
            st.warning("No matching jobs found.")
        else:
            for _, row in results.iterrows():
                st.subheader(f"{row['job_title']} at {row['company']}")
                st.write(f"📍 Location: {row['location']}")
                st.write(f"🛠️ Skills: {row['skills']}")
                st.write(f"📝 Description: {row['description']}")
                st.markdown("---")

with tab2:
    st.subheader("📘 Recommended Skills for a Job Role")
    job_role_input = st.text_input("Enter a job role (e.g., Data Scientist)", "Data Analyst")
    tfidf_matrix_roles = build_tfidf_matrix(jobs_df['job_title'])

    if st.button("Find Skills"):
        results, _ = recommend_skills(job_role_input, tfidf_matrix_roles, jobs_df)

        if results.empty:
            st.warning("No similar job roles found.")
        else:
            for _, row in results.iterrows():
                st.subheader(f"{row['job_title']} at {row['company']}")
                st.write(f"🛠️ Suggested Skills: {row['skills']}")
                st.write(f"📍 Location: {row['location']}")
                st.write(f"📝 Description: {row['description']}")
                st.markdown("---")
