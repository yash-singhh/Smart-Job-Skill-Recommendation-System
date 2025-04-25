import streamlit as st
import pandas as pd
import fitz  # PyMuPDF for PDF reading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import io

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\\Users\\Dell\\Desktop\\Frud_detection\\jobs.csv")

# TF-IDF initialized globally
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

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_skills_from_text(text):
    # Simple keyword matching for skill extraction (enhanceable via NLP)
    common_skills = ['python', 'java', 'c++', 'sql', 'html', 'css', 'javascript',
                     'machine learning', 'deep learning', 'nlp', 'pandas', 'numpy',
                     'react', 'node.js', 'django', 'flask', 'excel', 'power bi']
    text_lower = text.lower()
    found_skills = [skill for skill in common_skills if skill in text_lower]
    return ", ".join(found_skills)

# Streamlit UI
st.title("🔍 Smart Job & Skill Recommendation System")

# Load dataset
jobs_df = load_data()

# Tabs for three different modes
tab1, tab2, tab3 = st.tabs(["🎯 Find Jobs by Skills", "📘 Discover Skills by Job Role", "📄 Upload Resume for Job Suggestions"])

# ----------------------------- Tab 1: Jobs by Skills -----------------------------
with tab1:
    st.subheader("🎯 Job Recommendations Based on Your Skills")
    skill_input = st.text_input("Enter your skills (comma-separated)", "Python, ML, SQL")
    tfidf_matrix = build_tfidf_matrix(jobs_df['skills'])

    if st.button("Find Jobs", key="job_button_skills"):
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

# ----------------------------- Tab 2: Skills by Role -----------------------------
with tab2:
    st.subheader("📘 Recommended Skills for a Job Role")
    job_role_input = st.text_input("Enter a job role (e.g., Data Scientist)", "Data Analyst")
    tfidf_matrix_roles = build_tfidf_matrix(jobs_df['job_title'])

    if st.button("Find Skills", key="skills_button_role"):
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

# ----------------------------- Tab 3: Resume Upload -----------------------------
with tab3:
    st.subheader("📄 Upload Resume for Personalized Job Recommendations")
    uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])

    if uploaded_file is not None:
        # Read and extract resume content
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            resume_text = stringio.read()

        extracted_skills = extract_skills_from_text(resume_text)
        st.success(f"✅ Extracted Skills: {extracted_skills}")

        if extracted_skills:
            tfidf_matrix_resume = build_tfidf_matrix(jobs_df['skills'])
            results, _ = recommend_jobs(extracted_skills, tfidf_matrix_resume, jobs_df)

            if results.empty:
                st.warning("No matching jobs found.")
            else:
                st.subheader("🔎 Job Recommendations Based on Your Resume:")
                for _, row in results.iterrows():
                    st.subheader(f"{row['job_title']} at {row['company']}")
                    st.write(f"📍 Location: {row['location']}")
                    st.write(f"🛠️ Skills: {row['skills']}")
                    st.write(f"📝 Description: {row['description']}")
                    st.markdown("---")
        else:
            st.warning("Could not extract relevant skills from your resume.")
