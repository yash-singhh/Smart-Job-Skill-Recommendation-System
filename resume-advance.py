import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import io
import smtplib
from email.mime.text import MIMEText
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Initialize OpenAI API Key
openai.api_key = "API key"  # Replace with your OpenAI key

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\\Users\\Dell\\Desktop\\Frud_detection\\jobs.csv")

# TF-IDF
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
    common_skills = ['python', 'java', 'c++', 'sql', 'html', 'css', 'javascript',
                     'machine learning', 'deep learning', 'nlp', 'pandas', 'numpy',
                     'react', 'node.js', 'django', 'flask', 'excel', 'power bi']
    text_lower = text.lower()
    found_skills = [skill for skill in common_skills if skill in text_lower]
    return ", ".join(found_skills)

def send_email(recipient, subject, body):
    sender_email = "your_email@example.com"
    sender_password = "your_password"  # Use an app password or env var
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)

def generate_job_description(role):
    prompt = f"Write a professional job description for the role: {role}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=250
    )
    return response.choices[0].text.strip()

# Streamlit App
st.title("🔍 Smart Job & Skill Recommendation System")
jobs_df = load_data()

if "favorites" not in st.session_state:
    st.session_state.favorites = []

# Tabs
tabs = st.tabs(["🎯 Find Jobs by Skills", "📘 Discover Skills by Job Role",
                "📄 Upload Resume", "📬 Get Job Alerts", "⭐ Saved Jobs", "🛠️ JD Generator"])

# Tab 1: Jobs by Skills
with tabs[0]:
    st.subheader("🎯 Job Recommendations Based on Your Skills")
    skill_input = st.text_input("Enter your skills", "Python, ML, SQL")
    tfidf_matrix = build_tfidf_matrix(jobs_df['skills'])
    if st.button("Find Jobs", key="job_button_skills"):
        results, _ = recommend_jobs(skill_input, tfidf_matrix, jobs_df)
        if results.empty:
            st.warning("No matching jobs found.")
        else:
            for _, row in results.iterrows():
                st.subheader(f"{row['job_title']} at {row['company']}")
                st.write(f"📍 {row['location']}")
                st.write(f"🛠️ {row['skills']}")
                st.write(f"📝 {row['description']}")
                if st.button(f"❤️ Save {row['job_title']}", key=f"save_{_}"):
                    st.session_state.favorites.append(row.to_dict())
                st.markdown("---")

# Tab 2: Skills by Role
with tabs[1]:
    st.subheader("📘 Recommended Skills for a Job Role")
    job_role_input = st.text_input("Enter a job role", "Data Analyst")
    tfidf_matrix_roles = build_tfidf_matrix(jobs_df['job_title'])
    if st.button("Find Skills", key="skills_button_role"):
        results, _ = recommend_skills(job_role_input, tfidf_matrix_roles, jobs_df)
        for _, row in results.iterrows():
            st.subheader(f"{row['job_title']} at {row['company']}")
            st.write(f"🛠️ {row['skills']}")
            st.write(f"📍 {row['location']}")
            st.write(f"📝 {row['description']}")
            st.markdown("---")

# Tab 3: Resume Upload
with tabs[2]:
    st.subheader("📄 Upload Resume")
    uploaded_file = st.file_uploader("Upload resume", type=["pdf", "txt"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = uploaded_file.getvalue().decode("utf-8")
        extracted_skills = extract_skills_from_text(resume_text)
        st.success(f"✅ Extracted Skills: {extracted_skills}")
        tfidf_matrix_resume = build_tfidf_matrix(jobs_df['skills'])
        results, _ = recommend_jobs(extracted_skills, tfidf_matrix_resume, jobs_df)
        for _, row in results.iterrows():
            st.subheader(f"{row['job_title']} at {row['company']}")
            st.write(f"📍 {row['location']}")
            st.write(f"🛠️ {row['skills']}")
            st.write(f"📝 {row['description']}")
            st.markdown("---")

# Tab 4: Email Alerts
with tabs[3]:
    st.subheader("📬 Receive Job Alerts via Email")
    email = st.text_input("Your Email")
    location = st.text_input("Preferred Location (optional)")
    job_type = st.selectbox("Preferred Job Type", ["Any", "Full-Time", "Part-Time", "Internship"])
    if st.button("Subscribe"):
        sample_jobs = jobs_df.head(3).to_string(index=False)
        body = f"Hi,\n\nHere are some jobs for you:\n\n{sample_jobs}\n\nThanks,\nJobBot"
        send_email(email, "Your Job Recommendations", body)
        st.success("📧 Email sent!")

# Tab 5: Saved Jobs
with tabs[4]:
    st.subheader("⭐ Your Saved Jobs")
    if st.session_state.favorites:
        fav_df = pd.DataFrame(st.session_state.favorites)
        st.dataframe(fav_df)
        csv = fav_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, file_name="saved_jobs.csv", mime="text/csv")
    else:
        st.info("No saved jobs yet.")

# Tab 6: JD Generator
with tabs[5]:
    st.subheader("🛠️ Job Description Generator")
    employer_role = st.text_input("Enter role for JD", "Software Engineer")
    if st.button("Generate JD"):
        jd = generate_job_description(employer_role)
        st.text_area("Generated JD", jd, height=250)
