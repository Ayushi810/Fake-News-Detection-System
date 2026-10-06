import streamlit as st
import joblib
import re

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

@st.cache_resource
def load_model():
    model = joblib.load("model/fake_news_model.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

st.title("📰 Fake News Detection System")
st.write("Enter a news headline or short news article to classify it as **REAL** or **FAKE**.")

text = st.text_area(
    "News Text",
    placeholder="Paste a news headline or article here...",
    height=180
)

if st.button("🔍 Check News", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some news text.")
    else:
        cleaned = re.sub(r"\s+", " ", text.strip())
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities) * 100

        if prediction == "REAL":
            st.success(f"✅ Prediction: REAL NEWS")
        else:
            st.error(f"⚠️ Prediction: FAKE NEWS")

        st.info(f"Model confidence: {confidence:.2f}%")

st.divider()
st.caption("Machine Learning project using TF-IDF and Logistic Regression.")
st.caption("⚠️ This tool is an educational classifier and should not be treated as a fact-checking authority.")
