import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(
    page_title="AI Spam Email Detector",
    page_icon="📧"
)

st.title("📧 AI Spam Email Detector")

data = {
    "email": [
        "Congratulations! You won a free iPhone",
        "Claim your cash prize now",
        "Meeting scheduled for tomorrow",
        "Project report attached",
        "Win money instantly click here",
        "Let's discuss the assignment",
        "Urgent! Verify your bank account",
        "Your invoice is attached",
    ],
    "label": [
        "spam",
        "spam",
        "ham",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
    ],
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["email"])

model = MultinomialNB()
model.fit(X, df["label"])

email_text = st.text_area(
    "Paste Email Content",
    height=200
)

if st.button("Analyze Email"):
    if email_text.strip():

        email_vector = vectorizer.transform([email_text])
        prediction = model.predict(email_vector)[0]

        if prediction == "spam":
            st.error("🚨 Spam Email Detected")
        else:
            st.success("✅ Legitimate Email")

    else:
        st.warning("Please enter email content.")
