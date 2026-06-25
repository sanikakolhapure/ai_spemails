import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧"
)

st.title("📧 Spam Email Detection System")

# Sample Dataset
data = {
    "email": [
        "Congratulations! You won a free iPhone",
        "Claim your cash prize now",
        "Meeting scheduled for tomorrow",
        "Project report attached",
        "Win money instantly click here",
        "Let's discuss the assignment"
    ],
    "label": [
        "spam",
        "spam",
        "ham",
        "ham",
        "spam",
        "ham"
    ]
}

df = pd.DataFrame(data)

# Train Model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["email"])
y = df["label"]

model = MultinomialNB()
model.fit(X, y)

# User Input
email_text = st.text_area(
    "Enter Email Content",
    placeholder="Paste email text here..."
)

if st.button("Check Email"):

    if email_text.strip():

        email_vector = vectorizer.transform([email_text])
        prediction = model.predict(email_vector)[0]

        if prediction == "spam":
            st.error("🚨 Spam Email Detected")
        else:
            st.success("✅ Legitimate Email")

    else:
        st.warning("Please enter email text.")
