import streamlit as st
import requests
from newspaper import Article
from readability import Document
import requests as req
from bs4 import BeautifulSoup
import lxml.html.clean

st.set_page_config(page_title="News Bias Analyzer", layout="centered")

st.title("📰 News Bias Analyzer")
st.write("Paste a **news article URL** or the **raw article text** below to get a summary and political bias analysis.")

# Input fields
url_input = st.text_input("🔗 Article URL (leave blank if pasting text)", "")
text_input = st.text_area("📝 Or paste full article text here", height=300)

# Initialize content
content = ""

# Try to extract article from URL
if url_input:
    try:
        with st.spinner("Fetching article using newspaper3k..."):
            article = Article(url_input)
            article.download()
            article.parse()
            if article.text.strip():
                content = article.text
                st.success("✅ Article extracted with newspaper3k.")
            else:
                st.warning("⚠️ newspaper3k failed. Trying fallback parser...")
                # Fallback: readability + BeautifulSoup
                response = req.get(url_input, timeout=10)
                doc = Document(response.text)
                html = doc.summary()
                soup = BeautifulSoup(html, "html.parser")
                content = soup.get_text()
                if content.strip():
                    st.success("✅ Article extracted with fallback parser.")
                else:
                    st.warning("⚠️ Could not extract article text from the URL.")
    except Exception as e:
        st.error(f"❌ Failed to extract article: {e}")

# Fallback: Use manually pasted text
elif text_input:
    content = text_input

# Display extracted content (for confirmation)
if content.strip():
    st.text_area("🧾 Extracted Article Text", content, height=200)

# Analyze article content
if st.button("Analyze"):
    if not content.strip():
        st.warning("⚠️ Please enter a valid URL or article text.")
    else:
        with st.spinner("Summarizing and analyzing..."):
            try:
                response = requests.post(
                    "http://backend:8000/analyze",
                    json={"content": content, "source_url": url_input}
                )
                if response.status_code == 200:
                    result = response.json()

                    st.subheader("📝 Summary")
                    st.write(result.get("summary", "No summary returned."))

                    st.subheader("📊 Bias Prediction")
                    bias = result.get("bias", "Unknown")
                    source_bias = result.get("source_bias", "Unknown")
                    bias_color = {"Left": "blue", "Right": "red", "Neutral": "gray", "Unknown": "black"}
                    st.markdown(f"<h3 style='color:{bias_color.get(bias, 'black')}'>{bias}</h3>", unsafe_allow_html=True)

                    if source_bias != "Unknown":
                        st.write(f"🗞️ Source Bias (based on domain): `{source_bias}`")
                else:
                    st.error("❌ Backend error. Please ensure FastAPI is running on http://localhost:8000.")
            except Exception as e:
                st.error(f"❌ Error connecting to backend: {e}")
