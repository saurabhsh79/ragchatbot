import streamlit as st
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Multi Modal RAG System with Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
        .stApp {
            background-color: #f8f9fa;
        }
        .main > div {
            padding: 2rem;
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #1a1a1a;
        }
        .stButton>button {
            background-color: #0066cc;
            color: white;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.title("📊 Multi Modal RAG System with Sentiment Analysis")
st.markdown(
    "A professional tool to extract and analyze financial insights from documents and images using Multimodal RAG and sentiment analysis."
)
st.markdown("---")

# --- Tabs for Input and Output ---
tab1, tab2 = st.tabs(["📥 Input Data", "🔎 Output Analysis"])

# --- Tab 1: Input ---
with tab1:
    st.header("Upload Inputs and Ask Your Question")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Upload Financial Document")
        doc_file = st.file_uploader("Document File", type=["pdf", "docx", "txt"], label_visibility="collapsed")

    with col2:
        st.subheader("🖼️ Upload Financial Chart or Graph")
        image_file = st.file_uploader("Image File", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

        # Optional preview
        if image_file is not None:
            st.image(Image.open(image_file), caption="Uploaded Chart/Graph", use_column_width=True)

    st.subheader("❓ Ask a Financial Question")
    question = st.text_input("Your question", placeholder="e.g., What is the revenue trend in Q2 earnings report?")

    st.subheader("🤖 Select Model")
    model_choice = st.selectbox(
        "Choose a Multimodal RAG Model",
        options=["MiniGPT-4", "BLIP-2", "GIT + RAG", "Custom Fine-tuned Model"]
    )

    generate = st.button("Generate Answer and Analyze Sentiment")

# --- Tab 2: Output ---
with tab2:
    st.header("📊 Answer and Sentiment Insights")

    if generate:
        # Simulated output for prototype
        st.subheader("📌 Answer")
        st.markdown(
            """
            **Q:** What is the revenue trend in Q2 earnings report?  
            **A:** Based on the uploaded financial chart and Q2 report, the company’s revenue increased by **12.5%** compared to Q1, indicating a **positive growth trend** driven by increased product demand and operational efficiency.
            """
        )

        st.subheader("🧠 Sentiment Analysis")
        st.markdown("**Overall Sentiment:** 📈 Positive")
        st.progress(85)  # Simulated sentiment positivity score

        st.info("💡 The response indicates strong financial performance with positive market indicators.")
    else:
        st.warning("⚠️ Please upload files, enter a question, and click the button in the Input tab to see results here.")

# Footer
st.markdown("---")
st.caption("🔍 Built with Streamlit • AI-powered Financial Document Intelligence • 2025")
