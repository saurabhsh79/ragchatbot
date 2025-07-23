import streamlit as st
import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from transformers import pipeline
import plotly.graph_objects as go
from fpdf import FPDF
import io

# -------------------------
# STREAMLIT PAGE SETUP
# -------------------------
st.set_page_config(page_title="Financial QA System", page_icon="💰", layout="wide")

st.markdown(
    """
    <style>
        .main-title {
            font-size:40px !important;
            color:#004080;
            font-weight:bold;
        }
        .section-title {
            font-size:24px !important;
            color:#003366;
            font-weight:bold;
            margin-top:20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-title">💰 Financial Document QA System with Sentiment Analysis</p>', unsafe_allow_html=True)
st.write("Upload financial documents, ask questions, and get RAG-based answers with sentiment analysis. You can also export the results.")

# -------------------------
# SIDEBAR SETTINGS
# -------------------------
st.sidebar.header("⚙️ Settings")
OPENAI_API_KEY = st.sidebar.text_input("🔑 Enter OpenAI API Key:", type="password")

if not OPENAI_API_KEY:
    st.sidebar.warning("Please enter your OpenAI API Key.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.sidebar.markdown("---")
st.sidebar.info("📌 Built with LangChain + FAISS + OpenAI + HuggingFace")

# -------------------------
# SENTIMENT ANALYSIS MODEL
# -------------------------
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_analyzer = load_sentiment_model()

def analyze_sentiment(text: str):
    result = sentiment_analyzer(text[:512])[0]  # Limit to 512 tokens
    label = result['label']
    score = result['score']
    sentiment = "Positive" if label == "POSITIVE" else ("Negative" if label == "NEGATIVE" else "Neutral")
    return sentiment, round(score, 3)

# -------------------------
# PDF GENERATION FUNCTION
# -------------------------
def generate_pdf_report(data):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Financial QA Report", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    for idx, row in data.iterrows():
        pdf.ln(8)
        pdf.set_font("Arial", "B", 12)
        pdf.multi_cell(0, 8, f"Q{idx+1}: {row['Question']}")
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, f"Answer: {row['Answer']}")
        pdf.multi_cell(0, 8, f"Sentiment: {row['Sentiment']} (Confidence: {row['Confidence']})")
        pdf.ln(5)

    # ✅ FIX: Use dest='S' to get PDF as string, then wrap in BytesIO
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    pdf_output = io.BytesIO(pdf_bytes)
    return pdf_output


# -------------------------
# SESSION STATE TO STORE Q&A
# -------------------------
if "qa_history" not in st.session_state:
    st.session_state["qa_history"] = []

# -------------------------
# SECTION 1: DOCUMENT PROCESSING
# -------------------------
st.markdown('<p class="section-title">📄 1. Upload & Process Financial Document</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload PDF document:", type=["pdf"])
process_doc = st.button("Process Document")

if uploaded_file and process_doc:
    with st.spinner("Processing document and creating embeddings..."):
        temp_file = f"temp_{uploaded_file.name}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(temp_file)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        st.session_state["vectorstore"] = vectorstore

        st.success("✅ Document processed and indexed successfully!")

# -------------------------
# SECTION 2: QUESTION ANSWERING + SENTIMENT
# -------------------------
if "vectorstore" in st.session_state:
    st.markdown('<p class="section-title">❓ 2. Ask Financial Questions</p>', unsafe_allow_html=True)
    question = st.text_input("Enter your financial question:")

    if st.button("Get Answer"):
        with st.spinner("Generating answer using RAG..."):
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model_name="gpt-4o", temperature=0),
                chain_type="stuff",
                retriever=st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            result = qa_chain({"query": question})
            answer = result["result"]

            st.subheader("📌 Answer")
            st.write(answer)

            # Sentiment Analysis
            st.markdown('<p class="section-title">📊 3. Sentiment Analysis of the Answer</p>', unsafe_allow_html=True)
            sentiment, confidence = analyze_sentiment(answer)
            st.write(f"**Sentiment:** {sentiment} (Confidence: {confidence})")

            # Plot Sentiment Confidence Gauge
            sentiment_colors = {"Positive": "green", "Negative": "red", "Neutral": "gray"}
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                title={'text': f"Sentiment: {sentiment}"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': sentiment_colors.get(sentiment, "blue")},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 100], 'color': "white"}]}))
            st.plotly_chart(fig, use_container_width=True)

            # Save to Session History
            st.session_state["qa_history"].append({
                "Question": question,
                "Answer": answer,
                "Sentiment": sentiment,
                "Confidence": confidence
            })

            with st.expander("🔍 Sources Used"):
                for i, doc in enumerate(result["source_documents"], 1):
                    st.markdown(f"**Source {i}:** {doc.page_content[:500]}...")

# -------------------------
# SECTION 4: EXPORT RESULTS
# -------------------------
if st.session_state["qa_history"]:
    st.markdown('<p class="section-title">💾 4. Download Your QA Report</p>', unsafe_allow_html=True)
    df = pd.DataFrame(st.session_state["qa_history"])

    # CSV Download
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", data=csv_data, file_name="financial_qa_report.csv", mime="text/csv")

    # PDF Download
    pdf_data = generate_pdf_report(df)
    st.download_button("⬇️ Download PDF", data=pdf_data, file_name="financial_qa_report.pdf", mime="application/pdf")
