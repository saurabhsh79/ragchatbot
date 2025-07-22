import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from transformers import pipeline

# -------------------------
# STREAMLIT UI SETUP
# -------------------------
st.set_page_config(page_title="Financial QA System", page_icon="💰", layout="wide")
st.title("💰 Financial Document Question Answering System with Sentiment Analysis (RAG)")

st.markdown("""
Upload financial documents (PDFs), ask financial questions, and get:
1. **RAG-based answer**
2. **Sentiment of the answer (Positive / Negative / Neutral)**
""")

# -------------------------
# OPENAI API KEY
# -------------------------
OPENAI_API_KEY = st.sidebar.text_input("🔑 Enter OpenAI API Key:", type="password")
if not OPENAI_API_KEY:
    st.warning("Please enter your OpenAI API Key to proceed.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# -------------------------
# SENTIMENT ANALYSIS PIPELINE
# -------------------------
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_analyzer = load_sentiment_model()

def analyze_sentiment(text: str):
    result = sentiment_analyzer(text[:512])[0]  # Limit to first 512 tokens for speed
    label = result['label']
    score = result['score']
    sentiment = "Positive" if label == "POSITIVE" else ("Negative" if label == "NEGATIVE" else "Neutral")
    return sentiment, round(score, 3)

# -------------------------
# DOCUMENT UPLOAD
# -------------------------
uploaded_file = st.file_uploader("📄 Upload Financial Document (PDF only):", type=["pdf"])
process_doc = st.button("Process Document")

if uploaded_file and process_doc:
    with st.spinner("Processing document..."):
        temp_file = f"temp_{uploaded_file.name}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load PDF and extract text
        loader = PyPDFLoader(temp_file)
        documents = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        # Create embeddings and store in FAISS
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        st.session_state["vectorstore"] = vectorstore

        st.success("✅ Document processed and indexed successfully!")

# -------------------------
# QUESTION ANSWERING + SENTIMENT
# -------------------------
if "vectorstore" in st.session_state:
    question = st.text_input("❓ Ask a financial question based on the uploaded document:")

    if st.button("Get Answer"):
        with st.spinner("Generating answer..."):
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model_name="gpt-4o", temperature=0),
                chain_type="stuff",
                retriever=st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            result = qa_chain({"query": question})
            answer = result["result"]

            # Display Answer
            st.subheader("📌 Answer")
            st.write(answer)

            # Sentiment Analysis
            sentiment, confidence = analyze_sentiment(answer)
            st.subheader("📊 Sentiment Analysis")
            st.write(f"**Sentiment:** {sentiment} (Confidence: {confidence})")

            with st.expander("🔍 Sources Used"):
                for doc in result["source_documents"]:
                    st.write(f"📄 **Page Content:** {doc.page_content[:500]}...")

else:
    st.info("⬆️ Please upload and process a financial document first.")
