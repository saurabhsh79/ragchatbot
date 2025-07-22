import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# -------------------------
# STREAMLIT UI SETUP
# -------------------------
st.set_page_config(page_title="Financial QA System", page_icon="💰", layout="wide")
st.title("💰 Financial Document Question Answering System (RAG)")

st.markdown("""
Upload financial documents (PDFs or text), and ask financial questions.
The system will retrieve the most relevant sections and generate answers using RAG.
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
# DOCUMENT UPLOAD
# -------------------------
uploaded_file = st.file_uploader("📄 Upload Financial Document (PDF only):", type=["pdf"])
process_doc = st.button("Process Document")

if uploaded_file and process_doc:
    with st.spinner("Processing document..."):
        # Save uploaded file temporarily
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
# QUESTION ANSWERING
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
            st.subheader("📌 Answer")
            st.write(result["result"])

            with st.expander("🔍 Sources Used"):
                for doc in result["source_documents"]:
                    st.write(f"📄 **Page Content:** {doc.page_content[:500]}...")

else:
    st.info("⬆️ Please upload and process a financial document first.")
