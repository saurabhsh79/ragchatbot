import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def process_pdf(uploaded_file, vectorstore):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
    embeddings = OpenAIEmbeddings()

    if vectorstore is None:
        vectorstore = FAISS.from_documents(docs, embeddings)
    else:
        vectorstore.add_documents(docs)
    return vectorstore

def add_docs_to_rag(docs_to_add, vectorstore):
    embeddings = OpenAIEmbeddings()
    if vectorstore is None:
        vectorstore = FAISS.from_documents(docs_to_add, embeddings)
    else:
        vectorstore.add_documents(docs_to_add)
    return vectorstore
