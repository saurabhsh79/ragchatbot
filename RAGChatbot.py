import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import streamlit as st

# Step 1: Data Collection & Preprocessing
def load_financial_data():
    # Simulate loading financial data (Replace with real data loading)
    data = {
        "Year": ["2023", "2022"],
        "Revenue": ["$50B", "$45B"],
        "Net Income": ["$10B", "$8B"],
    }
    df = pd.DataFrame(data)
    return df

financial_data = load_financial_data()

# Step 2: Chunking & Embedding
class ChunkedFinancialData:
    def __init__(self, df, chunk_size=2):
        self.df = df
        self.chunk_size = chunk_size
        self.chunks = self.create_chunks()
    
    def create_chunks(self):
        text_chunks = []
        for i in range(0, len(self.df), self.chunk_size):
            chunk = " ".join(self.df.iloc[i].astype(str))
            text_chunks.append(chunk)
        return text_chunks

    def embed_chunks(self):
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = model.encode(self.chunks, convert_to_tensor=True)
        return embeddings

chunked_data = ChunkedFinancialData(financial_data)
embeddings = chunked_data.embed_chunks()

# Step 3: Advanced Retrieval (BM25 + Embeddings)
bm25 = BM25Okapi([chunk.split() for chunk in chunked_data.chunks])
vector_store = FAISS(HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2"))

# Populate Vector Store
for i, chunk in enumerate(chunked_data.chunks):
    vector_store.add_texts([chunk])

# Step 4: Query Processing (Adaptive Retrieval)
def retrieve_answer(query):
    query_embedding = vector_store.embed_query(query)
    faiss_results = vector_store.similarity_search(query, k=3)
    bm25_results = bm25.get_top_n(query.split(), chunked_data.chunks, n=3)
    
    combined_results = list(set(faiss_results + bm25_results))
    return combined_results

# Step 5: UI Development with Streamlit
st.title("Financial RAG Chatbot")
user_query = st.text_input("Ask a financial question:")
if user_query:
    response = retrieve_answer(user_query)
    st.write("Answer:", response)

# Step 6: Guardrails
BAD_WORDS = ["hack", "attack", "fraud"]
if any(bad_word in user_query.lower() for bad_word in BAD_WORDS):
    st.error("Invalid Query: Contains prohibited terms.")

# Step 7: Testing & Validation
# Expected test cases will be manually tested through UI
