import streamlit as st

st.title("Financial RAG Chatbot")
user_query = st.text_input("Ask a financial question:")
if user_query:
    response = retrieve_answer(user_query)
    st.write("Answer:", response)
