import streamlit as st

# Streamlit UI - Saurabh
st.title("RAG Financial Chatbot")

# User input - Saurabh
query = st.text_input("Enter your query:")

# Submit Button
if st.button("Submit"):
    if user_input:
        # Call backend API
        response = requests.post(BACKEND_URL, json={"input": user_input})
        

