import streamlit as st

def validate_input(user_input):
    """Validation logic for user input."""
    if not user_input:
        return False, "Input cannot be empty. Please enter some text."
    elif len(user_input) < 5:
        return False, "Input must be at least 5 characters long."
    return True, "Valid input! Your message is accepted."

# Streamlit UI
st.title("RAG Financial Chatbot")

# Text Input Box
user_input = st.text_input("Enter a message:")

# Submit Button
if st.button("Submit"):
    is_valid, message = validate_input(user_input)

    # Show custom messages
    if is_valid:
        st.success(message)
    else:
        st.error(message)

        

