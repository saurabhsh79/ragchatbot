import streamlit as st
import os
import io
import pandas as pd
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
import plotly.graph_objects as go
from fpdf import FPDF

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Financial Multi-Modal QA System", page_icon="💰", layout="wide")

# -------------------------
# CUSTOM CSS
# -------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size:38px !important;
            color:#003366;
            font-weight:bold;
            text-align:center;
        }
        .card {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-title">💰 Financial Multi-Modal QA System</p>', unsafe_allow_html=True)
st.write("Upload financial PDFs & charts, ask questions, analyze sentiments, and export results.")

# -------------------------
# SIDEBAR SETTINGS
# -------------------------
st.sidebar.header("⚙️ Settings")
OPENAI_API_KEY = st.sidebar.text_input("🔑 Enter OpenAI API Key:", type="password")
if not OPENAI_API_KEY:
    st.sidebar.warning("Please enter your OpenAI API Key.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# -------------------------
# LOAD MODELS
# -------------------------
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_blip2_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")  # lightweight model
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

sentiment_analyzer = load_sentiment_model()
blip_processor, blip_model = load_blip2_model()

def analyze_sentiment(text: str):
    result = sentiment_analyzer(text[:512])[0]
    label = result['label']
    score = result['score']
    sentiment = "Positive" if label == "POSITIVE" else ("Negative" if label == "NEGATIVE" else "Neutral")
    return sentiment, round(score, 3)

def generate_chart_caption(image: Image.Image):
    inputs = blip_processor(images=image, return_tensors="pt")
    output = blip_model.generate(**inputs, max_new_tokens=50)
    return blip_processor.decode(output[0], skip_special_tokens=True)

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
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return io.BytesIO(pdf_bytes)

# -------------------------
# SESSION STATE
# -------------------------
if "qa_history" not in st.session_state:
    st.session_state["qa_history"] = []

# -------------------------
# TABS LAYOUT
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Upload", "🖼️ Chart Upload", "❓ Ask Questions", "💾 Reports"])

# -------------------------
# TAB 1: DOCUMENT UPLOAD
# -------------------------
with tab1:
    st.subheader("📄 Upload Financial Documents")
    uploaded_file = st.file_uploader("Upload PDF document:", type=["pdf"])
    if st.button("Process Document"):
        if uploaded_file:
            with st.spinner("Processing document and creating embeddings..."):
                temp_file = f"temp_{uploaded_file.name}"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader(temp_file)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(documents)
                embeddings = OpenAIEmbeddings()
                st.session_state["vectorstore"] = FAISS.from_documents(docs, embeddings)
                st.success("✅ Document processed and indexed successfully!")

# -------------------------
# TAB 2: CHART UPLOAD
# -------------------------
with tab2:
    st.subheader("🖼️ Upload Financial Charts/Graphs")
    uploaded_images = st.file_uploader("Upload charts (PNG/JPEG):", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if st.button("Process Charts"):
        if uploaded_images:
            with st.spinner("Processing charts with BLIP-2..."):
                image_captions = []
                for img_file in uploaded_images:
                    image = Image.open(img_file).convert("RGB")
                    caption = generate_chart_caption(image)
                    image_captions.append((img_file.name, caption))
                chart_docs = [Document(page_content=cap, metadata={"source": name}) for name, cap in image_captions]
                if "vectorstore" in st.session_state:
                    st.session_state["vectorstore"].add_documents(chart_docs)
                else:
                    embeddings = OpenAIEmbeddings()
                    st.session_state["vectorstore"] = FAISS.from_documents(chart_docs, embeddings)
                st.session_state["chart_captions"] = image_captions
                st.success("✅ Charts processed and added!")

# -------------------------
# TAB 3: QUESTION ANSWERING
# -------------------------
with tab3:
    st.subheader("❓ Ask Financial Questions")
    if "vectorstore" in st.session_state:
        question = st.text_input("Enter your financial question:")
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

                sentiment, confidence = analyze_sentiment(answer)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"**📌 Answer:** {answer}")
                st.markdown(f"**Sentiment:** {sentiment} (Confidence: {confidence})")
                st.markdown('</div>', unsafe_allow_html=True)

                # Sentiment Gauge
                sentiment_colors = {"Positive": "green", "Negative": "red", "Neutral": "gray"}
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    title={'text': f"Sentiment: {sentiment}"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': sentiment_colors.get(sentiment, "blue")},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                     {'range': [50, 100], 'color': "white"}]}))
                st.plotly_chart(fig, use_container_width=True)

                # Save to History
                st.session_state["qa_history"].append({
                    "Question": question,
                    "Answer": answer,
                    "Sentiment": sentiment,
                    "Confidence": confidence
                })

                with st.expander("🔍 Sources Used"):
                    for i, doc in enumerate(result["source_documents"], 1):
                        st.markdown(f"**Source {i}:** {doc.page_content[:500]}...")

                if "chart_captions" in st.session_state:
                    with st.expander("🖼️ Relevant Charts"):
                        for name, cap in st.session_state["chart_captions"]:
                            if any(word.lower() in cap.lower() for word in question.split()):
                                st.image(name, caption=f"{name}: {cap}", use_column_width=True)

# -------------------------
# TAB 4: REPORTS
# -------------------------
with tab4:
    st.subheader("💾 Download Your QA Report")
    if st.session_state["qa_history"]:
        df = pd.DataFrame(st.session_state["qa_history"])
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", data=csv_data, file_name="financial_qa_report.csv", mime="text/csv")
        pdf_data = generate_pdf_report(df)
        st.download_button("⬇️ Download PDF", data=pdf_data, file_name="financial_qa_report.pdf", mime="application/pdf")
    else:
        st.info("No Q&A history available yet.")
