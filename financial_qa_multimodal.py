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
# STREAMLIT PAGE SETUP
# -------------------------
st.set_page_config(page_title="Financial Multi-Modal QA System", page_icon="💰", layout="wide")
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
st.markdown('<p class="main-title">💰 Financial Multi-Modal QA System</p>', unsafe_allow_html=True)
st.write("Upload financial PDFs & charts, ask questions, get RAG-based answers, sentiment analysis, and download reports.")

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
st.sidebar.info("📌 Built with LangChain + FAISS + OpenAI + BLIP-2 + HuggingFace")

# -------------------------
# LOAD MODELS
# -------------------------
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_blip2_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
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
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption

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
    pdf_output = io.BytesIO(pdf_bytes)
    return pdf_output

# -------------------------
# SESSION STATE
# -------------------------
if "qa_history" not in st.session_state:
    st.session_state["qa_history"] = []

# -------------------------
# SECTION 1: DOCUMENT PROCESSING
# -------------------------
st.markdown('<p class="section-title">📄 1. Upload & Process Financial Document</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload PDF document:", type=["pdf"])
if st.button("Process Document") and uploaded_file:
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
# SECTION 1B: CHART PROCESSING
# -------------------------
st.markdown('<p class="section-title">🖼️ 1B. Upload Financial Charts/Graphs</p>', unsafe_allow_html=True)
uploaded_images = st.file_uploader("Upload charts or graphs (PNG/JPEG):", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
if st.button("Process Charts") and uploaded_images:
    with st.spinner("Processing charts with BLIP-2 and creating embeddings..."):
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
        st.success("✅ Charts processed and added to the knowledge base!")

# -------------------------
# SECTION 2: QUESTION ANSWERING
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

            # -------------------------
            # SECTION 3: SENTIMENT ANALYSIS
            # -------------------------
            st.markdown('<p class="section-title">📊 3. Sentiment Analysis</p>', unsafe_allow_html=True)
            sentiment, confidence = analyze_sentiment(answer)
            st.write(f"**Sentiment:** {sentiment} (Confidence: {confidence})")

            # Gauge chart
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

            # Save to history
            st.session_state["qa_history"].append({
                "Question": question,
                "Answer": answer,
                "Sentiment": sentiment,
                "Confidence": confidence
            })

            with st.expander("🔍 Sources Used"):
                for i, doc in enumerate(result["source_documents"], 1):
                    st.markdown(f"**Source {i}:** {doc.page_content[:500]}...")

            # Relevant Charts
            if "chart_captions" in st.session_state:
                with st.expander("🖼️ Relevant Charts"):
                    for name, cap in st.session_state["chart_captions"]:
                        if any(word.lower() in cap.lower() for word in question.split()):
                            st.image(name, caption=f"{name}: {cap}", use_column_width=True)

# -------------------------
# SECTION 4: EXPORT RESULTS
# -------------------------
if st.session_state["qa_history"]:
    st.markdown('<p class="section-title">💾 4. Download Your QA Report</p>', unsafe_allow_html=True)
    df = pd.DataFrame(st.session_state["qa_history"])
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", data=csv_data, file_name="financial_qa_report.csv", mime="text/csv")
    pdf_data = generate_pdf_report(df)
    st.download_button("⬇️ Download PDF", data=pdf_data, file_name="financial_qa_report.pdf", mime="application/pdf")
