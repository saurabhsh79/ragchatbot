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
import tempfile

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Financial Multi-Modal QA", page_icon="💰", layout="wide")
st.title("💰 Financial Multi-Modal RAG System")

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
# CACHED MODEL LOADERS
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

# -------------------------
# FUNCTIONS
# -------------------------
def analyze_sentiment(text: str):
    result = sentiment_analyzer(text[:512])[0]
    label, score = result['label'], result['score']
    sentiment = "Positive" if label == "POSITIVE" else ("Negative" if label == "NEGATIVE" else "Neutral")
    return sentiment, round(score, 3)

def generate_chart_caption(image: Image.Image):
    inputs = blip_processor(images=image, return_tensors="pt")
    output = blip_model.generate(**inputs, max_new_tokens=40)
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
        pdf.multi_cell(0, 8, f"Q{idx+1}: {row['Question']}")
        pdf.multi_cell(0, 8, f"Answer: {row['Answer']}")
        pdf.multi_cell(0, 8, f"Sentiment: {row['Sentiment']} (Confidence: {row['Confidence']})")
        pdf.ln(5)
    return io.BytesIO(pdf.output(dest='S').encode('latin-1'))

# -------------------------
# SESSION STATE
# -------------------------
if "qa_history" not in st.session_state:
    st.session_state["qa_history"] = []

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📄 Documents", "🖼️ Charts", "❓ Q&A", "💾 Reports"])

# -------- TAB 1: DOCUMENT UPLOAD --------
with tab1:
    uploaded_file = st.file_uploader("Upload a financial PDF", type=["pdf"])
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            loader = PyPDFLoader(tmp_path)
            docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(loader.load())
            embeddings = OpenAIEmbeddings()
            st.session_state["vectorstore"] = FAISS.from_documents(docs, embeddings)
            st.success("✅ Document processed!")

# -------- TAB 2: CHART UPLOAD --------
with tab2:
    uploaded_images = st.file_uploader("Upload financial charts", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_images and st.button("Process Charts"):
        with st.spinner("Generating chart captions..."):
            image_captions = []
            for img in uploaded_images:
                img_obj = Image.open(img).convert("RGB")
                caption = generate_chart_caption(img_obj)
                image_captions.append((img.name, caption))
            chart_docs = [Document(page_content=cap, metadata={"source": name}) for name, cap in image_captions]
            if "vectorstore" in st.session_state:
                st.session_state["vectorstore"].add_documents(chart_docs)
            else:
                st.session_state["vectorstore"] = FAISS.from_documents(chart_docs, OpenAIEmbeddings())
            st.session_state["chart_captions"] = image_captions
            st.success("✅ Charts processed!")

# -------- TAB 3: QUESTION ANSWERING --------
with tab3:
    if "vectorstore" in st.session_state:
        question = st.text_input("Ask a financial question")
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
                st.subheader("📌 Answer")
                st.write(answer)
                st.write(f"**Sentiment:** {sentiment} (Confidence: {confidence})")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    title={'text': f"Sentiment: {sentiment}"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': {"Positive":"green","Negative":"red","Neutral":"gray"}.get(sentiment,"blue")}}))
                st.plotly_chart(fig, use_container_width=True)

                st.session_state["qa_history"].append({"Question": question, "Answer": answer,
                                                      "Sentiment": sentiment, "Confidence": confidence})

                with st.expander("🔍 Sources Used"):
                    for i, doc in enumerate(result["source_documents"], 1):
                        st.markdown(f"**Source {i}:** {doc.page_content[:300]}...")

                if "chart_captions" in st.session_state:
                    with st.expander("🖼️ Relevant Charts"):
                        for name, cap in st.session_state["chart_captions"]:
                            if any(word.lower() in cap.lower() for word in question.split()):
                                st.image(name, caption=f"{name}: {cap}", use_column_width=True)

# -------- TAB 4: REPORT DOWNLOAD --------
with tab4:
    if st.session_state["qa_history"]:
        df = pd.DataFrame(st.session_state["qa_history"])
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="financial_qa_report.csv", mime="text/csv")
        pdf_data = generate_pdf_report(df)
        st.download_button("⬇️ Download PDF", data=pdf_data, file_name="financial_qa_report.pdf", mime="application/pdf")
    else:
        st.info("No Q&A history yet.")
