import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from langchain.schema import Document

# Import our modules
from utils.data_fetcher import fetch_financial_data
from utils.rag_engine import process_pdf, add_docs_to_rag
from utils.sentiment_analysis import load_sentiment_model, analyze_sentiment
from utils.chart_analysis import load_blip2_model, generate_chart_caption
from utils.ai_summary import generate_ai_comparison_summary
from utils.pdf_report import generate_pdf_report
from utils.highlights_table import highlight_better

# --- PAGE CONFIG ---
st.set_page_config(page_title="Financial Multi-Modal RAG Dashboard", page_icon="💰", layout="wide")
st.title("💰 Financial Multi-Modal RAG Dashboard")

# --- API KEY ---
st.sidebar.header("⚙️ Settings")
OPENAI_API_KEY = st.sidebar.text_input("🔑 Enter OpenAI API Key:", type="password")
if not OPENAI_API_KEY:
    st.sidebar.warning("Please enter your OpenAI API Key.")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- LOAD MODELS ---
sentiment_analyzer = load_sentiment_model()
blip_processor, blip_model = load_blip2_model()

# --- SESSION STATE ---
if "qa_history" not in st.session_state:
    st.session_state["qa_history"] = []
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Documents", "🖼️ Charts", "📊 Compare", "❓ Q&A", "💾 Reports"])

# -------- TAB 1: DOCUMENT UPLOAD --------
with tab1:
    uploaded_file = st.file_uploader("Upload a financial PDF", type=["pdf"])
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing document..."):
            st.session_state["vectorstore"] = process_pdf(uploaded_file, st.session_state["vectorstore"])
            st.success("✅ Document processed!")

# -------- TAB 2: CHART UPLOAD --------
with tab2:
    uploaded_images = st.file_uploader("Upload financial charts", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_images and st.button("Process Charts"):
        with st.spinner("Generating chart captions..."):
            image_captions = []
            for img in uploaded_images:
                img_obj = Image.open(img).convert("RGB")
                caption = generate_chart_caption(img_obj, blip_processor, blip_model)
                image_captions.append((img.name, caption))
            chart_docs = [Document(page_content=cap, metadata={"source": name}) for name, cap in image_captions]
            st.session_state["vectorstore"] = add_docs_to_rag(chart_docs, st.session_state["vectorstore"])
            st.session_state["chart_captions"] = image_captions
            st.success("✅ Charts processed!")

# -------- TAB 3: LIVE FINANCIAL DATA (Compare Mode + Highlights Table + AI Summary) --------
with tab3:
    st.subheader("📊 Fetch & Compare Real-Time Financial Data")
    col_input1, col_input2 = st.columns(2)
    ticker1 = col_input1.text_input("Company 1 Symbol", value="AAPL")
    ticker2 = col_input2.text_input("Company 2 Symbol", value="MSFT")

    if st.button("Fetch & Compare"):
        with st.spinner(f"Fetching financial data for {ticker1} and {ticker2}..."):
            financial_text1, hist_chart1, kpis1 = fetch_financial_data(ticker1)
            financial_text2, hist_chart2, kpis2 = fetch_financial_data(ticker2)

            # KPI CARDS
            st.subheader("📌 Key Metrics Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### {ticker1}")
                for k, v in kpis1.items():
                    st.metric(k, f"{v}")
            with col2:
                st.markdown(f"### {ticker2}")
                for k, v in kpis2.items():
                    st.metric(k, f"{v}")

            # HIGHLIGHTS TABLE
            st.subheader("✅ Automatic Highlights (Better KPI Highlighted in Green)")
            compare_df = pd.DataFrame({
                "KPI": list(kpis1.keys()),
                f"{ticker1}": list(kpis1.values()),
                f"{ticker2}": list(kpis2.values())
            })
            st.dataframe(compare_df.style.apply(lambda df: highlight_better(df, ticker1, ticker2), axis=None))

            # PRICE CHART
            st.subheader("📈 Price Comparison")
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=hist_chart1["Date"], y=hist_chart1["Close"], mode="lines", name=f"{ticker1} Close"))
            fig_price.add_trace(go.Scatter(x=hist_chart2["Date"], y=hist_chart2["Close"], mode="lines", name=f"{ticker2} Close"))
            st.plotly_chart(fig_price, use_container_width=True)

            # AI SUMMARY
            st.subheader("🧠 AI-Generated Analyst Summary")
            summary = generate_ai_comparison_summary(ticker1, ticker2, kpis1, kpis2)
            st.write(summary)

            # Add to RAG
            docs_to_add = [
                Document(page_content=financial_text1, metadata={"source": f"YahooFinance_{ticker1}"}),
                Document(page_content=financial_text2, metadata={"source": f"YahooFinance_{ticker2}"})
            ]
            st.session_state["vectorstore"] = add_docs_to_rag(docs_to_add, st.session_state["vectorstore"])
            st.success(f"✅ {ticker1} and {ticker2} data added to RAG system!")

# -------- TAB 4: QUESTION ANSWERING --------
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

with tab4:
    if st.session_state["vectorstore"]:
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
                sentiment, confidence = analyze_sentiment(answer, sentiment_analyzer)
                st.subheader("📌 Answer")
                st.write(answer)
                st.write(f"**Sentiment:** {sentiment} (Confidence: {confidence})")
                st.session_state["qa_history"].append({
                    "Question": question, "Answer": answer,
                    "Sentiment": sentiment, "Confidence": confidence
                })
                with st.expander("🔍 Sources Used"):
                    for i, doc in enumerate(result["source_documents"], 1):
                        st.markdown(f"**Source {i}:** {doc.page_content[:300]}...")

# -------- TAB 5: REPORT DOWNLOAD --------
with tab5:
    if st.session_state["qa_history"]:
        df = pd.DataFrame(st.session_state["qa_history"])
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="financial_qa_report.csv", mime="text/csv")
        pdf_data = generate_pdf_report(df)
        st.download_button("⬇️ Download PDF", data=pdf_data, file_name="financial_qa_report.pdf", mime="application/pdf")
    else:
        st.info("No Q&A history yet.")
