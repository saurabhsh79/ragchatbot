import yfinance as yf
import pandas as pd
from datetime import date
import streamlit as st

@st.cache_data(ttl=86400)
def fetch_financial_data(ticker: str):
    stock = yf.Ticker(ticker)
    info = stock.info
    kpis = {
        "Current Price": info.get("currentPrice", "N/A"),
        "Market Cap": info.get("marketCap", "N/A"),
        "PE Ratio": info.get("trailingPE", "N/A"),
        "Dividend Yield": info.get("dividendYield", "N/A"),
        "52-Week High": info.get("fiftyTwoWeekHigh", "N/A"),
        "52-Week Low": info.get("fiftyTwoWeekLow", "N/A"),
    }
    income_stmt = stock.financials.fillna(0).astype(str).to_dict()
    balance_sheet = stock.balance_sheet.fillna(0).astype(str).to_dict()
    cash_flow = stock.cashflow.fillna(0).astype(str).to_dict()

    financial_text = (
        f"Basic Info (as of {date.today()}): {ticker} ({info.get('longName','')})\n"
        f"Market Cap: {kpis['Market Cap']}, PE Ratio: {kpis['PE Ratio']}, Current Price: {kpis['Current Price']}\n\n"
        f"Income Statement: {income_stmt}\n\nBalance Sheet: {balance_sheet}\n\nCash Flow: {cash_flow}"
    )

    hist = stock.history(period="6mo")
    hist_chart = hist[["Close", "Volume"]].reset_index()
    return financial_text, hist_chart, kpis
