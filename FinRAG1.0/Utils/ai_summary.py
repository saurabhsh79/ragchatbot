from langchain.chat_models import ChatOpenAI

def generate_ai_comparison_summary(ticker1, ticker2, kpis1, kpis2):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    prompt = f"""
    You are a financial analyst. Compare the following two companies and give a concise 5-sentence summary:

    Company 1 ({ticker1}): {kpis1}
    Company 2 ({ticker2}): {kpis2}

    Focus on valuation, growth potential, and overall market strength.
    """
    return llm.predict(prompt)
