from fpdf import FPDF
import io

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
