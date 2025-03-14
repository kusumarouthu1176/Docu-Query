from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_bytes):
    """Extract text from a PDF file (received as bytes)."""
    reader = PdfReader(pdf_bytes)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

