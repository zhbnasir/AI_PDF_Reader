import streamlit as st
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from transformers import pipeline

# Function to extract text from a standard PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text

# Function to extract text from a scanned PDF using OCR
def extract_text_from_scanned_pdf(pdf_path, language="eng"):
    images = convert_from_path(pdf_path)
    text = ''
    for image in images:
        text += pytesseract.image_to_string(image, lang=language)
    return text

# Function to summarize text
def summarize_text(text, max_length=130):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Function to answer questions
def answer_question(text, question):
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    result = qa_pipeline(question=question, context=text)
    return result['answer']

# Streamlit app
st.set_page_config(page_title="AI PDF Reader", layout="wide")

# Sidebar
st.sidebar.title("AI PDF Reader")
st.sidebar.write("Upload a PDF to extract text, summarize, or ask questions.")

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

# Main content
if uploaded_file:
    st.write("### Extracted Text")
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text
    extracted_text = extract_text("temp.pdf")
    st.write(extracted_text)

    # Summarization
    if st.button("Summarize Text"):
        st.write("### Summary")
        summary = summarize_text(extracted_text)
        st.write(summary)

    # Question Answering
    st.write("### Ask a Question")
    question = st.text_input("Enter your question:")
    if question:
        answer = answer_question(extracted_text, question)
        st.write("### Answer")
        st.write(answer)
else:
    st.write("Please upload a PDF file to get started.")



# Sidebar for language selection
language = st.sidebar.selectbox("Select Language", ["English", "Afrikaans", "French", "Spanish", "German"])
language_code = {"English": "eng", "Afrikaans": "afr", "French": "fra", "Spanish": "spa", "German": "deu"}[language]

# Use the selected language for OCR
extracted_text = extract_text_from_scanned_pdf("temp.pdf", language=language_code)

