import os
from dotenv import load_dotenv
from pdfminer.high_level import extract_text

load_dotenv()

def pdf_to_text(pdf_path):
    return extract_text(pdf_path)

def convert_pdfs_to_text(pdf_dir, text_dir):
    os.makedirs(text_dir, exist_ok=True)
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            text = pdf_to_text(pdf_path)
            text_file = os.path.splitext(pdf_file)[0] + '.txt'
            text_path = os.path.join(text_dir, text_file)
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Converted {pdf_file} to text.")

