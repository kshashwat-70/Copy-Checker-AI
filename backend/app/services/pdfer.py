'''
{
    text: [],
    image: []
}
'''

import os
from dotenv import load_dotenv

load_dotenv()

file_path = os.environ.get("FILE_PATH")

import fitz  # PyMuPDF

def extract_text_and_images(stream):
    pdf = {"text": [], "image": []}
    
    doc = fitz.open("pdf", stream)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        pdf["text"].append(page.get_text())
        pdf["image"].append(page.get_images(full=True))

    return pdf
'''
# Usage
pdf = extract_text_and_images(file_path)
print("Extracted Text:", pdf["text"])
print("Extracted Images:", pdf["image"])
'''