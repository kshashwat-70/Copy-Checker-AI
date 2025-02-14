'''
{
    text: [],
    image: []
}
'''

import os
from dotenv import load_dotenv
from bson import Binary
from io import BytesIO
load_dotenv()

file_path = os.environ.get("FILE_PATH")

import fitz  # PyMuPDF

def extract_text_and_images(stream):
    if not isinstance(stream, BytesIO):
        raise ValueError("Expected a BytesIO object")
    pdf = {"text": [], "image": []}
    
    doc = fitz.open("pdf", stream)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        pdf["text"].append(page.get_text())
        binr = Binary(page.get_images(full=True))
        pdf["image"].append(binr)

    return pdf
'''
# Usage
pdf = extract_text_and_images(file_path)
print("Extracted Text:", pdf["text"])
print("Extracted Images:", pdf["image"])
'''