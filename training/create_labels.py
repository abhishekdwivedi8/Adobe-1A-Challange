# This is a helper script to make labeling easier.
# It reads a PDF and prints all text blocks on a specific page.
# You can then copy-paste the exact text into your labeled_data.json file.

# You'll need to install PyMuPDF: pip install PyMuPDF
import fitz  # PyMuPDF
import os

# --- Configuration ---
# PDF file you want to label from your sample_pdfs folder
PDF_FILE_NAME = "r_abhishek_online_tech_Dev.pdf" 
# The page number you want to look at
PAGE_NUMBER = 1  # Page numbers start from 1

# --- Main Script ---
pdf_path = os.path.join("sample_pdfs", PDF_FILE_NAME)

try:
    doc = fitz.open(pdf_path)
    print(f"--- Text blocks from '{PDF_FILE_NAME}', page {PAGE_NUMBER} ---")
    
    # Page numbers in PyMuPDF are 0-indexed, so we subtract 1
    page = doc.load_page(PAGE_NUMBER - 1) 
    
    # Get all text blocks from the page
    text_blocks = page.get_text("blocks")
    
    # Sort blocks by their vertical position on the page
    text_blocks.sort(key=lambda block: block[1])
    
    for i, block in enumerate(text_blocks):
        # block[4] contains the text content
        block_text = block[4].strip().replace('\n', ' ')
        print(f"Block {i+1}:")
        print(f"'{block_text}'")
        print("-" * 20)

    doc.close()

except FileNotFoundError:
    print(f"Error: The file '{pdf_path}' was not found.")
    print("Please make sure the file name is correct and it's inside the 'training/sample_pdfs/' folder.")
except Exception as e:
    print(f"An error occurred: {e}")