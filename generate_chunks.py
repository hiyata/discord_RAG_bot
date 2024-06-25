import os
import fitz  
import json
from dotenv import load_dotenv
import nltk

# Use punkt for tokenization
nltk.download('punkt')

# Load path and chunks
def load_env_variables():
    load_dotenv()
    global PDF_FILE_PATH, CHUNKS_DIR
    PDF_FILE_PATH = os.getenv('PDF_FILE_PATH')
    CHUNKS_DIR = 'pdf_chunks'

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        raise

# Split text
def split_text_by_sentences(text):
    return nltk.sent_tokenize(text)

def split_text_by_paragraphs(text):
    return text.split('\n\n')

def split_text_with_overlap(text, chunk_size=1000, overlap=200):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

# Save to json
def save_chunks(chunks, file_path):
    with open(file_path, 'w') as file:
        json.dump(chunks, file, indent=2)

# Load pdf path and chunks directory
load_env_variables()

if not PDF_FILE_PATH:
    raise ValueError("PDF_FILE_PATH environment variable is not set. Please check your .env file.")

# Create directory for storing chunks if it doesn't exist
os.makedirs(CHUNKS_DIR, exist_ok=True)

# Extract text from the PDF
pdf_text = extract_text_from_pdf(PDF_FILE_PATH)

# Split text into different levels
sentence_chunks = split_text_by_sentences(pdf_text)
paragraph_chunks = split_text_by_paragraphs(pdf_text)
document_chunks = [pdf_text]  # Entire document as a single chunk

# Save the chunks to JSON files
save_chunks(sentence_chunks, os.path.join(CHUNKS_DIR, 'sentence_chunks.json'))
save_chunks(paragraph_chunks, os.path.join(CHUNKS_DIR, 'paragraph_chunks.json'))
save_chunks(document_chunks, os.path.join(CHUNKS_DIR, 'document_chunks.json'))

print(f"Chunks have been successfully saved in the '{CHUNKS_DIR}' directory.")