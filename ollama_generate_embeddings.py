import os
import fitz  # PyMuPDF
import joblib
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import nltk

# Ensure NLTK resources are available
nltk.download('punkt')

# Function to load environment variables
def load_env_variables():
    load_dotenv()
    global PDF_FILE_PATH, OLLAMA_API_URL, OLLAMA_MODEL_NAME, EMBEDDINGS_DIR
    PDF_FILE_PATH = os.getenv('PDF_FILE_PATH')
    OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://127.0.0.1:11434')
    OLLAMA_MODEL_NAME = os.getenv('OLLAMA_MODEL_NAME', 'llama3')
    EMBEDDINGS_DIR = 'pdf_chunks'

# Function to extract text from PDF
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

# Functions to split text
def split_text_by_sentences(text):
    return nltk.sent_tokenize(text)

def split_text_by_paragraphs(text):
    return text.split('\n\n')

def split_text_with_overlap(text, chunk_size=1000, overlap=200):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

# Function to get embeddings from Ollama
def get_embedding_from_ollama(text, model_name):
    url = f"{OLLAMA_API_URL}/api/embeddings"
    headers = {"Content-Type": "application/json"}
    data = {"model": model_name, "prompt": text}
    
    response = requests.post(url, headers=headers, json=data)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
        raise
    return response.json()["embedding"]

# Functions to generate, store, and load embeddings
def generate_embeddings_for_chunks(chunks, model_name):
    embeddings = []
    for index, chunk in enumerate(chunks):
        print(f"Processing chunk {index + 1}/{len(chunks)}")
        try:
            embedding = get_embedding_from_ollama(chunk, model_name)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Failed to get embedding for chunk {index + 1}: {e}")
            embeddings.append([0]*768)  # Assuming embedding size is 768, use zeros as placeholder
        print(f"Completed chunk {index + 1}/{len(chunks)}")
    return embeddings

def store_embeddings(embeddings, file_path):
    with open(file_path, 'wb') as file:
        joblib.dump(embeddings, file)

def load_embeddings(file_path):
    with open(file_path, 'rb') as file:
        return joblib.load(file)

# Load or create all levels of embeddings
def load_or_create_all_embeddings(pdf_path, embeddings_dir, model_name):
    os.makedirs(embeddings_dir, exist_ok=True)

    # Extract text
    pdf_text = extract_text_from_pdf(pdf_path)

    # Split text into different levels
    sentence_chunks = split_text_by_sentences(pdf_text)
    paragraph_chunks = split_text_by_paragraphs(pdf_text)
    document_chunks = [pdf_text]

    # Generate and store/load embeddings for each level
    levels = ["sentence", "paragraph", "document"]
    all_embeddings = {}

    for level, chunks in zip(levels, [sentence_chunks, paragraph_chunks, document_chunks]):
        embeddings_file_path = os.path.join(embeddings_dir, f"{level}_embeddings.pkl")
        if os.path.exists(embeddings_file_path):
            embeddings = load_embeddings(embeddings_file_path)
        else:
            embeddings = generate_embeddings_for_chunks(chunks, model_name)
            store_embeddings(embeddings, embeddings_file_path)
        all_embeddings[level] = (embeddings, chunks)

    return all_embeddings

# Main script execution
if __name__ == "__main__":
    load_env_variables()

    if not PDF_FILE_PATH:
        raise ValueError("PDF_FILE_PATH environment variable is not set. Please check your .env file.")

    all_embeddings = load_or_create_all_embeddings(PDF_FILE_PATH, EMBEDDINGS_DIR, OLLAMA_MODEL_NAME)
    print(f"Embeddings have been successfully generated and saved in the '{EMBEDDINGS_DIR}' directory.")