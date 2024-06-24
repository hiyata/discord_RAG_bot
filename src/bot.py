import discord
import fitz  # PyMuPDF
import os
import json
from dotenv import load_dotenv
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Function to load environment variables
def load_env_variables():
    load_dotenv()
    global PDF_FILE_PATH, DISCORD_BOT_TOKEN, OLLAMA_API_URL, BLOCKED_USERNAMES, EMBEDDINGS_DIR
    PDF_FILE_PATH = os.getenv('PDF_FILE_PATH')
    DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
    OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://127.0.0.1:11434')
    BLOCKED_USERNAMES = os.getenv('BLOCKED_USERNAMES', '').split(',')
    EMBEDDINGS_DIR = 'pdf_chunks'

# Load initial environment variables
load_env_variables()

if not PDF_FILE_PATH or not DISCORD_BOT_TOKEN:
    raise ValueError("One or more environment variables are not set. Please check your .env file.")

# Initialize the Discord client
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
discord_client = discord.Client(intents=intents)

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
    import nltk
    nltk.download('punkt')
    return nltk.sent_tokenize(text)

def split_text_by_paragraphs(text):
    return text.split('\n\n')

def split_text_with_overlap(text, chunk_size=1000, overlap=200):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

# Function to get embeddings from Ollama
def get_embedding_from_ollama(text, model_name="llama3"):
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
def generate_embeddings_for_chunks(chunks, model_name="llama3"):
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
def load_or_create_all_embeddings(pdf_path, embeddings_dir):
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
            embeddings = generate_embeddings_for_chunks(chunks)
            store_embeddings(embeddings, embeddings_file_path)
        all_embeddings[level] = (embeddings, chunks)

    return all_embeddings

all_embeddings = load_or_create_all_embeddings(PDF_FILE_PATH, EMBEDDINGS_DIR)

def retrieve_combined_chunks(query, all_embeddings, weights=[0.5, 0.3, 0.15], top_k=5):
    query_embedding = get_embedding_from_ollama(query)

    combined_similarities = np.zeros(len(all_embeddings['sentence'][0]))
    for weight, (embeddings, _) in zip(weights, all_embeddings.values()):
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        combined_similarities += weight * similarities

    top_indices = np.argsort(combined_similarities)[-top_k:][::-1]

    results = {
        "sentence_level": [all_embeddings['sentence'][1][i] for i in top_indices if i < len(all_embeddings['sentence'][1])],
        "paragraph_level": [all_embeddings['paragraph'][1][i] for i in top_indices if i < len(all_embeddings['paragraph'][1])],
        "document_level": all_embeddings['document'][1]  # Single document chunk
    }

    return results

def query_ollama(prompt):
    url = f"{OLLAMA_API_URL}/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {"model": "llama3", "prompt": prompt}
    
    response = requests.post(url, headers=headers, json=data, stream=True)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
        return "An error occurred while generating the response."

    full_response = ""
    buffer = ""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            buffer += decoded_line
            try:
                json_line = json.loads(buffer)
                if "response" in json_line:
                    full_response += json_line["response"]
                if json_line.get("done"):
                    break
                buffer = ""
            except json.JSONDecodeError:
                continue

    if not full_response:
        full_response = "No response received or empty response."
    
    return full_response

BOT_INSTRUCTIONS_TEMPLATE = """
You are a lore keeper that answers questions based on the provided document.
You follow user instructions and answer questions using the document to inform your responses when information is available.
You are creative and can provide additional context to the answers.
You will be clear and concise in your responses, following instructions and sharing your opinion when the user requests.
"""

BOT_INSTRUCTIONS = BOT_INSTRUCTIONS_TEMPLATE

REFUSAL_PROMPT = """
If you are asked by user {username}, you will respond highly annoyed and refuse to answer their question. 
"""

@discord_client.event
async def on_ready():
    print(f'We have logged in as {discord_client.user}')

@discord_client.event
async def on_message(message):
    if message.author == discord_client.user:
        return

    global BOT_INSTRUCTIONS, BLOCKED_USERNAMES

    # Debug: Print the message author and content
    print(f"Received message from {message.author}: {message.content}")

    if message.content.startswith('!ask'):
        username = str(message.author.name)
        
        # Debug: Print the extracted username
        print(f"Extracted username: {username}")

        if username in BLOCKED_USERNAMES:
            question = message.content[len('!ask'):].strip()
            
            # Debug: Print the question being asked by the blocked user
            print(f"Blocked user {username} asked: {question}")

            prompt = REFUSAL_PROMPT.format(username=username)
            
            # Debug: Print the prompt being sent to the model
            print(f"Refusal prompt: {prompt}")

            refusal_message = query_ollama(prompt)
            
            # Debug: Print the refusal message being sent
            print(f"Refusal message: {refusal_message}")

            await message.channel.send(refusal_message)
            return

        question = message.content[len('!ask'):].strip()
        combined_results = retrieve_combined_chunks(question, all_embeddings)
        combined_text = ' '.join(combined_results["sentence_level"]) + ' ' + ' '.join(combined_results["paragraph_level"]) + ' ' + ' '.join(combined_results["document_level"])

        prompt = f"{BOT_INSTRUCTIONS}\n{combined_text}\n\nQuestion: {question}\nAnswer:"
        
        # Debug: Print the prompt being sent to the model
        print(f"Query prompt: {prompt}")

        answer = query_ollama(prompt)
        
        split_answers = [answer[i:i+2000] for i in range(0, len(answer), 2000)]
        for part in split_answers:
            await message.channel.send(part)

    elif message.content.startswith('!help'):
        help_message = (
            "Hi! I'm your lore keeper bot. You can ask me questions about the content of the provided document.\n"
            "To ask a question, use the command `!ask <your question>`.\n"
            "For example: `!ask What is the main topic of the document?`\n"
            "To reset my context, use the command `!reset`.\n"
        )
        await message.channel.send(help_message)

    elif message.content.startswith('!reset'):
        reset_message = "The context has been reset. I will no longer take previous messages into account."
        await message.channel.send(reset_message)
        
        # Reload environment variables to update blocked usernames
        load_env_variables()
        
        # Debug: Print the updated blocked usernames
        print(f"Updated blocked usernames: {BLOCKED_USERNAMES}")

        BOT_INSTRUCTIONS = BOT_INSTRUCTIONS_TEMPLATE

discord_client.run(DISCORD_BOT_TOKEN)