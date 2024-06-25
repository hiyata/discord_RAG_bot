import discord
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from embeddings import load_or_create_all_embeddings, get_embedding_from_ollama
import requests

# Function to load environment variables
def load_env_variables():
    load_dotenv()
    global PDF_FILE_PATH, DISCORD_BOT_TOKEN, OLLAMA_API_URL, OLLAMA_MODEL_NAME, BLOCKED_USERNAMES, EMBEDDINGS_DIR
    PDF_FILE_PATH = os.getenv('PDF_FILE_PATH')
    DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
    OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://127.0.0.1:11434')
    OLLAMA_MODEL_NAME = os.getenv('OLLAMA_MODEL_NAME', 'llama3')
    BLOCKED_USERNAMES = os.getenv('BLOCKED_USERNAMES', '').split(',')
    EMBEDDINGS_DIR = 'pdf_chunks'

# Function to load instructions from file
def load_instructions(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Load initial environment variables
load_env_variables()

if not PDF_FILE_PATH or not DISCORD_BOT_TOKEN:
    raise ValueError("One or more environment variables are not set. Please check your .env file.")

# Load instructions
BOT_INSTRUCTIONS = load_instructions('positive_instructions.txt')
REFUSAL_PROMPT_TEMPLATE = load_instructions('negative_instructions.txt')

# Initialize the Discord client
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
discord_client = discord.Client(intents=intents)

# Load embeddings
all_embeddings = load_or_create_all_embeddings(PDF_FILE_PATH, EMBEDDINGS_DIR, OLLAMA_MODEL_NAME)

def retrieve_combined_chunks(query, all_embeddings, weights=[0.5, 0.3, 0.15], top_k=5):
    query_embedding = get_embedding_from_ollama(query, OLLAMA_MODEL_NAME)

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
    data = {"model": OLLAMA_MODEL_NAME, "prompt": prompt}
    
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

            prompt = REFUSAL_PROMPT_TEMPLATE.format(username=username)
            
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
        
        # Split the answer into parts to fit Discord's message limit
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

        BOT_INSTRUCTIONS = load_instructions('positive_instructions.txt')
        REFUSAL_PROMPT_TEMPLATE = load_instructions('negative_instructions.txt')

discord_client.run(DISCORD_BOT_TOKEN)