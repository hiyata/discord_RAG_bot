# PDF Discord Bot

This bot allows you to ask questions based on a provided PDF document. It integrates with Discord and uses either the OpenAI API or Ollama API to generate responses. The bot processes a PDF document, generates embeddings for text chunks, and uses these embeddings to answer user queries.

### Features

Power your discord bot with Retrieval Augmented Generation (RAG) through Ollama or OpenAI's api. 

You can extract chunks from any pdf and have your bot search the pdf for the most relevant chunks. 

You can set custom instructions, and refusal instructions. The code will extract the user name and pass the custom instructions to the bot. 

## Setup

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/hiyata/discord_ai_bot.git
   cd discord_ai_bot
   ```

2. **Install Requirements**:

    ```bash
    pip install -r requirements.txt 
    ```

3. **Set up the environment**:

    Create a .env file in the root directory of your project. The .env file should have the following format:

    ```bash
    PDF_FILE_PATH=path/to/your/pdf_file.pdf
    DISCORD_BOT_TOKEN=your_discord_bot_token
    OPENAI_API_KEY=your_openai_api_key
    OPENAI_MODEL_NAME=text-davinci-003  # Replace with the openai model
    OLLAMA_API_URL=http://127.0.0.1:11434  # Or use the Ollama API URL
    OLLAMA_MODEL_NAME=llama3  # Specify your Ollama model
    BLOCKED_USERNAMES=BLOCKED_USER1,BLOCKED_USER2
    ```

    There is an example ```.env.example```, replace each entry with your specifications. 

4. **Preparing your pdf**

    When using ```Ollama```, generate the embeddings from your chosen model with

    ```bash
    python ollama_generate_embeddings.py
    ```
    
    If you want to use ```Openai```, you can generate chunks using

    ```
    python generate_chunks.py
    ```

    Using ```Ollama``` we extract the embeddings using the model. For ```OpenAI``` we generate chunks using ```punkt```.

    For both codes, we extract the context at the *sentence*, *paragraph*, and *document* level, generating three files. They are assigned seperate weights for importance during retrieval, with the default weights for each being ```0.5, 0.3, 0.15``` respectively. 

## Usage

1. **Run the bot using Ollama or OpenAI's api**
    
    Run **Ollama** with:

    ```bash
    python bot_ollama.py
    ```

    If you prefer to use **OpenAI**'s api, then run the bot with:

    ```bash
    python bot_openai.py
    ```

2. **Talk to your bot on discord**

- Ask questions with the command `!ask <your question>`.
- For help, use the command `!help`.
- Reset the model with `!reset`

