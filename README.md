# PDF Discord Bot

This bot answers questions about a PDF document in a Discord channel.

## Setup

1. Clone the repository.
2. Create a `.env` file in the root directory with the following content:
    DISCORD_BOT_TOKEN=your_discord_bot_token
    OPENAI_API_KEY=your_openai_api_key
    PDF_FILE_PATH=path/to/your/pdf_file.pdf

3. Install the required packages:
    ```bash 
    pip install -r requirements.txt
    ```

4. Run the bot:

    ```bash 
    python bot.py
    ```

## Usage

- Ask questions with the command `!ask <your question>`.
- For help, use the command `!help`.

## .env File Format

The '.env' file should have the following format:

```bash
    DISCORD_BOT_TOKEN=your_discord_bot_token
    OPENAI_API_KEY=your_openai_api_key
    PDF_FILE_PATH=path/to/your/pdf_file.pdf
```