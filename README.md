# gpt_start_langchain

A LangChain chatbot with TF-IDF RAG, tool calling, and memory — built for students learning AI concepts.

## Features

- **RAG** — retrieves relevant course material using TF-IDF before answering
- **Tools** — calculator, course Q&A, email summary, and session bye
- **Memory** — keeps conversation context across turns via LangGraph's MemorySaver
- **Azure OpenAI** — powered by GPT-4 through Azure

## Requirements

- Python 3.14+
- An Azure OpenAI deployment

## Setup

1. **Clone the repo and create a virtual environment:**

   ```bash
   python -m venv venv          # or "py -m venv venv"
   venv\Scripts\activate        # Windows
   # source venv/bin/activate   # Linux/macOS
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** in the project root:

   ```env
   AZURE_OPENAI_API_KEY=your_key_here
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2024-12-01-preview
   AZURE_OPENAI_DEPLOYMENT=gpt-4.1
   ```

4. **Add your course material** to `assets/course_notes.txt`.

## Usage

```bash
python main.py
```

Type your message and press Enter. Type `exit` or `quit` to end the session.

```
You: What topics should I study first?
Bot: Great question! Let me check the course material...

You: How much is 150 * 3?
Bot: Let the calculator shine — that's 450!

You: exit
```

## Project Structure

```
main.py               # Entry point
src/
  bot.py              # BetterChatbot class
  settings.py         # Logger setup
assets/
  course_notes.txt    # Course material for RAG
```
