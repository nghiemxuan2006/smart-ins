# Insurance Document Chatbot

Two chat interfaces are available to interact with your insurance documents.

## 1. Terminal Chat (Simple)

Run the command-line chat interface:

```bash
python chat.py
```

This provides a simple terminal-based chat where you can:
- Ask questions about your insurance documents
- See AI-generated answers based on your documents
- View source documents that were used to generate the answer
- Type 'quit' or 'exit' to end the session

### Example Usage:

```
🧑 You: What is the coverage limit for medical expenses?

🤖 Assistant: Based on the insurance policy documents, the coverage limit for medical expenses is $100,000 per person.

📚 Sources:
  1. Section: Coverage Limits
     Type: semantic
     Content: Medical expenses are covered up to $100,000 per person...
```

## 2. Web Chat UI (Advanced)

Run the Streamlit web interface:

```bash
streamlit run chat_ui.py
```

This will open a web browser with an interactive chat interface featuring:
- **Clean, modern UI** with chat bubbles
- **Chat history** that persists during your session
- **Expandable source documents** to verify answers
- **Settings sidebar** to toggle source display
- **Clear chat** button to start fresh

### Features:

- 💬 **Interactive Chat**: Ask questions in natural language
- 📚 **Source Citations**: See which documents the AI used
- 🔄 **Chat History**: Review previous questions and answers
- ⚙️ **Customizable**: Toggle source display on/off
- 🎨 **Beautiful UI**: Professional Streamlit interface

## Requirements

Make sure you have:
1. Populated your vector database by running `main.py`
2. Set up your `.env` file with:
   - `OPENAI_API_KEY`
   - `QDRANT_URL`
   - `QDRANT_API_KEY`

## Install Streamlit

If you haven't installed Streamlit yet:

```bash
pip install streamlit
```

## How It Works

Both interfaces use:
- **RAG (Retrieval Augmented Generation)**: Finds relevant document chunks from your insurance PDFs
- **OpenAI GPT-4o-mini**: Generates accurate answers based on retrieved context
- **Qdrant Vector Database**: Stores and searches document embeddings
- **LangChain**: Orchestrates the entire pipeline

## Tips

- Be specific in your questions for better answers
- The AI will only answer based on your uploaded documents
- If information isn't in your documents, the AI will tell you
- Sources help you verify the accuracy of answers
