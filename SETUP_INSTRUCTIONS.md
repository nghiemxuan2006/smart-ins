# Setup Instructions for OpenAI Embeddings with Qdrant

## Changes Made

1. **Updated `embed/embedding.py`**:
   - Added support for both OpenAI and HuggingFace embeddings
   - Added `embedding_provider` parameter to choose between providers
   - Changed default model to `text-embedding-3-small` (OpenAI's model)

2. **Updated `main.py`**:
   - Updated the embedding configuration to use OpenAI embeddings
   - Changed model name from `openai-text-embedding-3-small` to `text-embedding-3-small`

3. **Installed Packages**:
   - `langchain-openai`: LangChain integration for OpenAI
   - `openai`: OpenAI Python SDK

## Required Environment Variables

Create a `.env` file in your project root with the following:

```env
# OpenAI API Key (REQUIRED for embeddings)
OPENAI_API_KEY=sk-your-actual-openai-api-key

# Qdrant Configuration (REQUIRED)
QDRANT_URL=https://your-qdrant-instance.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key

# HuggingFace Token (for OCR)
HF_TOKEN=your-huggingface-token

# DeepSeek OCR URL
OCR_URL=https://alphaxiv--deepseek-ocr-modal-serve.modal.run/run/image
```

## How to Get API Keys

### 1. OpenAI API Key
- Go to https://platform.openai.com/api-keys
- Create a new API key
- Copy and paste it into your `.env` file

### 2. Qdrant Cloud
- Go to https://cloud.qdrant.io/
- Create a free cluster
- Get your cluster URL and API key from the dashboard

## Usage

The system now uses:
- **OpenAI's `text-embedding-3-small`** model for embeddings (1536 dimensions)
- **Qdrant** for vector storage
- **LangChain** for orchestration

You can switch between OpenAI and HuggingFace embeddings by changing the `embedding_provider` parameter:

```python
# Use OpenAI embeddings (default)
vector_store_manager = embed_chunks_to_qdrant(
    chunks=chunks,
    collection_name="pru",
    embedding_model="text-embedding-3-small",
    embedding_provider="openai",
    use_local=False
)

# Or use HuggingFace embeddings
vector_store_manager = embed_chunks_to_qdrant(
    chunks=chunks,
    collection_name="pru",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    embedding_provider="huggingface",
    use_local=True
)
```

## Run Your Pipeline

Once you have your `.env` file set up with the correct API keys:

```bash
python main.py
```

This will:
1. Extract text from PDF using DeepSeek OCR
2. Apply hybrid chunking (header-aware + semantic)
3. Embed chunks using OpenAI's text-embedding-3-small
4. Store embeddings in Qdrant
5. Test similarity search
