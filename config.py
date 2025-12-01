import os
from dotenv import load_dotenv

load_dotenv()

OCR_URL = os.environ.get("OCR_URL")

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")