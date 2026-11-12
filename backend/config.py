import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Trust the operating system's certificate store for HTTPS (Anthropic API, Hugging Face).
# Python's bundled CA list does not include the certificate chain on some networks.
# This must run before anthropic or sentence-transformers make any connection.
import truststore
truststore.inject_into_ssl()

# Use the locally cached embedding model without checking huggingface.co on load.
# An explicit HF_HUB_OFFLINE in the environment still takes precedence.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

@dataclass
class Config:
    """Configuration settings for the RAG system"""
    # Anthropic API settings
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"  # Model for generating responses (verified accepted by the API)
    
    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Document processing settings
    CHUNK_SIZE: int = 800       # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100     # Characters to overlap between chunks
    MAX_RESULTS: int = 5         # Maximum search results to return
    MAX_HISTORY: int = 2         # Number of conversation messages to remember
    
    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

config = Config()


