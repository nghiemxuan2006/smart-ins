from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.documents import Document

from config import QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY


class VectorStoreManager:
    """
    Manager for embedding chunks and storing them in Qdrant vector database.
    """
    
    def __init__(
        self,
        collection_name: str = "pru",
        embedding_model: str = "text-embedding-3-small",
        embedding_provider: str = "openai",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        use_local: bool = True
    ):
        """
        Initialize the vector store manager.
        
        Args:
            collection_name: Name of the Qdrant collection
            embedding_model: Embedding model to use
            embedding_provider: Provider of the embedding model ('openai' or 'huggingface')
            qdrant_url: Qdrant server URL (for cloud/remote)
            qdrant_api_key: Qdrant API key (for cloud)
            use_local: Use local in-memory Qdrant instance
        """
        self.collection_name = collection_name
        self.use_local = use_local
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        
        # Initialize embeddings based on provider
        print(f"Loading embedding model: {embedding_model} from {embedding_provider}")
        
        if embedding_provider.lower() == "openai":
            self.embeddings = OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=OPENAI_API_KEY
            )
        else:  # huggingface
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        # Initialize Qdrant client
        if use_local:
            print("Using local in-memory Qdrant")
            self.client = QdrantClient(":memory:")
            self.connection_params = {"location": ":memory:"}
        elif qdrant_url:
            print(f"Connecting to Qdrant at: {qdrant_url}")
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key
            )
            self.connection_params = {"url": qdrant_url, "api_key": qdrant_api_key}
        else:
            # Use local persistent storage
            print("Using local persistent Qdrant storage")
            self.client = QdrantClient(path="./qdrant_storage")
            self.connection_params = {"path": "./qdrant_storage"}
        
        self.vector_store = None
    
    def create_collection(self, vector_size: int = 384):
        """
        Create a new collection in Qdrant.
        
        Args:
            vector_size: Size of the embedding vectors (default 384 for all-MiniLM-L6-v2)
        """
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Collection '{self.collection_name}' created successfully")
        except Exception as e:
            print(f"Collection might already exist or error occurred: {e}")
    
    def chunks_to_documents(self, chunks: List) -> List[Document]:
        """
        Convert Chunk objects to LangChain Document objects.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk.content,
                metadata={
                    "chunk_id": i,
                    "header": chunk.header,
                    "level": chunk.level,
                    "chunk_type": chunk.chunk_type,
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    "size": len(chunk.content),
                    **(chunk.metadata if hasattr(chunk, 'metadata') and chunk.metadata else {})
                }
            )
            documents.append(doc)
        
        return documents
    
    def clear_collection(self):
        """
        Delete the collection if it exists to start fresh.
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name in collection_names:
                self.client.delete_collection(self.collection_name)
                print(f"Collection '{self.collection_name}' cleared successfully")
            else:
                print(f"Collection '{self.collection_name}' does not exist, will create new one")
        except Exception as e:
            print(f"Error clearing collection: {e}")
    
    def embed_and_store(self, chunks: List, batch_size: int = 100, clear_existing: bool = True) -> QdrantVectorStore:
        """
        Embed chunks and store them in Qdrant.
        
        Args:
            chunks: List of Chunk objects
            batch_size: Batch size for embedding and storing
            clear_existing: If True, clears the collection before storing (default: True)
            
        Returns:
            QdrantVectorStore instance
        """
        print(f"\nEmbedding and storing {len(chunks)} chunks...")
        
        # Clear collection if requested
        if clear_existing:
            self.clear_collection()
        
        # Convert chunks to documents
        documents = self.chunks_to_documents(chunks)
        
        # Create vector store and add documents using connection parameters
        self.vector_store = QdrantVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            **self.connection_params
        )
        
        print(f"Successfully embedded and stored {len(chunks)} chunks in Qdrant")
        return self.vector_store
    
    def similarity_search(
        self,
        query: str,
        k: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Query string
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of Document objects
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call embed_and_store first.")
        
        if score_threshold:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            filtered_results = [doc for doc, score in results if score >= score_threshold]
            return filtered_results
        else:
            return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10
    ) -> List[tuple]:
        """
        Perform similarity search and return results with scores.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call embed_and_store first.")
        
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def get_collection_info(self):
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return None
    
    def delete_collection(self):
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' deleted successfully")
        except Exception as e:
            print(f"Error deleting collection: {e}")


def embed_chunks_to_qdrant(
    chunks: List,
    collection_name: str = "pru",
    embedding_model: str = "text-embedding-3-small",
    embedding_provider: str = "openai",
    use_local: bool = False,
    clear_existing: bool = True
) -> VectorStoreManager:
    """
    Convenience function to embed chunks and store in Qdrant.
    
    Args:
        chunks: List of Chunk objects
        collection_name: Name of the Qdrant collection
        embedding_model: Embedding model to use
        embedding_provider: Provider of the embedding model ('openai' or 'huggingface')
        use_local: Use local Qdrant instance
        clear_existing: If True, clears the collection before storing (default: True)
        
    Returns:
        VectorStoreManager instance
    """
    manager = VectorStoreManager(
        collection_name=collection_name,
        embedding_model=embedding_model,
        embedding_provider=embedding_provider,
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        use_local=use_local
    )
    
    manager.embed_and_store(chunks, clear_existing=clear_existing)
    
    return manager
