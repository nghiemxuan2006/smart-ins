from pathlib import Path
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

from config import QDRANT_URL, QDRANT_API_KEY

load_dotenv()


class InsuranceChatbot:
    """
    A chatbot for answering questions about insurance documents using RAG.
    """
    
    def __init__(
        self,
        collection_name: str = "insurance_docs",
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        k_results: int = 3
    ):
        """
        Initialize the chatbot.
        
        Args:
            collection_name: Name of the Qdrant collection
            model_name: OpenAI model to use for chat
            temperature: Temperature for response generation
            k_results: Number of documents to retrieve
        """
        self.collection_name = collection_name
        self.k_results = k_results
        
        # Initialize embeddings
        print("Loading embeddings model...")
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize Qdrant client and vector store
        print(f"Connecting to Qdrant at: {QDRANT_URL}")
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings
        )
        
        # Initialize retriever
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": k_results}
        )
        
        # Initialize LLM
        print(f"Loading {model_name}...")
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create RAG prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful insurance assistant. Answer questions based on the provided context from insurance documents.
            
If the answer cannot be found in the context, say "I don't have enough information to answer that question based on the provided documents."

Be concise and accurate. If relevant, cite specific sections or policies mentioned in the context.

Context:
{context}"""),
            ("user", "{question}")
        ])
        
        # Create RAG chain
        self.rag_chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("Chatbot initialized successfully!\n")
    
    def _format_docs(self, docs: List) -> str:
        """Format retrieved documents into a single string."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            header = doc.metadata.get('header', 'N/A')
            content = doc.page_content
            formatted.append(f"Document {i} (Section: {header}):\n{content}")
        return "\n\n".join(formatted)
    
    def ask(self, question: str, show_sources: bool = True) -> dict:
        """
        Ask a question and get an answer.
        
        Args:
            question: User's question
            show_sources: Whether to show source documents
            
        Returns:
            Dictionary with answer and sources
        """
        # Get answer
        answer = self.rag_chain.invoke(question)
        
        # Get source documents if requested
        sources = []
        if show_sources:
            docs = self.retriever.invoke(question)
            for doc in docs:
                sources.append({
                    "header": doc.metadata.get('header', 'N/A'),
                    "chunk_type": doc.metadata.get('chunk_type', 'N/A'),
                    "content": doc.page_content[:200] + "..."
                })
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def chat(self):
        """Start an interactive chat session."""
        print("="*60)
        print("Insurance Document Chatbot")
        print("="*60)
        print("\nAsk questions about your insurance documents.")
        print("Type 'quit' or 'exit' to end the conversation.\n")
        
        while True:
            try:
                # Get user input
                question = input("\n🧑 You: ").strip()
                
                # Check for exit commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                # Skip empty questions
                if not question:
                    continue
                
                # Get answer
                print("\n🤖 Assistant: ", end="", flush=True)
                result = self.ask(question, show_sources=True)
                print(result["answer"])
                
                # Show sources
                if result["sources"]:
                    print("\n📚 Sources:")
                    for i, source in enumerate(result["sources"], 1):
                        print(f"\n  {i}. Section: {source['header']}")
                        print(f"     Type: {source['chunk_type']}")
                        print(f"     Content: {source['content']}")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.")


def main():
    """Main function to run the chatbot."""
    try:
        # Initialize chatbot
        chatbot = InsuranceChatbot(
            collection_name="insurance_docs",
            model_name="gpt-4o-mini",
            temperature=0.0,
            k_results=3
        )
        
        # Start chat session
        chatbot.chat()
        
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        print("\nMake sure:")
        print("1. Your .env file has OPENAI_API_KEY, QDRANT_URL, and QDRANT_API_KEY")
        print("2. You have run main.py to populate the vector database")
        print("3. Your Qdrant instance is accessible")


if __name__ == "__main__":
    main()
