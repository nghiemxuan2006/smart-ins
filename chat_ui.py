import streamlit as st
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

from config import QDRANT_URL, QDRANT_API_KEY

load_dotenv()


@st.cache_resource
def initialize_chatbot():
    """Initialize the chatbot (cached to avoid reloading)."""
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Initialize Qdrant
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="pru",
        embedding=embeddings
    )
    
    # Initialize retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful insurance assistant. Answer questions based on the provided context from insurance documents.

If the answer cannot be found in the context, say "I don't have enough information to answer that question based on the provided documents."

Be concise and accurate. If relevant, cite specific sections or policies mentioned in the context.

Context:
{context}"""),
        ("user", "{question}")
    ])
    
    def format_docs(docs: List) -> str:
        """Format retrieved documents into a single string."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            header = doc.metadata.get('header', 'N/A')
            content = doc.page_content
            formatted.append(f"Document {i} (Section: {header}):\n{content}")
        return "\n\n".join(formatted)
    
    # Create RAG chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever


def main():
    """Main Streamlit app."""
    
    # Page config
    st.set_page_config(
        page_title="Insurance Document Chatbot",
        page_icon="📋",
        layout="wide"
    )
    
    # Title
    st.title("📋 Insurance Document Chatbot")
    st.markdown("Ask questions about your insurance documents using AI-powered search.")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.session_state.show_sources = st.checkbox(
            "Show source documents",
            value=st.session_state.show_sources
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This chatbot uses:
        - **RAG (Retrieval Augmented Generation)**
        - **OpenAI GPT-4o-mini** for responses
        - **Qdrant** for vector storage
        - **LangChain** for orchestration
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chatbot
    try:
        rag_chain, retriever = initialize_chatbot()
    except Exception as e:
        st.error(f"❌ Error initializing chatbot: {e}")
        st.info("""
        Make sure:
        1. Your .env file has OPENAI_API_KEY, QDRANT_URL, and QDRANT_API_KEY
        2. You have run main.py to populate the vector database
        3. Your Qdrant instance is accessible
        """)
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message and st.session_state.show_sources:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}** - Section: {source['header']}")
                        st.text(source['content'])
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your insurance documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get answer
                    answer = rag_chain.invoke(prompt)
                    
                    # Get sources
                    sources = []
                    if st.session_state.show_sources:
                        docs = retriever.invoke(prompt)
                        for doc in docs:
                            sources.append({
                                "header": doc.metadata.get('header', 'N/A'),
                                "chunk_type": doc.metadata.get('chunk_type', 'N/A'),
                                "content": doc.page_content[:300] + "..."
                            })
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources
                    if sources and st.session_state.show_sources:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}** - Section: {source['header']}")
                                st.text(source['content'])
                                st.markdown("---")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })


if __name__ == "__main__":
    main()
