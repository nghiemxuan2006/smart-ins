from typing import List, Dict, Optional
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    content: str
    start_index: int
    end_index: int
    header: Optional[str] = None
    level: int = 0
    chunk_type: str = "content"
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HybridChunker:
    """
    Hybrid chunker using LangChain that combines header-aware and recursive semantic chunking.
    
    This chunker:
    1. Uses MarkdownHeaderTextSplitter to identify document structure
    2. Then applies RecursiveCharacterTextSplitter for semantic chunking within sections
    """
    
    def __init__(
        self,
        max_chunk_size: int = 2000,
        min_chunk_size: int = 100,
        overlap: int = 200,
        respect_headers: bool = True
    ):
        """
        Initialize the hybrid chunker.
        
        Args:
            max_chunk_size: Maximum size of each chunk in characters
            min_chunk_size: Minimum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            respect_headers: Whether to respect header boundaries
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        self.respect_headers = respect_headers
        
        # Define header splits for markdown
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]
        
        # Initialize LangChain splitters
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )
        
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # Multiple paragraph breaks
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentences
                ", ",      # Clauses
                " ",       # Words
                "",        # Characters
            ]
        )
    
    def chunk_text(self, text: str) -> List[Chunk]:
        """
        Main method to chunk text using hybrid approach with LangChain.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of Chunk objects
        """
        if self.respect_headers:
            # Step 1: Split by headers using MarkdownHeaderTextSplitter
            header_splits = self.markdown_splitter.split_text(text)
            
            # Step 2: Apply recursive semantic chunking to each section
            chunks = []
            current_pos = 0
            
            for doc in header_splits:
                section_chunks = self._process_section(doc, current_pos)
                chunks.extend(section_chunks)
                current_pos += len(doc.page_content)
            
            return chunks
        else:
            # Direct recursive splitting without header awareness
            documents = self.recursive_splitter.create_documents([text])
            return self._documents_to_chunks(documents)
    
    def _process_section(self, doc, start_index: int) -> List[Chunk]:
        """
        Process a section (document) from header splitting.
        
        Args:
            doc: LangChain Document object
            start_index: Starting position in original text
            
        Returns:
            List of Chunk objects
        """
        # Extract header information from metadata
        header = self._extract_header_from_metadata(doc.metadata)
        level = self._extract_level_from_metadata(doc.metadata)
        
        # If section is already small enough, return as single chunk
        if len(doc.page_content) <= self.max_chunk_size:
            return [Chunk(
                content=doc.page_content,
                start_index=start_index,
                end_index=start_index + len(doc.page_content),
                header=header,
                level=level,
                chunk_type="header_aware",
                metadata=doc.metadata
            )]
        
        # Apply recursive semantic chunking to this section
        sub_documents = self.recursive_splitter.create_documents([doc.page_content])
        
        chunks = []
        current_offset = start_index
        
        for sub_doc in sub_documents:
            chunk = Chunk(
                content=sub_doc.page_content,
                start_index=current_offset,
                end_index=current_offset + len(sub_doc.page_content),
                header=header,
                level=level,
                chunk_type="semantic",
                metadata={**doc.metadata, **sub_doc.metadata}
            )
            chunks.append(chunk)
            current_offset += len(sub_doc.page_content)
        
        return chunks
    
    def _extract_header_from_metadata(self, metadata: Dict) -> Optional[str]:
        """Extract the most specific header from metadata."""
        # Check for headers in order of specificity (H6 to H1)
        for i in range(6, 0, -1):
            header_key = f"Header {i}"
            if header_key in metadata:
                return metadata[header_key]
        return None
    
    def _extract_level_from_metadata(self, metadata: Dict) -> int:
        """Extract the header level from metadata."""
        for i in range(6, 0, -1):
            header_key = f"Header {i}"
            if header_key in metadata:
                return i
        return 0
    
    def _documents_to_chunks(self, documents: List, start_index: int = 0) -> List[Chunk]:
        """
        Convert LangChain Documents to Chunk objects.
        
        Args:
            documents: List of LangChain Document objects
            start_index: Starting index in original text
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        current_pos = start_index
        
        for doc in documents:
            chunk = Chunk(
                content=doc.page_content,
                start_index=current_pos,
                end_index=current_pos + len(doc.page_content),
                header=None,
                level=0,
                chunk_type="semantic",
                metadata=doc.metadata
            )
            chunks.append(chunk)
            current_pos += len(doc.page_content)
        
        return chunks
    
    def chunk_documents(self, documents: List) -> List[Chunk]:
        """
        Chunk a list of LangChain Documents.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of Chunk objects
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_text(doc.page_content)
            # Merge document metadata with chunk metadata
            for chunk in chunks:
                chunk.metadata = {**doc.metadata, **chunk.metadata}
            all_chunks.extend(chunks)
        
        return all_chunks


def chunk_markdown_file(
    file_path: str,
    max_chunk_size: int = 2000,
    min_chunk_size: int = 100,
    overlap: int = 200
) -> List[Chunk]:
    """
    Convenience function to chunk a markdown file using LangChain.
    
    Args:
        file_path: Path to the markdown file
        max_chunk_size: Maximum size of each chunk
        min_chunk_size: Minimum size of each chunk
        overlap: Number of characters to overlap
        
    Returns:
        List of Chunk objects
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chunker = HybridChunker(
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        overlap=overlap
    )
    
    return chunker.chunk_text(text)

