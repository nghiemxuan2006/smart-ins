from pathlib import Path

from ocr.deepseek import handle_file_deepseek_ocr
from chunk.hybrid_chunking import chunk_markdown_file
from embed.embedding import embed_chunks_to_qdrant

# Create output directory if it doesn't exist
output_dir = "output/deepseek-ocr"
Path(output_dir).mkdir(parents=True, exist_ok=True)

file_path = "data/epolicy_Extracted_47.pdf"

if __name__ == "__main__":
    # Step 1: Perform OCR on the PDF
    md_result_file_path = handle_file_deepseek_ocr(file_path, output_dir)
    # md_result_file_path = "output/deepseek-ocr/epolicy_Extracted_47/epolicy_Extracted_47.md"
    print(f"DeepSeek OCR result saved to: {md_result_file_path}")
    
    # Step 2: Apply hybrid chunking to the markdown result
    print(f"\nApplying hybrid chunking... on markdown file: {md_result_file_path}")
    chunks = chunk_markdown_file(
        md_result_file_path,
        max_chunk_size=1000,  # Adjust as needed
        min_chunk_size=100,
        overlap=50
    )
    
    # Display chunking results
    print(f"\nTotal chunks created: {len(chunks)}")
    
    # Optionally save chunks to a file
    chunks_output_path = md_result_file_path.replace('.md', '_chunks.txt')
    with open(chunks_output_path, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks, 1):
            f.write(f"{'='*60}\n")
            f.write(f"CHUNK {i}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Header: {chunk.header if chunk.header else 'None'}\n")
            f.write(f"Level: {chunk.level}\n")
            f.write(f"Type: {chunk.chunk_type}\n")
            f.write(f"Size: {len(chunk.content)} characters\n")
            f.write(f"\n{chunk.content}\n\n")
    
    print(f"\nChunks saved to: {chunks_output_path}")
    
    # Step 3: Embed chunks and store in Qdrant
    print("\n" + "="*60)
    print("Step 3: Embedding and storing in Qdrant")
    print("="*60)
    
    vector_store_manager = embed_chunks_to_qdrant(
        chunks=chunks,
        collection_name="insurance_docs",
        embedding_model="text-embedding-3-small",
        embedding_provider="openai",
        use_local=False  # Set to False for remote Qdrant
    )
    
    # Step 4: Test similarity search
    print("\n" + "="*60)
    print("Step 4: Testing similarity search")
    print("="*60)
    
    test_query = "What is the insurance policy coverage?"
    print(f"\nQuery: {test_query}")
    
    results = vector_store_manager.similarity_search_with_score(test_query, k=3)
    
    print(f"\nTop {len(results)} results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n--- Result {i} (Score: {score:.4f}) ---")
        print(f"Header: {doc.metadata.get('header', 'None')}")
        print(f"Chunk Type: {doc.metadata.get('chunk_type', 'N/A')}")
        print(f"Content preview: {doc.page_content[:200]}...")
    
    # Get collection info
    print("\n" + "="*60)
    print("Collection Information")
    print("="*60)
    info = vector_store_manager.get_collection_info()
    print(f"\nCollection Info: {info}")
    
    print("\nPipeline completed successfully!")