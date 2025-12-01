from pathlib import Path

from ocr.deepseek import handle_file_deepseek_ocr
from chunk.hybrid_chunking import chunk_markdown_file

# Create output directory if it doesn't exist
output_dir = "output/deepseek-ocr"
Path(output_dir).mkdir(parents=True, exist_ok=True)

file_path = "data/epolicy_Extracted_47.pdf"

if __name__ == "__main__":
    # Step 1: Perform OCR on the PDF
    md_result_file_path = handle_file_deepseek_ocr(file_path, output_dir)
    # md_result_file_path = "output/deepseek-ocr/epolicy_1_20.md"
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
    print("\nChunk details:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Header: {chunk.header if chunk.header else 'None'}")
        print(f"Level: {chunk.level}")
        print(f"Type: {chunk.chunk_type}")
        print(f"Size: {len(chunk.content)} characters")
        print(f"Content preview: {chunk.content[:100]}...")
    
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