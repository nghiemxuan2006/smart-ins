import requests
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from pdf2image import convert_from_path

load_dotenv()
url = os.environ.get("OCR_URL")
if not url:
    raise ValueError("OCR_URL not found in environment variables.")
file_path = "data/epolicy_18_20.pdf"

# Create output directory if it doesn't exist
output_dir = "output/deepseek-ocr"
Path(output_dir).mkdir(parents=True, exist_ok=True)

def process_image(image_path: str) -> dict:
    """Process a single image through the API"""
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
    return response.json()

def process_pdf(pdf_path: str, output_dir: str) -> None:
    """Convert PDF to images and process each page"""
    print(f"Converting PDF to images: {pdf_path}")
    images = convert_from_path(pdf_path)
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Create temp directory for images
    temp_dir = os.path.join(output_dir, "temp_images")
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    
    all_results = []
    all_ocr_text = []
    
    # Process each page
    for i, image in enumerate(images, start=1):
        print(f"Processing page {i}/{len(images)}...")
        
        # Save image temporarily
        image_path = os.path.join(temp_dir, f"page_{i}.jpg")
        image.save(image_path, "JPEG")
        
        # Process image
        result = process_image(image_path)
        all_results.append({
            "page": i,
            "result": result
        })
        
        # Extract ocr_text if available
        if "ocr_text" in result:
            all_ocr_text.append(result["ocr_text"])
        
        print(f"Page {i} completed.")
    
    # Save combined JSON
    json_output_path = os.path.join(output_dir, f"{base_name}.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # Save combined Markdown (only ocr_text)
    md_output_path = os.path.join(output_dir, f"{base_name}.md")
    with open(md_output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_ocr_text))
    
    # Clean up temp images
    import shutil
    shutil.rmtree(temp_dir)
    
    print("\nResponse saved to:")
    print(f"  JSON: {json_output_path}")
    print(f"  Markdown: {md_output_path}")

def process_image_file(image_path: str, output_dir: str) -> None:
    """Process a single image file"""
    print(f"Processing image: {image_path}")
    
    result = process_image(image_path)
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save as JSON
    json_output_path = os.path.join(output_dir, f"{base_name}.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # Save as Markdown (only ocr_text)
    md_output_path = os.path.join(output_dir, f"{base_name}.md")
    with open(md_output_path, "w", encoding="utf-8") as f:
        if "ocr_text" in result:
            f.write(result["ocr_text"])
        else:
            # Fallback: write the entire JSON as formatted text
            f.write(json.dumps(result, ensure_ascii=False, indent=2))
    
    print("\nResponse saved to:")
    print(f"  JSON: {json_output_path}")
    print(f"  Markdown: {md_output_path}")

# Main execution
if __name__ == "__main__":
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
    elif file_path.lower().endswith(".pdf"):
        process_pdf(file_path, output_dir)
    else:
        process_image_file(file_path, output_dir)