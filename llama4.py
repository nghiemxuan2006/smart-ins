import os
import base64
import requests
from dotenv import load_dotenv
from pdf2image import convert_from_path
from pathlib import Path

load_dotenv()
API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
}

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        img_bytes = f.read()
        return "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode("utf-8")

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def process_image(image_data: str, page_num: int) -> str:
    """Process a single image through the API"""
    response = query({
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Lấy toàn bộ text trong ảnh, bao gom cả text trong bảng nếu có. Trả về định dạng markdown, giữ nguyên định dạng bảng nếu có."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data
                        }
                    }
                ]
            }
        ],
        "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct:groq"
    })

    print(f"Response for page {page_num}: {response}")

    return response["choices"][0]["message"]["content"]

def pdf_to_text(pdf_path: str, output_dir: str = "output/llama4-output") -> list:
    """Convert PDF to images and extract text from each page"""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert PDF to images
    print(f"Converting PDF to images: {pdf_path}")
    images = convert_from_path(pdf_path)
    
    results = []
    
    # Process each page
    for i, image in enumerate(images, start=1):
        print(f"Processing page {i}/{len(images)}...")
        
        # Save image temporarily
        image_path = os.path.join(output_dir, f"page_{i}.jpg")
        image.save(image_path, "JPEG")
        
        # Encode and process image
        encoded_image = encode_image(image_path)
        text = process_image(encoded_image, i)
        
        results.append({
            "page": i,
            "image_path": image_path,
            "text": text
        })
        
        # Save text output for this page
        text_output_path = os.path.join(output_dir, f"page_{i}.md")
        with open(text_output_path, "w", encoding="utf-8") as f:
            f.write(text if text else "")
        
        print(f"Page {i} completed. Saved to {text_output_path}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Replace with your PDF path
    pdf_path = "/Users/thanhta/Desktop/WS/smart-ins/data/epolicy_1.pdf"
    
    if os.path.exists(pdf_path):
        results = pdf_to_text(pdf_path)
        
        # Print results
        print("\n" + "="*50)
        print("EXTRACTION RESULTS")
        print("="*50 + "\n")
        
        for result in results:
            print(f"\n--- Page {result['page']} ---")
            print(result['text'])
            print()
    else:
        print(f"PDF file not found: {pdf_path}")
        print("Please update the pdf_path variable with your PDF file path.")
