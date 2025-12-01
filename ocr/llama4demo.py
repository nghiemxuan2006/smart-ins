import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()
API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
}

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        img_bytes = f.read()
        return "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode("utf-8")

local_image = encode_image("/Users/thanhta/Desktop/WS/smart-ins/data/epolicy-image.png")

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

response = query({
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Lấy toàn bộ text trong ảnh, bao gom cả text trong bảng nếu có."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": local_image
                    }
                }
            ]
        }
    ],
    "model": "meta-llama/Llama-4-Maverick-17B-128E-Instruct:groq"
})

print(response)

# print(response["choices"][0]["message"]["content"])
