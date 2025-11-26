# # Initialize PaddleOCR instance
# from paddleocr import PaddleOCR, PPStructureV3
# ocr = PPStructureV3(
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False, lang="vi")

# # Run OCR inference on a sample image 
# result = ocr.predict(
#     input="C:\\Users\\nghiemxuan\\Downloads\\epolicy-hop-dong-dien-tu_NhatNguyen.pdf")

# # Visualize the results and save the JSON results
# for res in result:
#     res.save_to_img("output")
#     res.save_to_json("output")

# # Extract detected text lines
# lines = []
# for line in result:
#     for word_info in line:
#         lines.append(word_info[1][0])

# # Join lines with Markdown line breaks
# markdown_text = '\n\n'.join(lines)

# # Save to a markdown file
# with open('output.md', 'w', encoding='utf-8') as f:
#     f.write(markdown_text)

# print("Text extracted and saved to output.md")

# for res in result:
#     res.save_to_markdown("output")

from paddleocr import PPStructureV3

pipeline = PPStructureV3(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    lang="vi",
)
# output = pipeline.predict(input="C:\\Users\\nghiemxuan\\Downloads\\epolicy-hop-dong-dien-tu_NhatNguyen.pdf")
output = pipeline.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png",
)
for res in output:
    print(res, type(res))
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")