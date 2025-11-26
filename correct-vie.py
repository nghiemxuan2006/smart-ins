from protonx import ProtonX
import os
import json
from copy import deepcopy
from html.parser import HTMLParser

# Configure API key (already present in workspace; ensure environment var is set)
os.environ["PROTONX_API_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1lb25naGllbUBnbWFpbC5jb20iLCJpYXQiOjE3NjM4MTEwMDksImV4cCI6MTc2NjQwMzAwOX0.TDif6SqLsbB3XrrLpmmQARrlkcRF3GTo-fQx5fRKg2g"
client = ProtonX()


def correct_text_vie(text: str) -> str:
    """Correct Vietnamese text using ProtonX while returning a single best correction."""
    if not isinstance(text, str) or not text.strip():
        return text
    try:
        # Request multiple candidates to select highest score.
        res = client.text.correct(input=text, top_k=3)
        # Expected shape (per user): {"model": ..., "data": [{"input": ..., "candidates": [{"output": ..., "score": ...}, ...]}]}
        if isinstance(res, dict) and "data" in res:
            data_list = res.get("data") or []
            if data_list and isinstance(data_list[0], dict):
                candidates = data_list[0].get("candidates") or []
                # Choose candidate with max score
                best = None
                for c in candidates:
                    if isinstance(c, dict) and "output" in c:
                        if best is None or (c.get("score", 0) >= best.get("score", 0)):
                            best = c
                if best:
                    return best.get("output", text)
        # Fallbacks for other potential shapes
        if isinstance(res, list) and res:
            first = res[0]
            if isinstance(first, dict):
                # Try common key names
                for key in ("output", "text", "corrected"):
                    if key in first:
                        return first[key]
            if isinstance(first, str):
                return first
        if isinstance(res, dict):
            for key in ("output", "text", "corrected"):
                if key in res:
                    return res[key]
        if isinstance(res, str):
            return res
        return text
    except Exception:
        # Fail-safe: if API fails, return original text
        return text


class TableHTMLCorrector(HTMLParser):
    """Minimal HTML table text corrector that preserves original HTML structure."""

    def __init__(self):
        super().__init__()
        self.result = []

    def handle_starttag(self, tag, attrs):
        attrs_str = "".join(f' {k}="{v}"' for k, v in attrs)
        self.result.append(f"<{tag}{attrs_str}>")

    def handle_endtag(self, tag):
        self.result.append(f"</{tag}>")

    def handle_startendtag(self, tag, attrs):
        attrs_str = "".join(f' {k}="{v}"' for k, v in attrs)
        self.result.append(f"<{tag}{attrs_str} />")

    def handle_data(self, data):
        corrected = correct_text_vie(data)
        self.result.append(corrected)

    def handle_comment(self, data):
        self.result.append(f"<!--{data}-->")

    def get_html(self):
        return "".join(self.result)


def correct_html_table(html: str) -> str:
    if not isinstance(html, str) or "<table" not in html:
        return html
    parser = TableHTMLCorrector()
    try:
        parser.feed(html)
        return parser.get_html()
    except Exception:
        return html


def process_json_structure(obj):
    """Recursively traverse the JSON structure and correct Vietnamese text fields while preserving schema."""
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            # Specific known fields that are free-text
            if k in {"block_content", "label", "text_type"} and isinstance(v, str):
                # For HTML table content, run HTML-aware correction
                new_obj[k] = correct_html_table(v) if "<table" in v else correct_text_vie(v)
            elif k in {"rec_texts"} and isinstance(v, list):
                new_obj[k] = [correct_text_vie(t) for t in v]
            elif k in {"pred_html"} and isinstance(v, str):
                new_obj[k] = correct_html_table(v)
            else:
                new_obj[k] = process_json_structure(v)
        return new_obj
    if isinstance(obj, list):
        return [process_json_structure(item) for item in obj]
    # Primitive
    return obj


def main():
    # Input JSON path (the file provided in attachments)
    input_path = os.path.join("output", "test-epolicy-hop-dong-dien-tu_NhatNguyen_0_res.json")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    corrected = process_json_structure(deepcopy(data))

    # Preserve structure and write corrected output
    save_path = "corrected_test-epolicy-hop-dong-dien-tu_NhatNguyen_0_res.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(corrected, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
