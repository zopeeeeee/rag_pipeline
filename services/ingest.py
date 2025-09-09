import os, uuid, json
from typing import List, Dict
from pypdf import PdfReader
import docx

def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_md(path):
    return read_txt(path)

def read_pdf(path):
    text = []
    reader = PdfReader(path)
    for p in reader.pages:
        page_text = p.extract_text() or ""
        text.append(page_text)
    return "\n".join(text)

def read_docx(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def chunk_text(text: str, chunk_words=300, overlap_words=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_words]
        chunks.append(" ".join(chunk))
        i += chunk_words - overlap_words
    return chunks

def parse_file(path: str) -> List[Dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt"]:
        text = read_txt(path)
    elif ext in [".md", ".markdown"]:
        text = read_md(path)
    elif ext in [".pdf"]:
        text = read_pdf(path)
    elif ext in [".docx"]:
        text = read_docx(path)
    else:
        raise ValueError("Unsupported file type: " + ext)

    chunks = chunk_text(text, chunk_words=300, overlap_words=50)
    out = []
    for i, c in enumerate(chunks):
        out.append({
            "id": uuid.uuid4().hex,
            "text": c,
            "source": os.path.basename(path),
            "chunk_id": i
        })
    return out

if __name__ == "_main_":
    import sys
    p = sys.argv[1]
    chunks = parse_file(p)
    print(f"Created {len(chunks)} chunks from {p}")
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/chunks_preview.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)