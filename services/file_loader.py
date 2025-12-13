from pypdf import PdfReader
from docx import Document


def extract_text_from_file(path: str, extension: str) -> str:
    extension = extension.lower()

    if extension == "txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if extension == "pdf":
        reader = PdfReader(path)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)

    if extension == "docx":
        doc = Document(path)
        return "\n".join([para.text for para in doc.paragraphs])

    raise ValueError("Unsupported file type")
