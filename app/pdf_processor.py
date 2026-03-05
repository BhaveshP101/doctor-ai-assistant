import fitz
import re
import pytesseract
import os
from pdf2image import convert_from_path

# -------------------------------
# Tesseract configuration
# -------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

# Poppler path
POPPLER_PATH = r"D:\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"


# -------------------------------
# Extract text from PDF
# -------------------------------
def extract_text_from_pdf(pdf_path):

    text = ""

    print("Loading PDF...")

    doc = fitz.open(pdf_path)

    # Try extracting selectable text first
    for page in doc:
        text += page.get_text()

    doc.close()

    # If PDF is scanned → use OCR
    if not text.strip():

        print("No selectable text found. Using OCR...")

        page_count = 1136   # total pages in the book
        batch_size = 10

        for i in range(1, page_count, batch_size):

            last_page = min(i + batch_size, page_count)

            print(f"OCR pages {i} to {last_page}")

            pages = convert_from_path(
                pdf_path,
                dpi=300,
                first_page=i,
                last_page=last_page,
                poppler_path=POPPLER_PATH
            )

            for page in pages:
                text += pytesseract.image_to_string(page)

    return text


# -------------------------------
# Split text into remedy chunks
# -------------------------------
def chunk_text(text):

    print("Chunking remedies...")

    lines = text.split("\n")

    chunks = []
    current_chunk = ""

    for line in lines:

        line = line.strip()

        # Detect medicine names (ALL CAPS)
        if re.match(r'^[A-Z][A-Z\s\-]{3,40}$', line) and not line.startswith(
            ("MIND", "HEAD", "FEVER", "NOSE", "MOUTH", "STOMACH", "CHEST", "BACK", "SKIN")
        ):

            if current_chunk:
                chunks.append(current_chunk.strip())

            current_chunk = line + "\n"

        else:
            current_chunk += line + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    print("Total remedies detected:", len(chunks))

    return chunks