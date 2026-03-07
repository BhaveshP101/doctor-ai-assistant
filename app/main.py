from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os

from app.pdf_processor import extract_text_from_pdf, chunk_text
from app.embedder import model
from app.vector_store import create_faiss_index, search_index


app = FastAPI(
    title="Doctor AI Medical Assistant",
    description="AI-powered semantic search for medicine recommendation",
    version="1.0"
)

templates = Jinja2Templates(directory="templates")
chunks = None
index = None

@app.on_event("startup")
def load_ai_database():
    global chunks, index

    import pickle

    print("Loading saved AI database...")

    with open("data/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    with open("data/embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    print(f"Total remedies loaded: {len(chunks)}")

    index = create_faiss_index(embeddings)

    print("AI system ready.")
# ---------------------------------------------------
# FILE PATHS
# ---------------------------------------------------

PDF_PATH = os.path.join("data", "Homoeopathic.pdf")
TEXT_PATH = os.path.join("data", "book_text.txt")

print("Preparing medical database...")

# ---------------------------------------------------
# LOAD BOOK TEXT (FAST STARTUP)
# ---------------------------------------------------

if os.path.exists(TEXT_PATH):

    print("Loading saved book text...")

    with open(TEXT_PATH, "r", encoding="utf-8") as f:
        medical_text = f.read()

else:

    print("Running OCR on book (first time only)...")

    medical_text = extract_text_from_pdf(PDF_PATH)

    with open(TEXT_PATH, "w", encoding="utf-8") as f:
        f.write(medical_text)

    print("Book text saved for future runs.")

# ---------------------------------------------------

import pickle

print("Loading saved AI database...")

with open("data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

with open("data/embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

print(f"Total remedies loaded: {len(chunks)}")

print("Creating FAISS index...")
index = create_faiss_index(embeddings)

print("System ready.")

# ---------------------------------------------------

class SymptomRequest(BaseModel):
    symptoms: str


@app.get("/")
def home():
    return {"message": "Doctor AI Backend Running Successfully"}


@app.get("/doctor", response_class=HTMLResponse)
def doctor_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze-symptoms")
def analyze_symptoms(request: SymptomRequest):

    query = request.symptoms.lower()

    # split symptoms by comma
    symptoms = [s.strip() for s in query.split(",")]

    query_embedding = model.encode([query])

    # search more candidates first
    results = search_index(index, query_embedding[0], k=50)

    ranked_results = []

    for i in results:

        text_block = chunks[i]
        text_lower = text_block.lower()

        lines = text_block.split("\n")

        medicine_name = lines[0].strip()

        # ❌ filter non-remedy headings
        if len(medicine_name.split()) > 3:
            continue

        if "system" in medicine_name.lower():
            continue

        if "chapter" in medicine_name.lower():
            continue

        score = 0
        matched = []

        # phrase symptom matching
        for symptom in symptoms:
            if symptom in text_lower:
                score += 3
                matched.append(symptom)

        # keyword bonus
        important_words = ["thirst", "urine", "urination", "weakness", "diabetes", "emaciation"]

        for word in important_words:
            if word in text_lower:
                score += 1

        description = " ".join(lines[1:6])

        ranked_results.append({
            "medicine_name": medicine_name,
            "description": description,
            "score": score,
            "matched_symptoms": matched
        })

    # remove weak matches
    ranked_results = [r for r in ranked_results if r["score"] > 1]

    # sort best remedies
    ranked_results = sorted(ranked_results, key=lambda x: x["score"], reverse=True)

    return {
        "query": request.symptoms,
        "results": ranked_results[:5]
    }