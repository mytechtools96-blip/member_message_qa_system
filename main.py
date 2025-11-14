
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer
import spacy
import re
from dateutil import parser as date_parser

# -------------------------
# LIFESPAN: Load FAISS + embeddings ONCE
# -------------------------

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):

    global messages, embeddings, index, idx_by_user, model, nlp

    # Load saved messages
    with open("messages_saved.json","r") as f:
        messages = json.load(f)

    # Load numpy embeddings
    embeddings = np.load("message_embeddings.npy")

    # Load FAISS index
    index = faiss.read_index("faiss_index.ivf")

    # Load idx_by_user
    with open("idx_by_user.json","r") as f:
        idx_by_user = json.load(f)

    # Load embedding model + spaCy NER
    model = SentenceTransformer("all-MiniLM-L6-v2")
    nlp = spacy.load("en_core_web_sm")

    yield


app = FastAPI(lifespan=lifespan)


# -------------------------
# Retrieval: SAME AS NOTEBOOK
# -------------------------
def retrieve(query, top_k=5):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_norm = q_emb / np.linalg.norm(q_emb)

    emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sims = (q_norm @ emb_norm.T)[0]

    best = sims.argsort()[-top_k:][::-1]
    return [(int(i), float(sims[i])) for i in best]


# -------------------------
# Extractors (same as notebook Step 10)
# -------------------------
def extract_dates(text):
    doc = nlp(text)
    out = []
    for ent in doc.ents:
        if ent.label_ == "DATE":
            try:
                out.append(str(date_parser.parse(ent.text, fuzzy=True).date()))
            except:
                out.append(ent.text)
    return list(set(out))

def extract_numbers(text):
    return list(set([int(x) for x in re.findall(r"\\b\\d+\\b", text)]))

def extract_locations(text):
    doc = nlp(text)
    return list(set([ent.text for ent in doc.ents if ent.label_ in ("GPE","LOC")]))


def detect_intent(q):
    ql = q.lower()
    if "how many" in ql: return "how_many"
    if "when" in ql: return "when"
    if "where" in ql: return "where"
    if "restaurant" in ql: return "restaurant"
    if "trip" in ql or "travel" in ql: return "travel"
    return "general"


def synthesize(question, retrieved):
    messages_text = [messages[i]["message"] for i,_ in retrieved]

    dates, nums, locs = [], [], []
    for t in messages_text:
        dates.extend(extract_dates(t))
        nums.extend(extract_numbers(t))
        locs.extend(extract_locations(t))

    intent = detect_intent(question)

    if intent == "when":
        return ", ".join(dates) if dates else "No date found."
    if intent == "how_many":
        return str(nums) if nums else "No numbers found."
    if intent == "where":
        return ", ".join(locs) if locs else "No locations found."
    if intent == "restaurant":
        return "Restaurant details not clearly mentioned."
    if intent == "travel":
        if locs: return ", ".join(locs)
        if dates: return ", ".join(dates)
        return "No travel info found."

    return messages_text[0] if messages_text else "No relevant info found."


# -------------------------
# API Model
# -------------------------
class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Query):
    try:
        ret = retrieve(q.question)
        ans = synthesize(q.question, ret)
        return {"answer": ans}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
