# backend/main.py
import os
import uuid
import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.cloud import firestore

# Configure Gemini API (must set GENAI_API_KEY as env var in runtime)
GENAI_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_KEY:
    raise RuntimeError("Set GENAI_API_KEY environment variable (Gemini API Key from Google AI Studio).")
genai.configure(api_key=GENAI_KEY)
client = genai.Client()

# Firestore client (Cloud Run can use default service account if authorized)
FIRESTORE_PROJECT = os.getenv("FIRESTORE_PROJECT")
save_enabled = bool(FIRESTORE_PROJECT)
if save_enabled:
    db = firestore.Client(project=FIRESTORE_PROJECT)

app = FastAPI(title="MoodM8 API")

# Simple risk keywords for immediate escalate (server backup of model safety)
RISK_KEYWORDS = [
    "suicide","kill myself","want to die","end my life",
    "hurt myself","hurting myself","die by suicide"
]

SYSTEM_PROMPT = (
    "You are MoodM8 — a short, empathetic, non-judgmental support assistant for young adults. "
    "Keep replies concise (2-6 sentences), ask one gentle clarifying question, offer a 1-2 simple coping step, "
    "and give safe resources if the user indicates crisis. Do NOT provide medical or legal advice. "
    "If the user expresses intent to self-harm or immediate danger, respond with supportive, urgent-escalation language and provide emergency resource suggestions."
)

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    history: Optional[List[dict]] = None  # [{"role":"user","text":"..."}, ...]
    save_opt_in_ciphertext: Optional[str] = None  # ciphertext produced client-side if user opts to save

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    timestamp: str
    saved: bool = False

def contains_risk(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in RISK_KEYWORDS)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    now = datetime.datetime.utcnow().isoformat() + "Z"

    # Server-side quick risk check (extra safety)
    if contains_risk(req.message):
        # Immediate crisis response (short, supportive)
        reply = (
            "I'm really sorry — it sounds like you're in serious distress. "
            "If you're in India and in immediate danger, please call 112 now or the mental health helpline KIRAN: 1800-599-0019. "
            "If you want, I can connect you with coping steps and nearby support options. You are not alone."
        )
        # do NOT store unless user opted in with ciphertext
        saved = False
        # If client provided ciphertext opt-in, store ciphertext only (server *cannot* decrypt)
        if req.save_opt_in_ciphertext and save_enabled:
            try:
                doc = {
                    "session_id": session_id,
                    "ciphertext": req.save_opt_in_ciphertext,
                    "created_at": now,
                    "opt_in": True
                }
                db.collection("saved_sessions").document(session_id).set(doc)
                saved = True
            except Exception as e:
                # don't fail conversation for logging errors
                print("Firestore save failed:", e)
        return ChatResponse(session_id=session_id, reply=reply, timestamp=now, saved=saved)

    # Build prompt for Gemini: system + history + new message
    parts = [SYSTEM_PROMPT]
    if req.history:
        for m in req.history:
            # simple plain text concatenation — role label helps
            parts.append(f"{m.get('role','user')}: {m.get('text')}")
    parts.append(f"user: {req.message}")
    prompt = "\n\n".join(parts)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",  # uses the free tier model family
            contents=prompt,
            max_output_tokens=400
        )
        # Some SDKs return response.text
        reply_text = getattr(response, "text", None) or (response.candidates[0].content[0].text if hasattr(response, "candidates") else str(response))
    except Exception as e:
        print("GenAI call failed:", e)
        raise HTTPException(status_code=500, detail="AI generation failed")

    saved = False
    if req.save_opt_in_ciphertext and save_enabled:
        try:
            doc = {
                "session_id": session_id,
                "ciphertext": req.save_opt_in_ciphertext,
                "created_at": now,
                "opt_in": True,
                "snippet": reply_text[:200]
            }
            db.collection("saved_sessions").document(session_id).set(doc)
            saved = True
        except Exception as e:
            print("Firestore save failed:", e)

    return ChatResponse(session_id=session_id, reply=reply_text.strip(), timestamp=now, saved=saved)
