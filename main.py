# main.py
import os, json
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Posterio AI API", version="1.0")

# ---- Models ----
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    messages: List[Message] = []
    message: str
    force_template: Optional[bool] = False

# ---- Endpoint ----
@app.post("/chat")
async def chat(req: ChatRequest):
    SYSTEM_PROMPT = """
    You are Posterio, an AI productivity assistant.
    By default, respond conversationally like a coach.
    If the user explicitly asks for a template or `force_template=true`,
    respond in valid JSON with goal, templates, action_items, deadline, reminders, and meta.
    """

    # Retain last 5 messages
    history = req.messages[-5:] if req.messages else []
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    conversation += [{"role": m.role, "content": m.content} for m in history]
    conversation.append({"role": "user", "content": req.message})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation,
            max_tokens=600,
            temperature=0.7
        )
        raw_output = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        if req.force_template or "template" in req.message.lower():
            try:
                output = json.loads(raw_output)
            except json.JSONDecodeError:
                output = {"error": "Invalid JSON returned", "raw": raw_output}
        else:
            output = {"reply_text": raw_output}

        output["meta"] = {"tokens_used": tokens_used, "model": "gpt-4o-mini"}
        return output

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "Posterio AI API is live! Use POST /chat to interact."}


