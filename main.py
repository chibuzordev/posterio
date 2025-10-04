# main.py
import os, json, re
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(
    title="Posterio AI Chat API",
    description="""
Posterio is an AI-powered productivity assistant that helps users achieve their goals through
personalized insights, templates, and reminders.

### Core Endpoints
- **POST /chat** → Conversational and Template-based goal planning assistant  
- **GET /** → Health check  
- **GET /docs** → Swagger UI  
- **GET /redoc** → Redoc documentation  

**Developer Note:**  
This version is running on Render and connects to the OpenAI API (`gpt-4o-mini` by default).
""",
    version="1.0.0",
    contact={
        "name": "DecisionSpaak AI Team",
        "url": "https://www.decisionspaak.com",
        "email": "decisionspaak@gmail.com",
    },
    license_info={
        "name": "Proprietary License - DecisionSpaak",
        "url": "https://www.decisionspaak.com/legal",
    },
)

# ---- Models ----
class Message(BaseModel):
    role: str = Field(..., example="user")
    content: str = Field(..., example="Help me build a morning focus routine")

class ChatRequest(BaseModel):
    session_id: str = Field(..., example="session-abc123")
    messages: List[Message] = Field(default=[], example=[{"role": "user", "content": "I want to be more productive"}])
    message: str = Field(..., example="Suggest a daily focus habit plan")
    force_template: Optional[bool] = Field(default=False, example=False)

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
            SYSTEM_PROMPT += """
            IMPORTANT:
            The user has requested TEMPLATE MODE.
            Respond ONLY in strict JSON using this schema:
            {
              "create_goal": null | {"goal": "string", "category": "string"},
              "templates": [{"title": "string", "text": "string"}],
              "action_items": [
                {"title": "string", "due": "YYYY-MM-DD HH:MI:SS"}
              ],
              "deadline": "YYYY-MM-DD HH:MI:SS",
              "reminders": [
                {"frequency": "daily|weekly|custom", "time": "HH:MM:SS", "message": "string"}
              ],
              "meta": {"tokens_used": 0}
            }
            DO NOT include explanations or normal dialogue.
            """
            try:
                output = json.loads(raw_output)
            except json.JSONDecodeError:
                cleaned = re.search(r"\{.*\}", raw_output, re.DOTALL)
                if cleaned:
                    try:
                        output = json.loads(cleaned.group())
                    except:
                        output = {"error": "Still invalid JSON", "raw": raw_output}
                else:
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



