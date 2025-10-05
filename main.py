# main.py
import os
import json
import re
from typing import List, Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_prompt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

SYSTEM_PROMPT_CONVERSATIONAL = load_prompt("prompts/system_prompt_conversational.txt")
SYSTEM_PROMPT_TEMPLATE = load_prompt("prompts/system_prompt_template.txt")

class Message(BaseModel):
    role: str = Field(..., example="user", description="Role of the message sender ('user' or 'assistant')")
    content: str = Field(..., example="Help me build a morning focus routine", description="The text of the message")

class ChatRequest(BaseModel):
    session_id: str = Field(..., example="session-12345", description="Unique session ID for this chat")
    messages: List[Message] = Field(default=[], description="Conversation history (max last 5 messages)")
    message: str = Field(..., example="Help me plan a daily study routine", description="New user message")
    force_template: Optional[bool] = Field(False, example=False, description="Force JSON output mode if true")

class Reminder(BaseModel):
    day_of_week: Optional[str] = Field(None, example="Monday", description="Day of the week (null = daily)")
    frequency: str = Field(..., example="daily", description="Reminder frequency: daily, weekly, or custom")
    time: str = Field(..., example="07:00:00", description="Time of day in HH:MM:SS")
    message: str = Field(..., example="Morning reminder", description="Reminder message")

class ChatResponse(BaseModel):
    reply_text: Optional[str] = Field(None, description="Assistant reply (in conversational mode)")
    create_goal: Optional[Dict[str, Any]] = Field(None, description="Goal definition, if generated")
    templates: Optional[List[Dict[str, Any]]] = Field(None, description="List of generated templates")
    action_items: Optional[List[Dict[str, Any]]] = Field(None, description="Generated actionable tasks")
    deadline: Optional[str] = Field(None, example="2025-12-27 10:00:00", description="Goal deadline if specified")
    reminders: Optional[List[Reminder]] = Field(None, description="List of reminder objects")
    meta: Dict[str, Any] = Field(..., description="Metadata including tokens used, model name, etc")

app = FastAPI(
    title="Posterio GenAI Chat API",
    description="""
    ### 🧠 What this API does
    Posterio is an AI-powered productivity assistant designed to help users achieve their goals.
    It can:
    - Have **natural conversations** to coach and motivate users.
    - Generate **structured JSON templates** for goals, reminders, and action plans.
    
    Use the `/chat` endpoint to send user messages.  
    Toggle between conversational or structured (template) output using `force_template`.

    ---
    **Example Use Cases:**
    - Build a morning routine  
    - Plan a 3-month fitness goal  
    - Schedule daily reminders for focus tasks
    """,
    version="1.2.0",
    contact={
        "name": "Posterio AI",
        "url": "https://posterio.ai",
        "email": "support@posterio.ai",
    },
)

@app.get("/", summary="Health Check", tags=["System"])
async def health():
    """Simple health check endpoint."""
    return {"message": "Posterio API is live. Try POST /chat"}

# ---- JSON PARSER ----
def extract_and_fix_json(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {"error": "No JSON detected", "raw_output": raw}
        text = m.group(0)
        text = text.replace("“", '"').replace("”", '"')
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        try:
            return json.loads(text)
        except Exception:
            return {"error": "JSON parse failed", "raw_output": raw}

# ---- ENDPOINT ----
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    """
    Engage with Posterio AI.  
    - **Default Mode:** Conversational (friendly chat)
    - **Template Mode:** Returns structured JSON (set `force_template=True`)
    """
    history = req.messages[-5:] if req.messages else []
    system_prompt = SYSTEM_PROMPT_TEMPLATE if req.force_template or "template" in req.message.lower() else SYSTEM_PROMPT_CONVERSATIONAL

    conversation = [{"role": "system", "content": system_prompt}]
    conversation += [{"role": m.role, "content": m.content} for m in history]
    conversation.append({"role": "user", "content": req.message})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation,
            max_tokens=700,
            temperature=0.6
        )
        raw_output = response.choices[0].message.content
        tokens_used = getattr(response.usage, "total_tokens", 0)

        if req.force_template or "template" in req.message.lower():
            output = extract_and_fix_json(raw_output)
        else:
            output = {"reply_text": raw_output}

        output["meta"] = {"tokens_used": tokens_used, "model": "gpt-4o-mini"}
        return output
    except Exception as e:
        return {"reply_text": f"Error: {str(e)}", "meta": {"tokens_used": 0, "model": "error"}}

