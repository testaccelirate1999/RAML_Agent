# raml_server.py
# Run: uvicorn raml_server:app --reload --port 8001

import os, sys, json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from raml_agent import RAMLAgent

app = FastAPI(title="RAML Generation API", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_agent: Optional[RAMLAgent] = None
def get_agent() -> RAMLAgent:
    global _agent
    if _agent is None:
        _agent = RAMLAgent(verbose=True)
    return _agent

def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

# ── Models ────────────────────────────────────────────────────────────────────
class CreateSessionRequest(BaseModel):
    project_name: str

class ChatRequest(BaseModel):
    message: str

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

# ── Sessions ──────────────────────────────────────────────────────────────────
@app.post("/sessions")
def create_session(req: CreateSessionRequest):
    return get_agent().create_session(project_name=req.project_name).to_dict()

@app.get("/sessions")
def list_sessions():
    return {"sessions": get_agent().list_sessions()}

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    agent = get_agent()
    if not agent.get_session(session_id):
        raise HTTPException(404, "Session not found")
    agent.delete_session(session_id)
    return {"deleted": session_id}

# ── Chat (SSE) ────────────────────────────────────────────────────────────────
@app.post("/sessions/{session_id}/chat")
def chat(session_id: str, req: ChatRequest):
    agent = get_agent()
    if not agent.get_session(session_id):
        raise HTTPException(404, "Session not found")

    def stream():
        try:
            session  = agent.get_session(session_id)
            is_first = len(session.history) == 0
            yield sse({"type": "status",
                       "msg": "Retrieving context..." if is_first else "Applying feedback..."})

            result = agent.chat(session_id=session_id, message=req.message)

            yield sse({
                "type":         "sources",
                "sources":      result["sources"],
                "lessons_used": result.get("lessons_used", []),
            })
            yield sse({
                "type":          "chunk",
                "message":       result["message"],
                "files":         result["files"],
                "changed_files": result["changed_files"],
                "deleted_files": result.get("deleted_files", []),
                "is_first_turn": result["is_first_turn"],
            })
            # If a new lesson was silently saved, tell the UI to refresh lessons panel
            if result.get("lesson_saved"):
                yield sse({"type": "lesson_saved", "lesson": result["lesson_saved"]})

            yield sse({"type": "done", "tokens_used": result["tokens_used"],
                       "session": result["session"]})
        except Exception as e:
            yield sse({"type": "error", "msg": str(e)})

    return StreamingResponse(stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── File access ───────────────────────────────────────────────────────────────
@app.get("/sessions/{session_id}/files")
def get_files(session_id: str):
    session = get_agent().get_session(session_id)
    if not session: raise HTTPException(404, "Session not found")
    return {"files": [{"path": p, "size": len(c), "lines": c.count('\n')+1}
                      for p, c in session.files.items()]}

@app.get("/sessions/{session_id}/files/{file_path:path}")
def get_file(session_id: str, file_path: str):
    try:
        content = get_agent().get_file(session_id, file_path)
        return {"path": file_path, "content": content, "lines": content.count('\n')+1}
    except FileNotFoundError:
        raise HTTPException(404, f"'{file_path}' not found")

@app.get("/sessions/{session_id}/download")
def download_zip(session_id: str):
    agent   = get_agent()
    session = agent.get_session(session_id)
    if not session: raise HTTPException(404, "Session not found")
    zip_bytes = agent.get_zip(session_id)
    filename  = f"{session.project_name.replace(' ','-')}.zip"
    return Response(content=zip_bytes, media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"})

# ── Lessons (view + delete only — saving is fully automatic) ─────────────────
@app.get("/lessons")
def list_lessons():
    agent = get_agent()
    if not agent._lessons_ready:
        return {"lessons": [], "count": 0, "warning": "Lesson memory not connected"}
    lessons = agent._lesson_memory.list_all()
    return {"lessons": lessons, "count": len(lessons)}

@app.delete("/lessons/{lesson_id}")
def delete_lesson(lesson_id: str):
    agent = get_agent()
    if not agent._lessons_ready:
        raise HTTPException(503, "Lesson memory not available")
    agent._lesson_memory.delete(lesson_id)
    return {"deleted": lesson_id}