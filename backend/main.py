import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
from typing import List, Optional, Dict, Any
import asyncio
import threading
import time
import logging
import json

from backend.config import CORS_ORIGINS, API_HOST, API_PORT, DATABASE_TYPE, DEBUG
from backend.schemas import (
    ChatCreate, ChatRename, ChatResponse, ChatDetailResponse,
    MessageCreate, MessageResponse, StreamMessage, HealthResponse,
    DiagnosisStartRequest, DiagnosisAnswerRequest, DiagnosisTurnResponse,
)
from src.core.chat_db import (
    init_db, add_chat, get_chats, get_messages, 
    add_message, rename_chat as db_rename_chat, 
    delete_chat as db_delete_chat, get_chat_by_id
)
from src.generation.generation import create_conversational_chain
from src.retrieval.retrieval import load_retriever_from_disk
from src.core.logging_utils import setup_logger, log_timing, log_query_start, log_query_complete
from src.cot.phase2_questionnaire_engine import (
    load_phase1_canonical_schema,
    build_phase2_initial_state,
    submit_phase2_answer,
    get_current_question,
    build_phase2_summary,
)
from src.cot.phase3_candidate_narrowing import update_state_with_phase3
from src.cot.phase4_final_decision import update_state_with_phase4

# Image analysis (CLIP-based, runs locally — zero API cost)
try:
    from src.clip.image_analyzer import CLIPDiseaseAnalyzer
    IMAGE_ANALYSIS_AVAILABLE = True
except ImportError:
    print("CLIP image analysis module not available. Install: pip install transformers torch")
    IMAGE_ANALYSIS_AVAILABLE = False

# Initialize logger
api_logger = setup_logger('api')

# Initialize FastAPI app
app = FastAPI(
    title="Aloo Sahayak API",
    description="Backend API for Potato Disease Assistant with RAG",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if isinstance(CORS_ORIGINS, list) else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
qa_chain = None
retriever = None
image_analyzer = None
phase_schema: Optional[Dict[str, Any]] = None
phase5_profile: Dict[str, Any] = {}

PHASE_DIAGNOSIS_STATE_KEY = "phase_diagnosis_state"


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _phase5_profile_path() -> str:
    return os.path.join(_project_root(), "data", "COT", "phase5_weight_profile.json")


def _load_phase5_profile_if_available() -> Dict[str, Any]:
    path = _phase5_profile_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _compact_phase_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Persist only compact state; Phase 03/04 can be recomputed from answers."""
    compact = {
        "version": state.get("version"),
        "chat_id": state.get("chat_id"),
        "created_at": state.get("created_at"),
        "updated_at": state.get("updated_at"),
        "completed": bool(state.get("completed", False)),
        "step_index": int(state.get("step_index", 0)),
        "factor_sequence": state.get("factor_sequence", []),
        "answers": state.get("answers", {}),
        "history": state.get("history", []),
    }
    # Keep phase4 decision if present and completed.
    if isinstance(state.get("phase4"), dict):
        compact["phase4"] = state.get("phase4")
    return compact


def _extract_latest_phase_state(chat_id: str) -> Optional[Dict[str, Any]]:
    messages = get_messages(chat_id)
    for row in reversed(messages):
        metadata = row[3] if len(row) > 3 else None
        if isinstance(metadata, dict) and isinstance(metadata.get(PHASE_DIAGNOSIS_STATE_KEY), dict):
            return metadata.get(PHASE_DIAGNOSIS_STATE_KEY)
    return None


def _persist_phase_message(chat_id: str, sender: str, content: str, state: Dict[str, Any]) -> None:
    add_message(
        chat_id,
        sender,
        content,
        metadata={
            PHASE_DIAGNOSIS_STATE_KEY: _compact_phase_state(state),
            "phase_diagnosis": True,
        },
    )


def _apply_phase5_profile(state: Dict[str, Any]) -> Dict[str, Any]:
    """Apply tuned Phase 05 profile on top of Phase 02 state after each answer."""
    global phase_schema, phase5_profile
    if phase_schema is None:
        return state

    weights_override = phase5_profile.get("factor_weights") if isinstance(phase5_profile, dict) else None
    scoring_cfg = phase5_profile.get("scoring_config") if isinstance(phase5_profile, dict) else None

    update_state_with_phase3(
        state,
        phase_schema,
        factor_weights_override=weights_override,
        scoring_config=scoring_cfg,
    )
    return state


def _ensure_phase_state(chat_id: str) -> Dict[str, Any]:
    global phase_schema
    prev = _extract_latest_phase_state(chat_id)
    state = build_phase2_initial_state(chat_id=chat_id, schema=phase_schema, existing_state=prev)
    _apply_phase5_profile(state)
    return state


def _diagnosis_turn_response(chat_id: str, message: str, state: Dict[str, Any]) -> DiagnosisTurnResponse:
    summary = build_phase2_summary(state)
    return DiagnosisTurnResponse(
        chat_id=chat_id,
        completed=bool(summary.get("completed", False)),
        message=message,
        next_question=get_current_question(state),
        progress=summary.get("progress", {}),
        phase3=summary.get("phase3", {}),
        phase4=summary.get("phase4", {}),
        state=summary,
    )


def _build_next_question_message(state: Dict[str, Any]) -> str:
    next_question = get_current_question(state)
    if not next_question:
        return "[diagnosis] No pending question."
    q_no = int(state.get("step_index", 0)) + 1
    return f"[diagnosis] Q{q_no}: {next_question}"


def _build_final_diagnosis_message(state: Dict[str, Any]) -> str:
    phase4 = state.get("phase4", {}) if isinstance(state.get("phase4"), dict) else {}
    disease = phase4.get("selected_disease_name") or phase4.get("selected_disease_id") or "Unknown"
    confidence = float(phase4.get("confidence", 0.0) or 0.0)
    return f"[diagnosis] Final Disease Suggestion: {disease} (confidence {confidence:.0%})"

# Startup and Shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize database and load RAG chain on startup"""
    global qa_chain, retriever, phase_schema, phase5_profile
    
    print(" Starting Aloo Sahayak API...")
    
    # Initialize database
    try:
        init_db()
        print(" Database initialized successfully")
    except Exception as e:
        print(f" Database initialization failed: {e}")
        raise
    
    # Load retriever and RAG chain
    try:
        retriever = load_retriever_from_disk()
        qa_chain = create_conversational_chain(retriever)
        print(" RAG chain loaded successfully")
    except Exception as e:
        print(f" RAG chain loading failed: {e}")
        raise
    
    # Load CLIP image analyzer (non-blocking — app works without it)
    global image_analyzer
    if IMAGE_ANALYSIS_AVAILABLE:
        try:
            image_analyzer = CLIPDiseaseAnalyzer(load_faiss_index=True)
            print(" CLIP image analyzer loaded successfully")
        except Exception as e:
            print(f" CLIP image analyzer failed (non-fatal): {e}")
            image_analyzer = None

    # Load Phase diagnosis assets (non-blocking — diagnosis APIs guard availability)
    try:
        phase_schema = load_phase1_canonical_schema()
        print(" Phase diagnosis schema loaded")
    except Exception as e:
        phase_schema = None
        print(f" Phase diagnosis schema load failed (non-fatal): {e}")

    phase5_profile = _load_phase5_profile_if_available()
    if phase5_profile:
        print(" Phase 05 calibration profile loaded")
    else:
        print(" Phase 05 calibration profile not found; using default scoring")
    
    print(" API startup complete!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print(" Shutting down Aloo Sahayak API...")

# ==================== Health Check ====================
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check API health status"""
    return HealthResponse(
        status="healthy",
        message="API is running",
        database=DATABASE_TYPE
    )

# ==================== Chat Endpoints ====================

@app.get("/api/chats", response_model=List[ChatResponse])
async def list_chats():
    """Get all chats"""
    try:
        chats_data = get_chats()
        chats = []
        
        for chat_id, name, created_at in chats_data:
            messages = get_messages(chat_id)
            chats.append(ChatResponse(
                id=chat_id,
                name=name,
                created_at=created_at,
                message_count=len(messages) // 2  # Divide by 2 (user + assistant)
            ))
        
        return chats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chats", response_model=ChatResponse)
async def create_chat(chat: ChatCreate):
    """Create a new chat"""
    try:
        chat_id = str(uuid.uuid4())
        add_chat(chat_id, chat.name)
        
        return ChatResponse(
            id=chat_id,
            name=chat.name,
            created_at=get_chats()[0][2],  # Get created_at from newly created chat
            message_count=0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chats/{chat_id}", response_model=ChatDetailResponse)
async def get_chat(chat_id: str):
    """Get chat details with messages"""
    try:
        chats = get_chats()
        chat_found = False
        chat_name = None
        created_at = None
        
        for cid, name, created in chats:
            if cid == chat_id:
                chat_found = True
                chat_name = name
                created_at = created
                break
        
        if not chat_found:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        messages = get_messages(chat_id)
        message_list = []
        for i, msg in enumerate(messages):
            sender = msg[0]
            content = msg[1]
            timestamp = msg[2]
            metadata = msg[3] if len(msg) > 3 else None
            message_list.append(
                MessageResponse(
                    id=i,
                    sender=sender,
                    content=content,
                    timestamp=timestamp,
                    metadata=metadata
                )
            )
        
        return ChatDetailResponse(
            id=chat_id,
            name=chat_name,
            created_at=created_at,
            messages=message_list
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/chats/{chat_id}", response_model=ChatResponse)
async def update_chat(chat_id: str, chat: ChatRename):
    """Rename a chat"""
    try:
        chats = get_chats()
        chat_found = False
        
        for cid, name, created in chats:
            if cid == chat_id:
                chat_found = True
                created_at = created
                break
        
        if not chat_found:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        db_rename_chat(chat_id, chat.new_name)
        
        return ChatResponse(
            id=chat_id,
            name=chat.new_name,
            created_at=created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat"""
    try:
        db_delete_chat(chat_id)
        return {"status": "success", "message": "Chat deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Phase Diagnosis Endpoints ====================

@app.post("/api/chats/{chat_id}/diagnosis/start", response_model=DiagnosisTurnResponse)
async def diagnosis_start(chat_id: str, request: DiagnosisStartRequest):
    """Start or resume deterministic phase diagnosis workflow for a chat."""
    global phase_schema
    try:
        if phase_schema is None:
            raise HTTPException(status_code=503, detail="Phase diagnosis schema is not available")

        state = _ensure_phase_state(chat_id)
        message = "Diagnosis session started."

        if request.initial_observation and not state.get("completed", False):
            add_message(
                chat_id,
                "user",
                f"[diagnosis] {request.initial_observation}",
                metadata={"phase_diagnosis": True},
            )
            message = "Initial observation recorded."

        assistant_msg = _build_next_question_message(state)
        _persist_phase_message(chat_id, "assistant", assistant_msg, state)
        return _diagnosis_turn_response(chat_id, assistant_msg.replace("[diagnosis] ", "", 1), state)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chats/{chat_id}/diagnosis/answer", response_model=DiagnosisTurnResponse)
async def diagnosis_answer(chat_id: str, request: DiagnosisAnswerRequest):
    """Submit one answer for the current factor in diagnosis workflow."""
    global phase_schema
    try:
        if phase_schema is None:
            raise HTTPException(status_code=503, detail="Phase diagnosis schema is not available")

        state = _ensure_phase_state(chat_id)
        if state.get("completed", False):
            _persist_phase_message(chat_id, "assistant", "[diagnosis] already completed", state)
            return _diagnosis_turn_response(chat_id, "Questionnaire already completed.", state)

        submit_phase2_answer(
            state,
            request.answer,
            schema=phase_schema,
            use_phase4_llm=request.use_llm,
        )

        # Re-apply tuned calibration after each turn.
        _apply_phase5_profile(state)

        if state.get("completed", False):
            update_state_with_phase4(state, use_llm=request.use_llm)

        add_message(
            chat_id,
            "user",
            f"[diagnosis] {request.answer}",
            metadata={"phase_diagnosis": True},
        )

        if state.get("completed", False):
            assistant_msg = _build_final_diagnosis_message(state)
        else:
            assistant_msg = _build_next_question_message(state)

        _persist_phase_message(chat_id, "assistant", assistant_msg, state)
        return _diagnosis_turn_response(chat_id, assistant_msg.replace("[diagnosis] ", "", 1), state)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chats/{chat_id}/diagnosis/state", response_model=DiagnosisTurnResponse)
async def diagnosis_state(chat_id: str):
    """Fetch latest persisted diagnosis state for a chat."""
    try:
        state = _extract_latest_phase_state(chat_id)
        if not state:
            raise HTTPException(status_code=404, detail="Diagnosis session not started")
        return _diagnosis_turn_response(chat_id, "Diagnosis state loaded.", state)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Message Endpoints ====================

@app.get("/api/chats/{chat_id}/messages", response_model=List[MessageResponse])
async def get_chat_messages(chat_id: str):
    """Get all messages for a chat"""
    try:
        messages = get_messages(chat_id)
        
        result = []
        for i, msg in enumerate(messages):
            sender = msg[0]
            content = msg[1]
            timestamp = msg[2]
            metadata = msg[3] if len(msg) > 3 else None
            result.append(MessageResponse(id=i, sender=sender, content=content, timestamp=timestamp, metadata=metadata))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chats/{chat_id}/messages")
async def send_message(chat_id: str, message: MessageCreate):
    """Send message and get AI response (non-streaming)"""
    request_id = str(uuid.uuid4())[:8]
    request_start = time.perf_counter()
    
    try:
        log_query_start(api_logger, request_id, message.content[:100])
        
        if qa_chain is None:
            raise HTTPException(status_code=500, detail="RAG chain not loaded")
        
        user_query = message.content
        
        # Add user message to database
        db_start = time.perf_counter()
        add_message(
            chat_id,
            "user",
            user_query,
            metadata=None,
        )
        db_elapsed = time.perf_counter() - db_start
        
        log_timing(api_logger, "DB_ADD_USER_MESSAGE", {
            'duration_ms': round(db_elapsed * 1000, 2)
        })
        
        # Get chat history for context
        hist_start = time.perf_counter()
        messages = get_messages(chat_id)
        chat_history = [
            (msg[0], msg[1]) 
            for msg in messages[:-1]  # Exclude latest user message
            if msg[0] in ["user", "assistant"]
        ]
        
        # Reconstruct alternating pairs for context
        reconstructed_history = []
        for i in range(0, len(chat_history), 2):
            if i + 1 < len(chat_history):
                reconstructed_history.append((chat_history[i][1], chat_history[i+1][1]))
        
        hist_elapsed = time.perf_counter() - hist_start
        log_timing(api_logger, "HISTORY_RETRIEVAL", {
            'duration_ms': round(hist_elapsed * 1000, 2),
            'history_items': len(reconstructed_history)
        })
        
        # Add language instruction
        if message.language == "Hindi":
            query = user_query + "\n\nIMPORTANT: Respond to this question in Hindi (हिंदी में उत्तर दें)."
        else:
            query = user_query
        
        # Get AI response (measure time)
        chain_start = time.perf_counter()
        result = qa_chain.invoke({
            "question": query,
            "chat_history": reconstructed_history
        })
        chain_elapsed = time.perf_counter() - chain_start
        
        log_timing(api_logger, "AI_CHAIN_INVOCATION", {
            'duration_ms': round(chain_elapsed * 1000, 2)
        })
        
        ai_response = result.get("answer", "")
        source_docs = result.get("source_documents", [])
        
        # Serialize source documents to JSON-friendly structure (force JSON-safe types)
        serialized_sources = []
        for doc in source_docs:
            try:
                raw_meta = doc.metadata if hasattr(doc, 'metadata') else {}
                src = raw_meta.get('source', None)
                page_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                if src is not None:
                    src = str(src)
                safe_meta = {}
                for k, v in (raw_meta or {}).items():
                    try:
                        json.dumps(v)
                        safe_meta[str(k)] = v
                    except (TypeError, ValueError):
                        safe_meta[str(k)] = str(v)
            except Exception:
                src = None
                safe_meta = {}
                page_content = str(doc)

            serialized_sources.append({
                "source": src,
                "page_content": page_content,
                "metadata": safe_meta
            })

        # Add AI message to database
        db_ai_start = time.perf_counter()
        assistant_metadata = {"source_documents": serialized_sources}
        add_message(chat_id, "assistant", ai_response, metadata=assistant_metadata)
        db_ai_elapsed = time.perf_counter() - db_ai_start
        
        log_timing(api_logger, "DB_ADD_AI_MESSAGE", {
            'duration_ms': round(db_ai_elapsed * 1000, 2),
            'response_length': len(ai_response)
        })

        total_time = time.perf_counter() - request_start
        log_query_complete(api_logger, request_id, total_time, "SUCCESS")

        return {
            "user_message": user_query,
            "ai_response": ai_response,
            "sources_count": len(serialized_sources),
            "source_documents": serialized_sources,
            "language": message.language,
            "timings": {"total_ms": round(total_time * 1000, 2)}
        }
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.perf_counter() - request_start
        log_query_complete(api_logger, request_id, total_time, f"FAILED: {type(e).__name__}")
        api_logger.error(f"Error in send_message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WebSocket for Streaming ====================

@app.websocket("/api/ws/chats/{chat_id}/stream")
async def websocket_stream(websocket: WebSocket, chat_id: str):
    """WebSocket endpoint for real-time message streaming"""
    request_id = str(uuid.uuid4())[:8]
    request_start = time.perf_counter()
    
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            user_query = data.get("message", "")
            language = data.get("language", "English")
            
            if not user_query:
                await websocket.send_json({
                    "type": "error",
                    "error": "Message content is required"
                })
                continue
            
            log_query_start(api_logger, request_id, user_query[:100])
            
            # Add user message to database
            db_start = time.perf_counter()
            add_message(
                chat_id,
                "user",
                user_query,
                metadata=None,
            )
            db_elapsed = time.perf_counter() - db_start
            
            log_timing(api_logger, "DB_ADD_USER_WS", {
                'duration_ms': round(db_elapsed * 1000, 2)
            })
            
            # Notify client that user message received
            await websocket.send_json({
                "type": "user_received",
                "message": user_query
            })

            # FIX 6: Immediately signal the frontend that processing has started.
            # This eliminates the blank-wait UX during the condense + retrieval phase
            # (which can take 1-6 seconds before the first answer token arrives).
            await websocket.send_json({
                "type": "thinking",
                "message": "Searching knowledge base..."
            })

            # Get chat history
            hist_start = time.perf_counter()
            messages = get_messages(chat_id)
            chat_history = [
                (msg[0], msg[1])
                for msg in messages[:-1]  # Exclude latest user message
                if msg[0] in ["user", "assistant"]
            ]
            
            # Reconstruct alternating pairs
            reconstructed_history = []
            for i in range(0, len(chat_history), 2):
                if i + 1 < len(chat_history):
                    reconstructed_history.append((chat_history[i][1], chat_history[i+1][1]))
            
            hist_elapsed = time.perf_counter() - hist_start
            log_timing(api_logger, "HISTORY_RETRIEVAL_WS", {
                'duration_ms': round(hist_elapsed * 1000, 2),
                'history_items': len(reconstructed_history)
            })
            
            # Add language instruction
            if language == "Hindi":
                query = user_query + "\n\nIMPORTANT: Respond to this question in Hindi (हिंदी में उत्तर दें)."
            else:
                query = user_query
            
            # Stream the response and measure timings
            stream_start = time.perf_counter()
            full_response = ""
            source_docs = []
            first_chunk_time = None
            chunk_count = 0

            # Use an asyncio.Queue to receive chunks from a background thread.
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def run_blocking_stream():
                try:
                    for stream_output in qa_chain.stream({
                        "question": query,
                        "chat_history": reconstructed_history
                    }):
                        # Push each chunk into the asyncio queue from this thread
                        loop.call_soon_threadsafe(queue.put_nowait, stream_output)
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "content": str(e)})

            # Start background thread for blocking streaming work
            thread = threading.Thread(target=run_blocking_stream, daemon=True)
            thread.start()

            try:
                while True:
                    # Wait for the next chunk from the background thread
                    stream_output = await queue.get()

                    if stream_output.get('type') == 'chunk':
                        chunk_content = stream_output['content']
                        chunk_count += 1
                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter() - stream_start
                        full_response += chunk_content
                        await websocket.send_json({
                            "type": "chunk",
                            "content": chunk_content
                        })

                    elif stream_output.get('type') == 'complete':
                        full_response = stream_output.get('answer', full_response)
                        source_docs = stream_output.get('source_documents', [])
                        total_stream_elapsed = time.perf_counter() - stream_start

                        # Serialize source documents to JSON-friendly structure
                        serialized_sources = []
                        for doc in source_docs:
                            try:
                                raw_meta = doc.metadata if hasattr(doc, 'metadata') else {}
                                src = raw_meta.get('source', None)
                                page_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                                if src is not None:
                                    src = str(src)
                                safe_meta = {}
                                for k, v in (raw_meta or {}).items():
                                    try:
                                        json.dumps(v)
                                        safe_meta[str(k)] = v
                                    except (TypeError, ValueError):
                                        safe_meta[str(k)] = str(v)
                            except Exception:
                                src = None
                                safe_meta = {}
                                page_content = str(doc)

                            serialized_sources.append({
                                "source": src,
                                "page_content": page_content,
                                "metadata": safe_meta
                            })

                        log_timing(api_logger, "STREAMING_COMPLETE", {
                            'duration_ms': round(total_stream_elapsed * 1000, 2),
                            'chunks': chunk_count,
                            'response_length': len(full_response),
                            'first_chunk_ms': round(first_chunk_time * 1000, 2) if first_chunk_time else None
                        })

                        await websocket.send_json({
                            "type": "complete",
                            "content": full_response,
                            "sources_count": len(serialized_sources),
                            "source_documents": serialized_sources,
                            "timings": {
                                "first_chunk_ms": round(first_chunk_time * 1000, 2) if first_chunk_time else None,
                                "total_ms": round(total_stream_elapsed * 1000, 2)
                            }
                        })
                        break

                    elif stream_output.get('type') == 'error':
                        await websocket.send_json({
                            "type": "error",
                            "error": stream_output.get('content')
                        })
                        break

                # Add AI response to database after streaming completes (include sources metadata)
                db_ai_start = time.perf_counter()
                assistant_metadata = {"source_documents": serialized_sources}
                add_message(chat_id, "assistant", full_response, metadata=assistant_metadata)
                db_ai_elapsed = time.perf_counter() - db_ai_start
                
                log_timing(api_logger, "DB_ADD_AI_WS", {
                    'duration_ms': round(db_ai_elapsed * 1000, 2),
                    'response_length': len(full_response)
                })
                
                total_elapsed = time.perf_counter() - request_start
                log_query_complete(api_logger, request_id, total_elapsed, "SUCCESS")

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "error": f"Error generating response: {str(e)}"
                })
                total_elapsed = time.perf_counter() - request_start
                log_query_complete(api_logger, request_id, total_elapsed, f"FAILED: {type(e).__name__}")
    
    except WebSocketDisconnect:
        api_logger.info(f"WebSocket client disconnected - request_id={request_id}")
    except Exception as e:
        api_logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass

# ==================== Image Analysis Endpoint ====================



from fastapi import File, UploadFile, Form
from PIL import Image as PILImage
import io

@app.post("/api/analyze-image")
async def analyze_image(
    image: UploadFile = File(...),
    chat_id: str = Form(...),
    language: str = Form("English"),
    trigger_rag: bool = Form(True),
):
    """
    Analyze an uploaded potato image using CLIP reference matching.
    Returns disease prediction + confidence + optional RAG explanation.
    Classification runs locally (zero API cost). Only the RAG answer uses LLM.
    """
    request_id = str(uuid.uuid4())[:8]
    request_start = time.perf_counter()

    try:
        if not IMAGE_ANALYSIS_AVAILABLE or image_analyzer is None:
            raise HTTPException(
                status_code=503,
                detail="Image analysis not available. Install transformers+torch and run build_reference_index."
            )

        # Read and validate image
        image_bytes = await image.read()
        try:
            pil_image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        log_query_start(api_logger, request_id, f"IMAGE_ANALYSIS: {image.filename}")

        # CLIP analysis (runs locally — $0)
        analysis_start = time.perf_counter()
        analysis = image_analyzer.analyze_image(pil_image)
        analysis_elapsed = time.perf_counter() - analysis_start

        log_timing(api_logger, "CLIP_IMAGE_ANALYSIS", {
            'duration_ms': round(analysis_elapsed * 1000, 2),
            'prediction': analysis['display_name'],
            'confidence': analysis['confidence'],
        })

        # Build response
        response_data = {
            "prediction": analysis['display_name'],
            "confidence": analysis['confidence'],
            "top_candidates": analysis['all_candidates'],
            "matched_ref_images": [
                {
                    "image_path": img['image_path'],
                    "disease": img['display_name'],
                    "similarity_score": round(img['similarity_score'], 4),
                }
                for img in analysis['matched_ref_images'][:5]
            ],
            "rag_query": analysis['rag_query'],
            "rag_response": None,
            "source_documents": [],
        }

        # Optionally trigger RAG for detailed explanation (1 LLM call)
        if trigger_rag and qa_chain is not None:
            rag_query = analysis['rag_query']
            if language == "Hindi":
                rag_query += "\n\nIMPORTANT: Respond in Hindi (\u0939\u093f\u0902\u0926\u0940 \u092e\u0947\u0902 \u0909\u0924\u094d\u0924\u0930 \u0926\u0947\u0902)."

            # Save user message — include base64 image + CLIP analysis in metadata
            import base64 as _b64
            image_b64 = _b64.b64encode(image_bytes).decode('utf-8')
            user_msg = f"[Image uploaded: {image.filename}]\n{analysis['rag_query']}"
            user_meta = {
                "image_analysis": True,
                "image_b64": image_b64,
                "image_filename": image.filename,
            }
            add_message(chat_id, "user", user_msg, metadata=user_meta)

            # Get chat history
            messages_db = get_messages(chat_id)
            chat_history_raw = [
                (msg[0], msg[1])
                for msg in messages_db[:-1]
                if msg[0] in ["user", "assistant"]
            ]
            reconstructed_history = []
            for i in range(0, len(chat_history_raw), 2):
                if i + 1 < len(chat_history_raw):
                    reconstructed_history.append((chat_history_raw[i][1], chat_history_raw[i+1][1]))

            # Invoke RAG chain
            chain_start = time.perf_counter()
            result = qa_chain.invoke({
                "question": rag_query,
                "chat_history": reconstructed_history
            })
            log_timing(api_logger, "IMAGE_RAG_CHAIN", {
                'duration_ms': round((time.perf_counter() - chain_start) * 1000, 2)
            })

            ai_response = result.get("answer", "")
            source_docs = result.get("source_documents", [])

            # Serialize sources
            serialized_sources = []
            for doc in source_docs:
                try:
                    raw_meta = doc.metadata if hasattr(doc, 'metadata') else {}
                    src = raw_meta.get('source', None)
                    page_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    safe_meta = {}
                    for k, v in (raw_meta or {}).items():
                        try:
                            json.dumps(v)
                            safe_meta[str(k)] = v
                        except (TypeError, ValueError):
                            safe_meta[str(k)] = str(v)
                except Exception:
                    src = None
                    safe_meta = {}
                    page_content = str(doc)
                serialized_sources.append({
                    "source": src,
                    "page_content": page_content,
                    "metadata": safe_meta
                })

            assistant_meta = {
                "source_documents": serialized_sources,
                "image_analysis": True,
                "prediction": analysis['display_name'],
                "confidence": analysis['confidence'],
                "top_candidates": analysis['all_candidates'],
                "matched_ref_images": [
                    {
                        "image_path": img['image_path'],
                        "disease": img['display_name'],
                        "similarity_score": round(img['similarity_score'], 4),
                    }
                    for img in analysis['matched_ref_images'][:5]
                ],
            }
            add_message(chat_id, "assistant", ai_response, metadata=assistant_meta)

            response_data["rag_response"] = ai_response
            response_data["source_documents"] = serialized_sources

        total_time = time.perf_counter() - request_start
        log_query_complete(api_logger, request_id, total_time, "SUCCESS")
        response_data["timings"] = {"total_ms": round(total_time * 1000, 2)}
        return response_data

    except HTTPException:
        raise
    except Exception as e:
        total_time = time.perf_counter() - request_start
        log_query_complete(api_logger, request_id, total_time, f"FAILED: {type(e).__name__}")
        api_logger.error(f"Error in analyze_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail
        },
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "detail": "Internal server error"
        },
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Aloo Sahayak API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    
    print(f"Starting API on {API_HOST}:{API_PORT}")
    print(f"Database: {DATABASE_TYPE}")
    print(f"Debug: {DEBUG}")
    print(f"API Documentation: http://{API_HOST}:{API_PORT}/api/docs")
    
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        debug=DEBUG,
        reload=DEBUG
    )