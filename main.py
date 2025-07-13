from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Header
from pydantic import BaseModel
import uuid
from datetime import datetime
import time
from routers.predict_router import PredictRouter
from routers.train_router import TrainRouter
from config import CLIENT_SECRET_KEY
from utils.logger import api_logger

app = FastAPI(title="RAG API", description="RAG demo")

class PredictRequest(BaseModel):
    query: str
    session_id: str
    user_id: str
    caching_flag: Optional[bool] = True

class SessionRequest(BaseModel):
    user_id: str

class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    created_at: str
    status: str

# Authentication dependency
async def verify_client_secret(client_secret: str = Header(..., alias="X-Client-Secret")):
    """Verify client secret key for API authentication."""
    api_logger.debug(f"Verifying client secret key")
    if client_secret != CLIENT_SECRET_KEY:
        api_logger.warning(f"Invalid client secret key provided")
        raise HTTPException(
            status_code=401,
            detail="Invalid client secret key"
        )
    api_logger.debug(f"Client secret key verified successfully")
    return client_secret

@app.post("/create-session", response_model=SessionResponse)
async def create_session(
    req: SessionRequest,
    client_secret: str = Depends(verify_client_secret)
):
    """Create a new session for a user."""
    start_time = time.time()
    
    try:
        api_logger.info(f"API: Creating session for user: {req.user_id}")
        
        session_id = str(uuid.uuid4())
        api_logger.debug(f"API: Generated session ID: {session_id}")
        
        # You can store session info in MongoDB if needed
        # For now, we'll just return the session ID
        
        response = SessionResponse(
            session_id=session_id,
            user_id=req.user_id,
            created_at=datetime.utcnow().isoformat(),
            status="active"
        )
        
        response_time = time.time() - start_time
        api_logger.info(f"API: Session created successfully in {response_time:.3f}s | Session: {session_id} | User: {req.user_id}")
        
        return response
    except Exception as e:
        response_time = time.time() - start_time
        api_logger.error(f"API: Failed to create session for user {req.user_id} in {response_time:.3f}s", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create session: {str(e)}"
        )

@app.post("/train")
async def train(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    client_secret: str = Depends(verify_client_secret)
):
    """Upload a document; the router handles chunk→embed→store into vector database."""
    start_time = time.time()
    
    try:
        api_logger.info(f"API: Starting training for user: {user_id} | File: {file.filename} | Size: {file.size} bytes")
        
        train_router = TrainRouter()
        result = await train_router.train(user_id, file)
        
        response_time = time.time() - start_time
        api_logger.info(f"API: Training completed successfully in {response_time:.3f}s | User: {user_id} | File: {file.filename}")
        
        return result
    except Exception as e:
        response_time = time.time() - start_time
        api_logger.error(f"API: Training failed for user {user_id} in {response_time:.3f}s | File: {file.filename}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )

@app.post("/predict")
async def predict(
    req: PredictRequest,
    client_secret: str = Depends(verify_client_secret)
):
    """Embed query → search vectors → rerank → LLM → return answer."""
    start_time = time.time()
    
    try:
        api_logger.info(f"API: Starting prediction for user: {req.user_id} | Session: {req.session_id} | Query length: {len(req.query)}")
        api_logger.debug(f"API: Query preview: {req.query[:100]}{'...' if len(req.query) > 100 else ''}")
        
        predict_router = PredictRouter()
        result = await predict_router.predict_flow(query=req.query, session_id=req.session_id, user_id=req.user_id)
        
        response_time = time.time() - start_time
        api_logger.info(f"API: Prediction completed successfully in {response_time:.3f}s | User: {req.user_id} | Session: {req.session_id}")
        
        return result
    except Exception as e:
        response_time = time.time() - start_time
        api_logger.error(f"API: Prediction failed for user {req.user_id} in {response_time:.3f}s | Session: {req.session_id}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
