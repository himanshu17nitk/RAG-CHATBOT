"""
Evaluation Router for RAGAS-based RAG evaluation
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
import time
import json
from datetime import datetime

from services.ragas_evaluator import RAGASEvaluator
from services.retriever import RetrieverService
from services.llm_client import RAGClient
from db.mongo_utils import MongoUtils
from utils.logger import api_logger
from config import CLIENT_SECRET_KEY

# Authentication dependency
async def verify_client_secret(client_secret: str = Header(..., alias="X-Client-Secret")):
    """Verify client secret key for API authentication."""
    if client_secret != CLIENT_SECRET_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid client secret key"
        )
    return client_secret


class EvaluationRequest(BaseModel):
    query: str
    session_id: str
    user_id: str
    include_metadata: Optional[bool] = True


class BatchEvaluationRequest(BaseModel):
    queries: List[str]
    session_id: str
    user_id: str
    include_metadata: Optional[bool] = True


class EvaluationResponse(BaseModel):
    query: str
    answer: str
    context_chunks: List[str]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    evaluation_time: float


class EvaluationReportResponse(BaseModel):
    evaluation_summary: Dict[str, Any]
    metrics_summary: Dict[str, Dict[str, float]]
    recommendations: List[str]
    generated_at: str


class EvaluationRouter:
    def __init__(self):
        self.evaluator = RAGASEvaluator()
        self.retriever = RetrieverService()
        self.llm_client = RAGClient()
        self.mongo_utils = MongoUtils()
    
    async def evaluate_single_query(
        self,
        query: str,
        session_id: str,
        user_id: str,
        include_metadata: bool = True
    ) -> EvaluationResponse:
        """
        Evaluate a single query using the RAG pipeline
        """
        start_time = time.time()
        
        try:
            api_logger.info(f"EVAL: Starting evaluation for query: {query[:100]}...")
            
            # Step 1: Retrieve context chunks
            context_chunks = self.retriever.retrieve_similar_chunks(query=query, k=5)
            
            # Step 2: Get chat history
            chat_history = self.mongo_utils.get_chat_history(session_id)
            
            # Step 3: Generate answer using LLM
            # Note: This is a simplified version - you might want to use your full prompt
            system_prompt = f"Context: {' '.join(context_chunks)}\n\nQuestion: {query}\n\nAnswer:"
            json_response = self.llm_client.chat(prompt=system_prompt)
            answer = self.llm_client.get_response(json_response)["response"]
            
            # Step 4: Prepare metadata
            metadata = {}
            if include_metadata:
                metadata = {
                    "session_id": session_id,
                    "user_id": user_id,
                    "chat_history_length": len(chat_history),
                    "chunks_retrieved": len(context_chunks),
                    "model": self.llm_client.get_response(json_response).get("model", "unknown"),
                    "input_tokens": self.llm_client.get_response(json_response).get("input_tokens", 0),
                    "output_tokens": self.llm_client.get_response(json_response).get("output_tokens", 0)
                }
            
            # Step 5: Evaluate using RAGAS
            evaluation_result = self.evaluator.evaluate_single_response(
                query=query,
                answer=answer,
                context_chunks=context_chunks,
                metadata=metadata
            )
            
            evaluation_time = time.time() - start_time
            
            return EvaluationResponse(
                query=query,
                answer=answer,
                context_chunks=context_chunks,
                metrics=evaluation_result.metrics,
                metadata=metadata,
                evaluation_time=evaluation_time
            )
            
        except Exception as e:
            api_logger.error(f"EVAL: Evaluation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Evaluation failed: {str(e)}"
            )
    
    async def evaluate_batch_queries(
        self,
        queries: List[str],
        session_id: str,
        user_id: str,
        include_metadata: bool = True
    ) -> List[EvaluationResponse]:
        """
        Evaluate multiple queries in batch
        """
        results = []
        
        for i, query in enumerate(queries):
            try:
                result = await self.evaluate_single_query(
                    query=query,
                    session_id=session_id,
                    user_id=user_id,
                    include_metadata=include_metadata
                )
                results.append(result)
                api_logger.info(f"EVAL: Completed evaluation {i+1}/{len(queries)}")
            except Exception as e:
                api_logger.error(f"EVAL: Failed to evaluate query {i+1}: {e}")
                continue
        
        return results
    
    def get_evaluation_report(self) -> EvaluationReportResponse:
        """
        Generate evaluation report from all evaluations
        """
        try:
            report = self.evaluator.generate_evaluation_report()
            
            return EvaluationReportResponse(
                evaluation_summary=report["evaluation_summary"],
                metrics_summary=report["metrics_summary"],
                recommendations=report["recommendations"],
                generated_at=report["generated_at"]
            )
            
        except Exception as e:
            api_logger.error(f"EVAL: Failed to generate report: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate report: {str(e)}"
            )
    
    def export_results(self, filepath: str) -> Dict[str, str]:
        """
        Export evaluation results to file
        """
        try:
            self.evaluator.export_evaluation_results(filepath)
            return {"message": f"Results exported to {filepath}", "filepath": filepath}
        except Exception as e:
            api_logger.error(f"EVAL: Failed to export results: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to export results: {str(e)}"
            )


# Create router instance
evaluation_router = EvaluationRouter()

# Create FastAPI router
router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_query(
    req: EvaluationRequest,
    client_secret: str = Depends(verify_client_secret)
):
    """
    Evaluate a single query using RAGAS metrics
    """
    return await evaluation_router.evaluate_single_query(
        query=req.query,
        session_id=req.session_id,
        user_id=req.user_id,
        include_metadata=req.include_metadata or True
    )


@router.post("/evaluate-batch", response_model=List[EvaluationResponse])
async def evaluate_batch(
    req: BatchEvaluationRequest,
    client_secret: str = Depends(verify_client_secret)
):
    """
    Evaluate multiple queries in batch using RAGAS metrics
    """
    return await evaluation_router.evaluate_batch_queries(
        queries=req.queries,
        session_id=req.session_id,
        user_id=req.user_id,
        include_metadata=req.include_metadata or True
    )


@router.get("/report", response_model=EvaluationReportResponse)
async def get_evaluation_report(
    client_secret: str = Depends(verify_client_secret)
):
    """
    Get comprehensive evaluation report
    """
    return evaluation_router.get_evaluation_report()


@router.post("/export")
async def export_evaluation_results(
    filepath: str = "evaluation_results.json",
    client_secret: str = Depends(verify_client_secret)
):
    """
    Export evaluation results to JSON file
    """
    return evaluation_router.export_results(filepath) 