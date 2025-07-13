from typing import Dict, Any, List
from datetime import datetime
import json
import time

from services.llm_client import RAGClient, RAGError, RAGAPIError
from services.retriever import RetrieverService
from db.mongo_utils import MongoUtils
from utils.logger import api_logger

from prompt import CONTEXT_PROMPT



class PredictRouter:
    def __init__(self):
        self.llm_client = RAGClient()
        self.retriever = RetrieverService()
        self.mongo_utils = MongoUtils()

    async def predict_flow(self, query: str, session_id: str, user_id: str):
        """
        Main prediction function that orchestrates the RAG pipeline.
        """
        start_time = time.time()
        
        try:
            api_logger.info(f"ROUTER: Starting prediction flow | User: {user_id} | Session: {session_id} | Query length: {len(query)}")
            api_logger.debug(f"ROUTER: Query preview: {query[:100]}{'...' if len(query) > 100 else ''}")
            
            # Step 1: Retrieve similar chunks using RAG (includes query embedding)
            retrieval_start = time.time()
            api_logger.info(f"ROUTER: Starting RAG retrieval with query embedding | User: {user_id} | Session: {session_id} | Query length: {len(query)}")
            api_logger.debug(f"ROUTER: Query preview: {query[:100]}{'...' if len(query) > 100 else ''}")
            
            context_chunks = self.retriever.retrieve_similar_chunks(
                query=query, 
                k=5  # Get top 5 most relevant chunks
            )
            
            retrieval_time = time.time() - retrieval_start
            api_logger.info(f"ROUTER: RAG retrieval completed in {retrieval_time:.3f}s | Query embedded and chunks found: {len(context_chunks)}")
            api_logger.debug(f"ROUTER: Retrieved chunks preview: {str(context_chunks)[:200]}...")
            api_logger.debug(f"ROUTER: RAG process included query embedding and vector similarity search")
            
            # Step 2: Get chat history for context
            history_start = time.time()
            api_logger.info(f"ROUTER: Retrieving chat history | User: {user_id} | Session: {session_id}")
            
            chat_history = self.mongo_utils.get_chat_history(session_id)
            
            history_time = time.time() - history_start
            api_logger.info(f"ROUTER: Chat history retrieved in {history_time:.3f}s | History length: {len(chat_history)}")
            
            # Step 3: Create comprehensive prompt
            prompt_start = time.time()
            api_logger.info(f"ROUTER: Creating system prompt | User: {user_id} | Session: {session_id}")
            
            system_prompt = CONTEXT_PROMPT.format(chat_history=chat_history, context=context_chunks, query=query)
            
            prompt_time = time.time() - prompt_start
            api_logger.info(f"ROUTER: System prompt created in {prompt_time:.3f}s | Prompt length: {len(system_prompt)}")
            api_logger.debug(f"ROUTER: System prompt preview: {system_prompt[:300]}...")
            
            # Step 4: Call LLM for response
            llm_start = time.time()
            api_logger.info(f"ROUTER: Calling LLM for response | User: {user_id} | Session: {session_id}")
            
            json_response = self.llm_client.chat(prompt=system_prompt)
            response = self.llm_client.get_response(json_response)["response"]
            input_tokens = self.llm_client.get_response(json_response)["input_tokens"]
            output_tokens = self.llm_client.get_response(json_response)["output_tokens"]
            model = self.llm_client.get_response(json_response)["model"]

            llm_time = time.time() - llm_start
            api_logger.info(f"ROUTER: LLM response received in {llm_time:.3f}s | Model: {model} | Input tokens: {input_tokens} | Output tokens: {output_tokens}")
            api_logger.debug(f"ROUTER: LLM response preview: {response[:200]}...")

            # Step 5: Save chat session
            save_start = time.time()
            api_logger.info(f"ROUTER: Saving chat session | User: {user_id} | Session: {session_id}")
            
            self.mongo_utils.save_chat_session(
                user_id=user_id,
                session_id=session_id,
                query=query,
                response=response)
            
            save_time = time.time() - save_start
            api_logger.info(f"ROUTER: Chat session saved in {save_time:.3f}s")
            
            # Step 6: Return response
            total_time = time.time() - start_time
            api_logger.info(f"ROUTER: Prediction flow completed successfully in {total_time:.3f}s | User: {user_id} | Session: {session_id}")
            
            return {
                "response": response,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "chunks_retrieved": len(context_chunks),
                "chat_history_length": len(chat_history),
                "timing_breakdown": {
                    "retrieval": f"{retrieval_time:.3f}s",
                    "history": f"{history_time:.3f}s",
                    "prompt": f"{prompt_time:.3f}s",
                    "llm": f"{llm_time:.3f}s",
                    "save": f"{save_time:.3f}s",
                    "total": f"{total_time:.3f}s"
                }
            }
        
        except Exception as e:
            total_time = time.time() - start_time
            api_logger.error(f"ROUTER: Prediction flow failed for user {user_id} in {total_time:.3f}s | Session: {session_id}", exc_info=True)
            raise RAGError(f"Error in predict_flow: {str(e)}")
            