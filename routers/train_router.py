import os
import uuid
from typing import Dict, Any, List
from datetime import datetime
import hashlib
import time
from fastapi import UploadFile
from components.chunking import Chunker
from components.metadata_handling import MetadataHandler
from services.retriever import RetrieverService
from db.mongo_utils import MongoUtils
from utils.logger import api_logger

class TrainRouter:
    def __init__(self):
        self.chunker = Chunker()
        self.metadata_handler = MetadataHandler()
        self.retriever = RetrieverService()
        self.mongo_utils = MongoUtils()

    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file content"""
        try:
            import PyPDF2
            import io
            
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            api_logger.info(f"ROUTER: PDF text extraction completed | Pages: {len(pdf_reader.pages)}")
            return text.strip()
            
        except ImportError:
            api_logger.error("ROUTER: PyPDF2 not installed. Please install it with: pip install PyPDF2")
            raise Exception("PDF processing requires PyPDF2. Please install it with: pip install PyPDF2")
        except Exception as e:
            api_logger.error(f"ROUTER: PDF text extraction failed: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")

    def extract_text_from_file(self, file_content: bytes, filename: str) -> str:
        """Extract text from file based on file type"""
        file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        if file_extension == 'pdf':
            api_logger.info(f"ROUTER: Processing PDF file: {filename}")
            return self.extract_text_from_pdf(file_content)
        elif file_extension in ['txt', 'md', 'json', 'csv']:
            api_logger.info(f"ROUTER: Processing text file: {filename}")
            try:
                return file_content.decode('utf-8')
            except UnicodeDecodeError:
                # Try with different encoding
                try:
                    return file_content.decode('latin-1')
                except:
                    raise Exception(f"Failed to decode text file {filename}")
        else:
            api_logger.warning(f"ROUTER: Unknown file type: {file_extension}. Attempting UTF-8 decode")
            try:
                return file_content.decode('utf-8')
            except UnicodeDecodeError:
                raise Exception(f"Unsupported file type or encoding: {filename}")

    async def train(self, user_id: str, file: UploadFile):
        """
        Main training function that processes documents and stores them.
        """
        start_time = time.time()
        
        try:
            api_logger.info(f"ROUTER: Starting training process for user: {user_id} | File: {file.filename}")
            
            # Step 1: Read file content
            file_read_start = time.time()
            file_content = await file.read()
            filename = f"{user_id}_{file.filename}"
            file_read_time = time.time() - file_read_start
            
            api_logger.info(f"ROUTER: File read completed in {file_read_time:.3f}s | Size: {len(file_content)} bytes")
            
            # Step 2: Extract text from file
            text_extraction_start = time.time()
            api_logger.info(f"ROUTER: Extracting text from file | User: {user_id} | File: {filename}")
            
            file_text = self.extract_text_from_file(file_content, file.filename)
            
            text_extraction_time = time.time() - text_extraction_start
            api_logger.info(f"ROUTER: Text extraction completed in {text_extraction_time:.3f}s | Text length: {len(file_text)} chars")
            api_logger.debug(f"ROUTER: Generated filename: {filename}")
            api_logger.debug(f"ROUTER: Text preview: {file_text[:200]}{'...' if len(file_text) > 200 else ''}")

            # Step 3: Save file content to MongoDB
            mongo_start = time.time()
            api_logger.info(f"ROUTER: Saving document to MongoDB | User: {user_id} | File: {filename}")
            
            self.mongo_utils.save_uploaded_document(
                user_id=user_id,
                filename=filename,
                file_content=file_text
            )
            
            mongo_time = time.time() - mongo_start
            api_logger.info(f"ROUTER: Document saved to MongoDB in {mongo_time:.3f}s")
            
            # Step 4: Chunk the document
            chunk_start = time.time()
            api_logger.info(f"ROUTER: Starting document chunking | User: {user_id} | File: {filename}")
            
            text_chunks = self.chunker.chunk_text(file_text)
            
            chunk_time = time.time() - chunk_start
            api_logger.info(f"ROUTER: Document chunking completed in {chunk_time:.3f}s | Chunks: {len(text_chunks)}")
            
            if not text_chunks:
                api_logger.error(f"ROUTER: No chunks generated from document | User: {user_id} | File: {filename}")
                return {
                    "error": "Processing Error",
                    "message": "No chunks generated from document",
                    "user_id": user_id,
                    "filename": file.filename,
                    "status": "failed"
                }
            
            # Step 5: Create metadata for each chunk
            metadata_start = time.time()
            api_logger.info(f"ROUTER: Creating metadata for {len(text_chunks)} chunks | User: {user_id}")
            
            chunk_metadata = self.metadata_handler.get_metadata_list(
                texts=text_chunks,
                source=filename,
                user_id=user_id
            )
            
            metadata_time = time.time() - metadata_start
            api_logger.info(f"ROUTER: Metadata creation completed in {metadata_time:.3f}s | Metadata count: {len(chunk_metadata)}")
            
            # Step 6: Store chunks in vector database (includes embedding)
            vector_start = time.time()
            api_logger.info(f"ROUTER: Starting embedding and vector storage | User: {user_id} | Chunks: {len(text_chunks)}")
            api_logger.debug(f"ROUTER: First chunk preview: {text_chunks[0][:100] if text_chunks else 'None'}{'...' if text_chunks and len(text_chunks[0]) > 100 else ''}")
            
            self.retriever.store_chunks(
                texts=text_chunks,
                metadatas=chunk_metadata
            )
            
            vector_time = time.time() - vector_start
            api_logger.info(f"ROUTER: Embedding and vector storage completed in {vector_time:.3f}s | Chunks processed: {len(text_chunks)}")
            api_logger.debug(f"ROUTER: Embedding process included text-to-vector conversion and Qdrant storage")
            
            # Step 7: Prepare response
            total_time = time.time() - start_time
            api_logger.info(f"ROUTER: Training process completed successfully in {total_time:.3f}s | User: {user_id} | File: {filename}")
            
            result = {
                "user_id": user_id,
                "filename": file.filename,
                "processing_summary": {
                    "total_chunks": len(text_chunks),
                    "file_size": len(file_content),
                    "text_length": len(file_text),
                    "processing_time": datetime.utcnow().isoformat(),
                    "timing_breakdown": {
                        "file_read": f"{file_read_time:.3f}s",
                        "text_extraction": f"{text_extraction_time:.3f}s",
                        "mongo_save": f"{mongo_time:.3f}s",
                        "chunking": f"{chunk_time:.3f}s",
                        "metadata": f"{metadata_time:.3f}s",
                        "vector_storage": f"{vector_time:.3f}s",
                        "total": f"{total_time:.3f}s"
                    }
                },
                "storage_info": {
                    "vector_db": "Qdrant",
                    "document_db": "MongoDB",
                    "chunks_stored": len(text_chunks),
                    "file_stored": True
                },
                "status": "success",
                "message": f"Document '{file.filename}' processed and stored successfully"
            }
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            api_logger.error(f"ROUTER: Training process failed for user {user_id} in {total_time:.3f}s | File: {file.filename if hasattr(file, 'filename') else 'unknown'}", exc_info=True)
            
            return {
                "error": "Processing Error",
                "message": str(e),
                "user_id": user_id,
                "filename": file.filename if hasattr(file, 'filename') else "unknown",
                "status": "failed"
            }

    def generate_document_id(self, user_id: str, filename: str) -> str:
        """
        Generate a unique document ID based on the filename.
        """
        updated_filename = f"{user_id}_{filename}"
        return hashlib.sha256(updated_filename.encode()).hexdigest()
