from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import time
from services.embedding_client import EmbeddingClient
from utils.logger import api_logger
from qdrant_client.models import CollectionStatus


class RetrieverService:
    def __init__(self, collection_name: str = "rag_chunks"):
        self.embedding_client = EmbeddingClient()
        self.collection_name = collection_name

        # Qdrant in-memory setup
        self.qdrant = QdrantClient(path="./qdrant_data")

        if self.collection_name not in {c.name for c in self.qdrant.get_collections().collections}:
            # create it only once
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )

    def store_chunks(self, texts: List[str], metadatas: List[dict] = None):
        if metadatas is None:
            metadatas = []

        """Embed and store text chunks into Qdrant."""
        start_time = time.time()
        
        try:
            api_logger.info(f"SERVICE: Starting embedding and storage process | Chunks: {len(texts)} | Collection: {self.collection_name}")
            api_logger.debug(f"SERVICE: Text chunks preview: {str(texts[:2])[:200]}...")
            
            # Step 1: Generate embeddings
            embedding_start = time.time()
            api_logger.info(f"SERVICE: Generating embeddings for {len(texts)} chunks")
            
            embeddings = self.embedding_client.embed_texts(texts)
            
            embedding_time = time.time() - embedding_start
            api_logger.info(f"SERVICE: Embeddings generated successfully in {embedding_time:.3f}s | Chunks: {len(texts)} | Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
            api_logger.debug(f"SERVICE: First embedding preview: {str(embeddings[0][:5]) if embeddings else 'None'}...")

            # Step 2: Create points for vector storage
            point_creation_start = time.time()
            api_logger.info(f"SERVICE: Creating vector points for storage | Chunks: {len(texts)}")
            
            points = []
            
            for i, embedding in enumerate(embeddings):
                base_payload = {"text": texts[i]}
                extra= metadatas[i] if metadatas and i < len(metadatas) else {}
                payload= {**base_payload, **extra}
                point_id = str(uuid.uuid4())
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload,
                    )
                )
                if i < 2:  # Log first few point IDs for debugging
                    api_logger.debug(f"SERVICE: Created point {i+1} | ID: {point_id} | Text preview: {texts[i][:50]}...")
            
            point_creation_time = time.time() - point_creation_start
            api_logger.info(f"SERVICE: Vector points created in {point_creation_time:.3f}s | Points: {len(points)}")

            # Step 3: Store in Qdrant
            storage_start = time.time()
            api_logger.info(f"SERVICE: Storing {len(points)} points in Qdrant | Collection: {self.collection_name}")
            
            self.qdrant.upsert(collection_name=self.collection_name, points=points)
            info = self.qdrant.get_collection(self.collection_name)

            
            storage_time = time.time() - storage_start
            total_time = time.time() - start_time
            
            api_logger.info(f"SERVICE: Vector storage completed successfully in {storage_time:.3f}s | Total time: {total_time:.3f}s")
            api_logger.info(f"SERVICE: Embedding and storage summary | Chunks processed: {len(texts)} | Points stored: {len(points)} | Collection: {self.collection_name}")
            
            # Log timing breakdown
            api_logger.debug(f"SERVICE: Timing breakdown - Embedding: {embedding_time:.3f}s | Point creation: {point_creation_time:.3f}s | Storage: {storage_time:.3f}s | Total: {total_time:.3f}s")
            
        except Exception as e:
            total_time = time.time() - start_time
            api_logger.error(f"SERVICE: Embedding and storage failed in {total_time:.3f}s | Chunks: {len(texts)} | Collection: {self.collection_name}", exc_info=True)
            raise

    def retrieve_similar_chunks(self, query: str, k: int = 10) -> List[str]:
        """Embed query and search top-k similar chunks."""
        start_time = time.time()
        
        try:
            api_logger.info(f"SERVICE: Starting similarity search | Query length: {len(query)} | Top-k: {k}")
            api_logger.debug(f"SERVICE: Query preview: {query[:100]}{'...' if len(query) > 100 else ''}")
            
            # Step 1: Embed the query
            query_embedding_start = time.time()
            api_logger.info(f"SERVICE: Generating query embedding")
            
            query_vector = self.embedding_client.embed_text(query)
            
            query_embedding_time = time.time() - query_embedding_start
            api_logger.info(f"SERVICE: Query embedding generated in {query_embedding_time:.3f}s | Dimension: {len(query_vector)}")

            # Step 2: Search similar chunks
            search_start = time.time()
            api_logger.info(f"SERVICE: Searching for similar chunks | Collection: {self.collection_name}")
            
            search_results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=k,
            )
            
            # show collection stats
            info = self.qdrant.get_collection(self.collection_name)
            # retrieve ANY point to confirm presence
            any_points = self.qdrant.scroll(self.collection_name, limit=3)
            print("Sample points:", any_points)
            print(f"**************************** SEARCH RESULTS ARE : {search_results}")

            
            search_time = time.time() - search_start
            total_time = time.time() - start_time
            
            # Extract text from results
            result_texts = [hit.payload.get("text", "") for hit in search_results]

            print(f"*********************************RESULT TEXTS ARE: {result_texts}")
            
            api_logger.info(f"SERVICE: Similarity search completed in {search_time:.3f}s | Total time: {total_time:.3f}s | Results: {len(result_texts)}")
            api_logger.debug(f"SERVICE: Search results preview: {str(result_texts[:2])[:200]}...")
            
            # Log timing breakdown
            api_logger.debug(f"SERVICE: Search timing breakdown - Query embedding: {query_embedding_time:.3f}s | Vector search: {search_time:.3f}s | Total: {total_time:.3f}s")
            
            return result_texts
            
        except Exception as e:
            total_time = time.time() - start_time
            api_logger.error(f"SERVICE: Similarity search failed in {total_time:.3f}s | Query length: {len(query)} | Top-k: {k}", exc_info=True)
            raise
