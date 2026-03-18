import json
import uuid
from typing import List, Dict, Any, Optional

from backend.config import settings
from backend.utils import logger
from backend.endee_client import endee_client
from backend.embedding import embedding_service
from backend.document_loader import document_loader

class RAGPipeline:
    def __init__(self):
        self.index_name = settings.index_name
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        """Make sure the Endee index exists upon initialization."""
        try:
            logger.info(f"Ensuring index '{self.index_name}' exists...")
            endee_client.create_index(
                index_name=self.index_name,
                dim=settings.vector_dimension,
                space_type="cosine"
            )
        except Exception as e:
            logger.error(f"Failed to ensure index exists: {e}")
            # Non-blocking, as Endee might not be up yet
            
    def ingest_document(self, text: str, source_name: str) -> Dict[str, Any]:
        """
        Process a document: chunk -> embed -> insert to Endee.
        """
        # Ensure index exists before ingesting in case Endee just started
        self._ensure_index_exists()

        logger.info(f"Starting ingestion for '{source_name}'")
        
        # 1. Chunk document
        chunks = document_loader.process_text(text, source_name)
        if not chunks:
            return {"status": "error", "message": "No valid text found to process"}
            
        logger.info(f"Document split into {len(chunks)} chunks. Generating embeddings...")
        
        try:
            # 2. Extract texts and generate embeddings
            texts = [c["text"] for c in chunks]
            embeddings = embedding_service.get_embeddings_batch(texts)
            
            # 3. Format vectors for Endee insertion
            vectors = []
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                # We store both the text and metadata in the 'meta' field as a JSON string
                meta_dict = {
                    "text": chunk["text"],
                    "source": chunk["metadata"]["source"],
                    "chunk_id": chunk["metadata"]["chunk_id"]
                }
                
                # Generate unique ID for the vector
                doc_id = str(uuid.uuid4())
                
                vectors.append({
                    "id": doc_id,
                    "meta": json.dumps(meta_dict),
                    "vector": emb
                })
            
            # 4. Insert to Endee
            logger.info(f"Inserting {len(vectors)} vectors into Endee...")
            res = endee_client.insert_vectors(self.index_name, vectors)
            
            return {
                "status": "success", 
                "message": f"Successfully ingested {len(chunks)} chunks.",
                "endee_response": res
            }
            
        except Exception as e:
            logger.error(f"Error during ingestion pipeline: {e}")
            raise

    def query(self, user_query: str, history: list = None) -> Dict[str, Any]:
        """
        Answer a user query using RAG.
        Embed -> fetch chunks -> send to LLM (with optional conversation history).
        """
        logger.info(f"Processing query: '{user_query}'")
        
        try:
            # 1. Embed query
            query_embedding = embedding_service.get_embedding(user_query)
            
            # 2. Search Endee
            search_results = endee_client.search(
                index_name=self.index_name,
                vector=query_embedding,
                k=settings.top_k
            )
            
            # Extract retrieved documents
            retrieved_chunks = []
            
            # Depending on how the API responds, it might have a "results" array
            results = search_results.get("results", [])
            for res in results:
                try:
                    # Endee's msgpack search endpoint returns results as lists: [score, id, meta, filter, norm, [vector]]
                    if isinstance(res, list) and len(res) > 2:
                        score = res[0]
                        meta_str = res[2]
                    # Fallback for if it ever returns dictionaries
                    elif isinstance(res, dict) and "meta" in res:
                        score = res.get("score", 0.0)
                        meta_str = res["meta"]
                    else:
                        continue
                        
                    meta_data = json.loads(meta_str)
                    retrieved_chunks.append({
                        "text": meta_data.get("text", ""),
                        "source": meta_data.get("source", "Unknown"),
                        "score": score
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse metadata from Endee result: {e}")
            
            if not retrieved_chunks:
                logger.warning("No relevant chunks found for query.")
                return {
                    "answer": "I don't have enough context in my knowledge base to answer this question.",
                    "sources": [],
                    "chunks": []
                }
                
            # 3. Construct prompt
            context_text = "\n\n---\n\n".join([
                f"Source: {c['source']}\nContent: {c['text']}" for c in retrieved_chunks
            ])
            
            system_prompt = f"""You are an advanced AI assistant. Use the following retrieved context to answer the user's question accurately.
If you don't know the answer based on the context, say so. Do not hallucinate external information.

Context:
{context_text}"""

            # 4. Generate response (pass history for conversation memory)
            logger.info("Generating LLM answer...")
            structured_answer = embedding_service.generate_response(
                system_prompt, user_query, history=history or []
            )
            
            # Unique sources
            sources = list(set([c["source"] for c in retrieved_chunks]))
            
            return {
                "answer": structured_answer,
                "sources": sources,
                "chunks": retrieved_chunks
            }
            
        except Exception as e:
            logger.error(f"Error during query pipeline: {e}")
            raise

rag_pipeline = RAGPipeline()
