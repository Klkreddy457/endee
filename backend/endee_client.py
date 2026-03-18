import requests
from typing import List, Dict, Any, Optional

from backend.config import settings
from backend.utils import logger

class EndeeClient:
    def __init__(self):
        self.base_url = settings.endee_url.rstrip("/")
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if settings.endee_auth_token:
            self.headers["Authorization"] = settings.endee_auth_token

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        try:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                return response.json()
            elif "application/msgpack" in content_type:
                import msgpack
                # Unpack MessagePack into a Python dict/list
                unpacked = msgpack.unpackb(response.content, raw=False)
                # Ensure it's returned as a dictionary (e.g., wrap it if it's a list)
                if isinstance(unpacked, list):
                    return {"results": unpacked}
                return unpacked
            else:
                return {"message": response.text}
        except requests.exceptions.HTTPError as e:
            logger.error(f"Endee API Error ({response.status_code}): {response.text}")
            raise Exception(f"Endee DB error: {response.text}") from e

    def health_check(self) -> bool:
        """Check if Endee server is running."""
        try:
            res = requests.get(f"{self.base_url}/api/v1/health", headers=self.headers, timeout=5)
            # Response in open-source server for health check doesn't require auth but let's pass it anyway
            res.raise_for_status()
            return res.json().get("status") == "ok"
        except Exception as e:
            logger.error(f"Endee health check failed: {e}")
            return False

    def list_indexes(self) -> Dict[str, Any]:
        """List all available indexes."""
        res = requests.get(f"{self.base_url}/api/v1/index/list", headers=self.headers)
        return self._handle_response(res)

    def create_index(self, index_name: str, dim: int = settings.vector_dimension, 
                     space_type: str = "cosine", precision: str = "float32") -> Dict[str, Any]:
        """Create a new index if it doesn't already exist."""
        # First check if it exists
        try:
            indexes = self.list_indexes()
            for idx in indexes.get("indexes", []):
                if idx.get("name") == index_name:
                    logger.info(f"Index {index_name} already exists.")
                    return {"message": "Index already exists"}
        except Exception as e:
            logger.warning(f"Failed to check existing indexes, attempting to create anyway: {e}")

        payload = {
            "index_name": index_name,
            "dim": dim,
            "space_type": space_type,
            "precision": precision
        }
        res = requests.post(f"{self.base_url}/api/v1/index/create", json=payload, headers=self.headers)
        
        # 409 means it already exists, which is fine
        if res.status_code == 409:
            return {"message": "Index already exists"}
            
        return self._handle_response(res)

    def insert_vectors(self, index_name: str, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Insert vectors into an index.
        Format of vectors:
        [
            {
                "id": "doc_1_chunk_1",
                "meta": "{\"text\": \"doc content...\", \"source\": \"file.txt\"}",
                "vector": [0.1, 0.2, ...]
            }
        ]
        """
        if not vectors:
            return {"message": "No vectors to insert"}
            
        res = requests.post(f"{self.base_url}/api/v1/index/{index_name}/vector/insert", 
                            json=vectors, headers=self.headers)
        return self._handle_response(res)

    def search(self, index_name: str, vector: List[float], k: int = settings.top_k) -> Dict[str, Any]:
        """
        Search for top_k similar vectors.
        """
        payload = {
            "vector": vector,
            "k": k,
            "include_vectors": False
        }
        res = requests.post(f"{self.base_url}/api/v1/index/{index_name}/search", 
                            json=payload, headers=self.headers)
        return self._handle_response(res)

endee_client = EndeeClient()
