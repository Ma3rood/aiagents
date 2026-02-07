from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from app.core.config import settings
from app.core.logging_config import get_logger
import hashlib

logger = get_logger(__name__)


class QDrantService:
    """Service for interacting with QDrant vector database for category embeddings"""
    
    # Embedding dimension for qwen3-embedding-8b (4096 dimensions)
    EMBEDDING_DIMENSION = 4096
    
    def __init__(self):
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        
        # Initialize QDrant client
        if settings.QDRANT_API_KEY:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY
            )
        else:
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT
            )
        
        logger.info(f"QDrantService initialized - Host: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    
    def _generate_point_id(self, id_path: str) -> str:
        """Generate a unique point ID from the category id_path using hash"""
        # Use MD5 hash and convert to UUID format for QDrant compatibility
        hash_hex = hashlib.md5(id_path.encode()).hexdigest()
        return hash_hex
    
    def ensure_collection_exists(self) -> bool:
        """
        Ensure the collection exists, create if it doesn't.
        
        Returns:
            True if collection was created, False if it already existed
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.EMBEDDING_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
                
                # Create payload index for root_category for efficient filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="root_category",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                
                logger.info(f"Collection {self.collection_name} created with root_category index")
                return True
            else:
                logger.debug(f"Collection {self.collection_name} already exists")
                return False
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}", exc_info=True)
            raise
    
    def check_category_exists(self, id_path: str) -> bool:
        """
        Check if a category already exists in the collection.
        
        Args:
            id_path: The category ID path (e.g., "1 > 2 > 3")
            
        Returns:
            True if category exists, False otherwise
        """
        try:
            point_id = self._generate_point_id(id_path)
            
            # Try to retrieve the point
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=False,
                with_vectors=False
            )
            
            return len(points) > 0
            
        except Exception as e:
            logger.error(f"Error checking category existence: {str(e)}", exc_info=True)
            return False
    
    def upsert_category_embedding(
        self,
        id_path: str,
        category_path: str,
        root_category: str,
        semantic_description: str,
        relevant_attributes: List[str],
        embedding_vector: List[float]
    ) -> bool:
        """
        Insert or update a category embedding in the collection.
        
        Args:
            id_path: The category ID path (e.g., "1 > 2 > 3")
            category_path: The category name path (e.g., "Electronics > Phones > Smartphones")
            root_category: The root category name for filtering
            semantic_description: The semantic description of the category
            relevant_attributes: List of relevant attributes for this category
            embedding_vector: The embedding vector for the semantic description
            
        Returns:
            True if successful, False otherwise
        """
        try:
            point_id = self._generate_point_id(id_path)
            
            point = PointStruct(
                id=point_id,
                vector=embedding_vector,
                payload={
                    "id_path": id_path,
                    "category_path": category_path,
                    "root_category": root_category,
                    "semantic_description": semantic_description,
                    "relevant_attributes": relevant_attributes
                }
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.debug(f"Upserted category: {category_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error upserting category embedding: {str(e)}", exc_info=True)
            return False
    
    def upsert_category_embeddings_batch(
        self,
        categories: List[Dict[str, Any]]
    ) -> int:
        """
        Batch insert or update multiple category embeddings.
        
        Args:
            categories: List of category dicts with keys:
                - id_path, category_path, root_category, semantic_description, 
                - relevant_attributes, embedding_vector
                
        Returns:
            Number of successfully upserted categories
        """
        try:
            points = []
            for cat in categories:
                point_id = self._generate_point_id(cat["id_path"])
                points.append(PointStruct(
                    id=point_id,
                    vector=cat["embedding_vector"],
                    payload={
                        "id_path": cat["id_path"],
                        "category_path": cat["category_path"],
                        "root_category": cat["root_category"],
                        "semantic_description": cat["semantic_description"],
                        "relevant_attributes": cat["relevant_attributes"]
                    }
                ))
            
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"Batch upserted {len(points)} categories")
            
            return len(points)
            
        except Exception as e:
            logger.error(f"Error batch upserting category embeddings: {str(e)}", exc_info=True)
            return 0
    
    def search_categories(
        self,
        query_embedding: List[float],
        root_category_filter: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar categories using semantic search.
        
        Args:
            query_embedding: The embedding vector of the query
            root_category_filter: Optional root category to filter by
            top_k: Number of top results to return
            
        Returns:
            List of category dicts with similarity scores
        """
        try:
            # Build filter if root_category is provided
            search_filter = None
            if root_category_filter:
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="root_category",
                            match=MatchValue(value=root_category_filter)
                        )
                    ]
                )
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=top_k,
                with_payload=True
            )
            
            categories = []
            for result in results:
                categories.append({
                    "id_path": result.payload.get("id_path"),
                    "category_path": result.payload.get("category_path"),
                    "root_category": result.payload.get("root_category"),
                    "relevant_attributes": result.payload.get("relevant_attributes", []),
                    "semantic_description": result.payload.get("semantic_description", ""),
                    "score": result.score
                })
            
            logger.debug(f"Found {len(categories)} categories for query (filter: {root_category_filter})")
            return categories
            
        except Exception as e:
            logger.error(f"Error searching categories: {str(e)}", exc_info=True)
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value if info.status else "unknown"
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def delete_collection(self) -> bool:
        """Delete the entire collection (use with caution)"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.warning(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}", exc_info=True)
            return False


# Singleton instance
_qdrant_service: Optional[QDrantService] = None


def get_qdrant_service() -> QDrantService:
    """Get or create the QDrant service singleton"""
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QDrantService()
    return _qdrant_service
