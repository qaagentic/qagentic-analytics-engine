"""Service for creating and managing embeddings for failures."""

import logging
from typing import List, Optional, Union, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for creating and managing embeddings for failures."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with the specified embedding model."""
        logger.info(f"Initializing embedding service with model {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text.
        
        Args:
            text: The text to embed
            
        Returns:
            A list of floats representing the embedding vector
        """
        # Convert to numpy array then to list to ensure serialization works
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            A list of embedding vectors
        """
        embeddings = self.model.encode(texts)
        return [embedding.tolist() for embedding in embeddings]
    
    async def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(similarity)
        
    async def get_failure_embedding(
        self, 
        error_message: str, 
        error_type: Optional[str] = None,
        stack_trace: Optional[str] = None,
        test_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """
        Generate an embedding for a failure, combining various information sources.
        
        Args:
            error_message: The error message text
            error_type: Optional exception type
            stack_trace: Optional stack trace
            test_name: Optional test name
            context: Optional additional context about the failure
            
        Returns:
            An embedding vector representing the failure
        """
        # Combine the information into a single string, emphasizing the error message
        combined_text = error_message
        
        if error_type:
            combined_text = f"{error_type}: {combined_text}"
            
        if test_name:
            combined_text = f"{combined_text}\nTest: {test_name}"
            
        if stack_trace:
            # Include just the first few lines of the stack trace, which often contain
            # the most relevant information
            stack_lines = stack_trace.split('\n')[:5]
            stack_summary = '\n'.join(stack_lines)
            combined_text = f"{combined_text}\nTrace: {stack_summary}"
            
        if context:
            context_str = ' '.join([f"{k}:{v}" for k, v in context.items()])
            combined_text = f"{combined_text}\nContext: {context_str}"
            
        return await self.get_embedding(combined_text)
