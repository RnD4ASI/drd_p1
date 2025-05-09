import os
import logging
import numpy as np
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch
from src.azure_integration import AzureOpenAIClient

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Handles text embedding using various models."""
    
    def __init__(self, model_path: Optional[str] = None, provider: str = "huggingface"):
        """Initialize the EmbeddingModel.
        
        Parameters:
            model_path (Optional[str]): Path to the embedding model
            provider (str): Model provider, either 'huggingface' or 'azure_openai'
        """
        self.model_path = model_path
        self.provider = provider.lower()
        self.model = None
        self.tokenizer = None
        self.azure_client = None
        
        # Set device for local models
        if self.provider == "huggingface":
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
        
        # Initialize Azure client if using Azure OpenAI
        if self.provider == "azure_openai":
            self.azure_client = AzureOpenAIClient()
            if not self.azure_client.is_configured:
                logger.warning("Azure OpenAI not fully configured. Will fall back to HuggingFace if Azure embedding is requested.")
    
    def load_model(self, model_path: Optional[str] = None):
        """Load the embedding model.
        
        Parameters:
            model_path (Optional[str]): Path to the embedding model
        """
        try:
            if model_path:
                self.model_path = model_path
            
            if not self.model_path:
                raise ValueError("Model path not provided")
            
            # If using Azure OpenAI, no need to load a local model
            if self.provider == "azure_openai":
                logger.info(f"Using Azure OpenAI embedding model: {self.model_path}")
                return
            
            # Load local model for HuggingFace
            logger.info(f"Loading embedding model from {self.model_path}")
            
            # Check if the model is a SentenceTransformer model
            try:
                self.model = SentenceTransformer(self.model_path)
                logger.info("Loaded SentenceTransformer model")
                return
            except Exception as e:
                logger.warning(f"Failed to load as SentenceTransformer: {e}")
            
            # Try loading as a Hugging Face model
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
                logger.info("Loaded Hugging Face model")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Error in load_model: {e}")
            raise
    
    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Get embeddings for the given texts.
        
        Parameters:
            texts (Union[str, List[str]]): Text(s) to embed
            
        Returns:
            np.ndarray: Embedding vectors
        """
        try:
            # Use Azure OpenAI for embeddings if specified
            if self.provider == "azure_openai":
                if not self.azure_client or not self.azure_client.is_configured:
                    raise ValueError("Azure OpenAI client not configured. Check environment variables.")
                
                # Get embeddings from Azure OpenAI
                try:
                    embeddings = self.azure_client.get_embeddings(texts, self.model_path)
                    # Convert to numpy array if it's a list of lists
                    if isinstance(embeddings, list) and isinstance(embeddings[0], list):
                        return np.array(embeddings)
                    # If it's a single embedding, wrap it in an array
                    elif isinstance(embeddings, list) and not isinstance(texts, list):
                        return np.array([embeddings])
                    return np.array(embeddings)
                except Exception as e:
                    logger.error(f"Azure OpenAI embeddings failed: {e}")
                    raise
            
            # Use local models (HuggingFace or SentenceTransformer)
            else:
                if self.model is None:
                    raise ValueError("Model not loaded. Call load_model first.")
                
                # Convert single text to list
                if isinstance(texts, str):
                    texts = [texts]
                
                # Use SentenceTransformer if available
                if isinstance(self.model, SentenceTransformer):
                    embeddings = self.model.encode(texts)
                    return embeddings
                
                # Use Hugging Face model
                else:
                    embeddings = []
                    for text in texts:
                        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                        
                        # Use mean pooling for sentence embedding
                        attention_mask = inputs["attention_mask"]
                        token_embeddings = outputs.last_hidden_state
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        embedding = (sum_embeddings / sum_mask).squeeze().cpu().numpy()
                        embeddings.append(embedding)
                    
                    return np.array(embeddings)
                
        except Exception as e:
            logger.error(f"Error in get_embeddings: {e}")
            raise

def concatenate_attribute_data(attribute_name: str, attribute_definition: str) -> str:
    """Concatenate attribute name and definition for embedding.
    
    Parameters:
        attribute_name (str): Name of the attribute
        attribute_definition (str): Definition of the attribute
        
    Returns:
        str: Concatenated text
    """
    return f"{attribute_name}: {attribute_definition}"

def embed_attributes(attributes_df, model_path: str, provider: str = "huggingface") -> np.ndarray:
    """Embed data attributes using the specified model.
    
    Parameters:
        attributes_df: DataFrame containing attribute_name and attribute_definition columns
        model_path (str): Path to the embedding model or name of Azure OpenAI model
        provider (str): Model provider, either 'huggingface' or 'azure_openai'
        
    Returns:
        np.ndarray: Array of embedding vectors
    """
    try:
        # Concatenate attribute name and definition
        texts = [concatenate_attribute_data(row['attribute_name'], row['attribute_definition']) 
                for _, row in attributes_df.iterrows()]
        
        # Initialize and load the embedding model
        embedding_model = EmbeddingModel(model_path, provider=provider)
        embedding_model.load_model()
        
        # Get embeddings
        embeddings = embedding_model.get_embeddings(texts)
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Error in embed_attributes: {e}")
        raise
