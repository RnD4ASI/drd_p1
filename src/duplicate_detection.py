import os
import json
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
import uuid
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from src.azure_integration import AzureOpenAIClient

logger = logging.getLogger(__name__)

class DuplicateDetector:
    """Detects duplicate data attributes using language models."""
    
    def __init__(self, model_path: str, provider: str = "huggingface"):
        """Initialize the DuplicateDetector.
        
        Parameters:
            model_path (str): Path to the language model or name of Azure OpenAI model
            provider (str): Model provider, either 'huggingface' or 'azure_openai'
        """
        self.model_path = model_path
        self.provider = provider.lower()
        self.model = None
        self.tokenizer = None
        self.azure_client = None
        
        # Initialize Azure client if using Azure OpenAI
        if self.provider == "azure_openai":
            self.azure_client = AzureOpenAIClient()
            if not self.azure_client.is_configured:
                logger.warning("Azure OpenAI not fully configured. Will fall back to HuggingFace if Azure completion is requested.")
    
    def load_model(self):
        """Load the language model."""
        try:
            # If using Azure OpenAI, no need to load a local model
            if self.provider == "azure_openai":
                logger.info(f"Using Azure OpenAI language model: {self.model_path}")
                return
            
            # Load local model for HuggingFace
            logger.info(f"Loading language model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Language model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading language model: {e}")
            raise
    
    def generate_response(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
        """Generate a response from the language model.
        
        Parameters:
            prompt (str): Input prompt
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            
        Returns:
            str: Generated response
        """
        try:
            # Use Azure OpenAI for completions if specified
            if self.provider == "azure_openai":
                if not self.azure_client or not self.azure_client.is_configured:
                    raise ValueError("Azure OpenAI client not configured. Check environment variables.")
                
                # Get completion from Azure OpenAI
                try:
                    system_prompt = "You are an AI assistant that helps identify duplicate data attributes and selects the best attributes from duplicates."
                    response = self.azure_client.get_completion(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=self.model_path,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return response
                except Exception as e:
                    logger.error(f"Azure OpenAI completion failed: {e}")
                    raise
            
            # Use local HuggingFace model
            else:
                if self.model is None or self.tokenizer is None:
                    raise ValueError("Model not loaded. Call load_model first.")
                
                # Tokenize the prompt
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                # Generate response
                generation_config = GenerationConfig(
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=50,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
                
                # Decode the response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract the generated part (remove the prompt)
                prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
                response_tokens = self.tokenizer.decode(outputs[0][prompt_tokens:], skip_special_tokens=True)
                
                return response_tokens.strip()
            
        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            raise
    
    def detect_duplicates_in_cluster(self, cluster_attributes: pd.DataFrame) -> Dict[str, Any]:
        """Detect duplicates within a cluster.
        
        Parameters:
            cluster_attributes (pd.DataFrame): DataFrame containing attributes in a cluster
            
        Returns:
            Dict[str, Any]: Dictionary containing duplicate detection results
        """
        try:
            if len(cluster_attributes) <= 1:
                # No duplicates if only one attribute in the cluster
                return {
                    "attributes": cluster_attributes['attribute_name'].tolist(),
                    "duplicates": [],
                    "duplicate_groups": {}
                }
            
            # Prepare the prompt
            attributes_text = "\n".join([
                f"{i+1}. Name: {row['attribute_name']}, Definition: {row['attribute_definition']}"
                for i, (_, row) in enumerate(cluster_attributes.iterrows())
            ])
            
            prompt = f"""
            I have a set of data attributes that may contain duplicates. Please analyze the following attributes and identify which ones are duplicates of each other.

            {attributes_text}

            For each attribute, determine if it is a duplicate of any other attribute in the list. Two attributes are considered duplicates if they represent the same concept or data, even if they have different names or slightly different definitions.

            Return your analysis as a JSON object with the following structure:
            {{
                "duplicates": [
                    {{
                        "attribute_name": "name of the attribute",
                        "is_duplicate": true/false,
                        "duplicate_group_id": "group identifier for duplicates" (only if is_duplicate is true)
                    }},
                    ...
                ]
            }}

            Assign the same duplicate_group_id to all attributes that are duplicates of each other. Use a simple identifier like "group1", "group2", etc. If an attribute is not a duplicate, do not include a duplicate_group_id for it.
            """
            
            # Generate response
            response = self.generate_response(prompt)
            
            # Parse the JSON response
            try:
                # Extract JSON from the response (it might be surrounded by markdown code blocks)
                json_str = response
                
                # Try different patterns to extract JSON
                if "```json" in response:
                    parts = response.split("```json")
                    if len(parts) > 1 and "```" in parts[1]:
                        json_str = parts[1].split("```")[0].strip()
                elif "```" in response:
                    parts = response.split("```")
                    if len(parts) > 1:
                        json_str = parts[1].strip()
                elif "{" in response and "}" in response:
                    # Try to extract JSON object directly
                    start = response.find("{")
                    end = response.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        json_str = response[start:end+1]
                
                # Try to parse the JSON
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError:
                    # If parsing fails, try to create a structured result manually
                    logger.warning("Failed to parse JSON response, creating structured result manually")
                    # Create a default structure with no duplicates
                    result = {
                        "duplicates": [
                            {"attribute_name": name, "is_duplicate": False}
                            for name in cluster_attributes['attribute_name']
                        ]
                    }
                
                # Extract duplicate groups
                duplicate_groups = {}
                for item in result.get("duplicates", []):
                    if item.get("is_duplicate") and "duplicate_group_id" in item:
                        group_id = item["duplicate_group_id"]
                        if group_id not in duplicate_groups:
                            duplicate_groups[group_id] = []
                        duplicate_groups[group_id].append(item["attribute_name"])
                
                # Create result dictionary
                return {
                    "attributes": cluster_attributes['attribute_name'].tolist(),
                    "duplicates": result.get("duplicates", []),
                    "duplicate_groups": duplicate_groups
                }
                
            except Exception as e:
                logger.error(f"Error parsing JSON response: {e}")
                logger.error(f"Response: {response}")
                # Create a fallback result with no duplicates
                return {
                    "attributes": cluster_attributes['attribute_name'].tolist(),
                    "duplicates": [
                        {"attribute_name": name, "is_duplicate": False}
                        for name in cluster_attributes['attribute_name']
                    ],
                    "duplicate_groups": {}
                }
                
        except Exception as e:
            logger.error(f"Error in detect_duplicates_in_cluster: {e}")
            raise
    
    def select_best_attribute(self, duplicate_group: List[str], attributes_df: pd.DataFrame) -> str:
        """Select the best attribute from a duplicate group.
        
        Parameters:
            duplicate_group (List[str]): List of attribute names in the duplicate group
            attributes_df (pd.DataFrame): DataFrame containing all attributes
            
        Returns:
            str: Name of the best attribute
        """
        try:
            # Get the attributes in the duplicate group
            group_attributes = attributes_df[attributes_df['attribute_name'].isin(duplicate_group)]
            
            if len(group_attributes) <= 1:
                return duplicate_group[0]
            
            # Prepare the prompt
            attributes_text = "\n".join([
                f"{i+1}. Name: {row['attribute_name']}, Definition: {row['attribute_definition']}"
                for i, (_, row) in enumerate(group_attributes.iterrows())
            ])
            
            prompt = f"""
            I have a set of data attributes that have been identified as duplicates of each other. Please analyze the following attributes and select the one with the best naming convention and definition.

            {attributes_text}

            A good attribute should have:
            1. A clear, descriptive name that follows standard naming conventions
            2. A comprehensive and precise definition
            3. Consistency with industry standards or common practices

            Return your analysis as a JSON object with the following structure:
            {{
                "best_attribute": "name of the best attribute",
                "reasoning": "explanation for why this attribute was selected"
            }}
            """
            
            # Generate response
            response = self.generate_response(prompt)
            
            # Parse the JSON response
            try:
                # Extract JSON from the response
                json_str = response
                
                # Try different patterns to extract JSON
                if "```json" in response:
                    parts = response.split("```json")
                    if len(parts) > 1 and "```" in parts[1]:
                        json_str = parts[1].split("```")[0].strip()
                elif "```" in response:
                    parts = response.split("```")
                    if len(parts) > 1:
                        json_str = parts[1].strip()
                elif "{" in response and "}" in response:
                    # Try to extract JSON object directly
                    start = response.find("{")
                    end = response.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        json_str = response[start:end+1]
                
                # Try to parse the JSON
                try:
                    result = json.loads(json_str)
                    return result.get("best_attribute", duplicate_group[0])
                except json.JSONDecodeError:
                    # If parsing fails, try to extract the best attribute from the text
                    logger.warning("Failed to parse JSON response, extracting best attribute from text")
                    
                    # Look for patterns like "best_attribute": "attribute_name" or similar
                    best_attribute_pattern = r'"best_attribute"\s*:\s*"([^"]+)"'
                    match = re.search(best_attribute_pattern, response)
                    if match:
                        best_attr = match.group(1)
                        # Verify it's in the duplicate group
                        if best_attr in duplicate_group:
                            return best_attr
                    
                    # Fall back to the first attribute in the group
                    return duplicate_group[0]
                
            except Exception as e:
                logger.error(f"Error parsing JSON response: {e}")
                logger.error(f"Response: {response}")
                # Fall back to the first attribute in the group
                return duplicate_group[0]
                
        except Exception as e:
            logger.error(f"Error in select_best_attribute: {e}")
            # Fall back to the first attribute in the group
            return duplicate_group[0]


def detect_duplicates(attributes_with_clusters: pd.DataFrame, model_path: str, 
                     output_dir: Union[str, Path] = None, provider: str = "huggingface") -> pd.DataFrame:
    """Detect duplicates in clustered attributes.
    
    Parameters:
        attributes_with_clusters (pd.DataFrame): DataFrame containing attributes with cluster assignments
        model_path (str): Path to the language model
        output_dir (Union[str, Path]): Directory to save outputs
        
    Returns:
        pd.DataFrame: DataFrame with duplicate detection results
    """
    try:
        # Initialize duplicate detector
        detector = DuplicateDetector(model_path, provider=provider)
        detector.load_model()
        
        # Create output directory if provided
        if output_dir:
            output_dir = Path(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize result DataFrame
        result_df = attributes_with_clusters.copy()
        result_df['is_duplicate'] = False
        result_df['duplicate_group_id'] = None
        result_df['keep'] = True
        
        # Process each cluster
        unique_clusters = attributes_with_clusters['cluster_id'].unique()
        all_duplicate_groups = {}
        
        for cluster_id in unique_clusters:
            logger.info(f"Processing cluster {cluster_id}")
            
            # Get attributes in the cluster
            cluster_attributes = attributes_with_clusters[attributes_with_clusters['cluster_id'] == cluster_id]
            
            # Skip clusters with only one attribute
            if len(cluster_attributes) <= 1:
                logger.info(f"Cluster {cluster_id} has only one attribute, skipping")
                continue
            
            # Detect duplicates in the cluster
            duplicate_results = detector.detect_duplicates_in_cluster(cluster_attributes)
            
            # Save results if output_dir is provided
            if output_dir:
                with open(output_dir / f"cluster_{cluster_id}_duplicates.json", 'w') as f:
                    json.dump(duplicate_results, f, indent=4)
            
            # Update result DataFrame
            for item in duplicate_results.get("duplicates", []):
                attribute_name = item.get("attribute_name")
                is_duplicate = item.get("is_duplicate", False)
                
                if is_duplicate and "duplicate_group_id" in item:
                    group_id = item["duplicate_group_id"]
                    # Create a unique group ID across all clusters
                    unique_group_id = f"cluster_{cluster_id}_{group_id}"
                    
                    # Update DataFrame
                    result_df.loc[result_df['attribute_name'] == attribute_name, 'is_duplicate'] = True
                    result_df.loc[result_df['attribute_name'] == attribute_name, 'duplicate_group_id'] = unique_group_id
                    
                    # Add to all_duplicate_groups
                    if unique_group_id not in all_duplicate_groups:
                        all_duplicate_groups[unique_group_id] = []
                    all_duplicate_groups[unique_group_id].append(attribute_name)
        
        # Select the best attribute from each duplicate group
        for group_id, attribute_names in all_duplicate_groups.items():
            logger.info(f"Selecting best attribute for group {group_id}")
            
            # Skip groups with only one attribute
            if len(attribute_names) <= 1:
                continue
            
            # Select the best attribute
            best_attribute = detector.select_best_attribute(attribute_names, attributes_with_clusters)
            
            # Update result DataFrame
            for attribute_name in attribute_names:
                if attribute_name != best_attribute:
                    result_df.loc[result_df['attribute_name'] == attribute_name, 'keep'] = False
            
            # Save results if output_dir is provided
            if output_dir:
                with open(output_dir / f"group_{group_id}_best_attribute.json", 'w') as f:
                    json.dump({
                        "group_id": group_id,
                        "attributes": attribute_names,
                        "best_attribute": best_attribute
                    }, f, indent=4)
        
        # Save final results if output_dir is provided
        if output_dir:
            result_df.to_csv(output_dir / "duplicate_detection_results.csv", index=False)
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in detect_duplicates: {e}")
        raise
