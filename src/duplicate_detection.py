import os
import json
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
import uuid
import re
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from src.azure_integration import AzureOpenAIClient

logger = logging.getLogger(__name__)

class DuplicateDetector:
    """Detects duplicate data attributes using language models."""
    
    def __init__(self, model_path: str, provider: str = "huggingface", tmp_responses_dir: Optional[Union[str, Path]] = None):
        """Initialize the DuplicateDetector.
        
        Parameters:
            model_path (str): Path to the language model or name of Azure OpenAI model
            provider (str): Model provider, either 'huggingface' or 'azure_openai'
            tmp_responses_dir (Optional[Union[str, Path]]): Directory to save raw LLM responses. Defaults to None.
        """
        self.model_path = model_path
        self.provider = provider.lower()
        self.model = None
        self.tokenizer = None
        self.azure_client = None
        self.tmp_responses_dir = None
        if tmp_responses_dir:
            self.tmp_responses_dir = Path(tmp_responses_dir)
            os.makedirs(self.tmp_responses_dir, exist_ok=True)
        
        # Initialize Azure client if using Azure OpenAI
        if self.provider == "azure_openai":
            self.azure_client = AzureOpenAIClient()
            if not self.azure_client.is_configured:
                logger.warning("Azure OpenAI not fully configured. Will fall back to HuggingFace if Azure completion is requested.")
    
    # utility 1
    def load_model(self):
        """Load the language model if using open sourced language model."""
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
    
    # utility 2
    def generate_response(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.2, repeat_penalty: float = 1.2, provider: str = None) -> str:
        """Generate a response from a selected language model.
        
        Parameters:
            prompt (str): Input prompt
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            repeat_penalty (float): Penalty for token repetition
            provider (str, optional): Model provider to use ('huggingface' or 'azure_openai'). If None, uses the instance's provider.
            
        Returns:
            str: Generated response
        """
        # Use the provided provider if specified, otherwise use the instance's provider
        current_provider = provider if provider else self.provider
        logger.info(f"Generating response with {current_provider} model {self.model_path} (max_tokens={max_tokens}, temp={temperature})")
        try:
            # Use Azure OpenAI for completions if specified
            if current_provider == "azure_openai":
                if not self.azure_client or not self.azure_client.is_configured:
                    raise ValueError("Azure OpenAI client not configured. Check environment variables.")
                
                # Get completion from Azure OpenAI
                try:
                    system_prompt = "You are an helful AI assistant. You analyse the tasks given, think step by step and respond in a valid JSON format."
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
                
                logger.info("Tokenizing prompt for local model generation")
                # Tokenize the prompt
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                # Generate response
                logger.info("Starting model generation...")
                generation_config = GenerationConfig(
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    repeat_penalty=repeat_penalty,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
                
                # Decode the response
                # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract the generated part (remove the prompt)
                prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
                response_tokens = self.tokenizer.decode(outputs[0][prompt_tokens:], skip_special_tokens=True)
                
                logger.info(f"Model generation complete, generated {len(response_tokens)} characters")
                return response_tokens.strip()
            
        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            raise
    
    # utility 3
    def save_response(self, context_name: str, response_text: str):
        """Saves the raw LLM response to a file if tmp_responses_dir is set."""
        if not self.tmp_responses_dir:
            logger.error("tmp_responses_dir is not set. Cannot save LLM response.")
            return
        
        try:
            # Sanitize context_name for filename
            safe_context_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', context_name)
            max_len_context = 100 # Max length for context part of filename
            if len(safe_context_name) > max_len_context:
                safe_context_name = safe_context_name[:max_len_context]

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
            unique_id = uuid.uuid4().hex[:8]
            filename = self.tmp_responses_dir / f"{safe_context_name}_{timestamp}_{unique_id}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(response_text)
            logger.debug(f"Saved LLM response to {filename}")
        except Exception as e:
            logger.error(f"Failed to save LLM response for context '{context_name}': {e}")

    # utility 4     
    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response with enhanced fallback mechanisms for open source models.
        
        Parameters:
            response (str): The raw response from the language model
            
        Returns:
            Dict[str, Any]: Extracted JSON data or default structure
        """
        try:
            # Standard JSON extraction approaches
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
                logger.info("Successfully parsed JSON using standard extraction")
                return result
            except json.JSONDecodeError:
                logger.warning("Standard JSON extraction failed, trying advanced parsing")
                # Continue to advanced parsing if standard extraction fails
            
            # Advanced parsing for Qwen3 4B and similar models
            # These models often provide reasoning and then a JSON example
            
            # Look for a complete JSON structure with "duplicates" key
            json_pattern = r'\{\s*"duplicates"\s*:\s*\[.*?\]\s*\}'
            matches = re.findall(json_pattern, response, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        result = json.loads(match)
                        logger.info("Successfully parsed JSON using regex pattern")
                        return result
                    except json.JSONDecodeError:
                        continue
            
            # If no valid JSON found, try to extract structured information from the text
            logger.warning("No valid JSON found, attempting to extract structured information")
            
            # Extract duplicate groups from the reasoning
            # This is specific to the duplicate detection task
            groups = {}
            
            # Look for group definitions in the text
            group_pattern = r'group(\d+)[^\w].*?:([^\n]+)'
            group_matches = re.findall(group_pattern, response, re.IGNORECASE)
            
            # Also look for explicit lists of attributes in groups
            list_pattern = r'- ([\w_]+),\s*([\w_]+)(?:,\s*([\w_]+))?\s*\(group(\d+)\)'
            list_matches = re.findall(list_pattern, response, re.IGNORECASE)
            
            # Process group matches
            for group_id, attrs_text in group_matches:
                attrs = re.findall(r'([\w_]+)', attrs_text)
                if attrs:
                    group_name = f"group{group_id}"
                    if group_name not in groups:
                        groups[group_name] = []
                    groups[group_name].extend(attrs)
            
            # Process list matches
            for match in list_matches:
                attrs = [attr for attr in match[:-1] if attr]  # All items except the last (group id)
                group_id = match[-1]  # Last item is the group id
                group_name = f"group{group_id}"
                if group_name not in groups:
                    groups[group_name] = []
                groups[group_name].extend(attrs)
            
            # If no groups found through patterns, try a more aggressive approach
            if not groups:
                # Look for sections where attributes are listed together
                sections = re.split(r'\n\s*\n', response)
                for section in sections:
                    if "duplicate" in section.lower() and "group" in section.lower():
                        # Extract attribute names that appear in this section
                        attr_pattern = r'([a-zA-Z_]+(?:_[a-zA-Z_]+)*)'  # Match attribute_name format
                        attrs = re.findall(attr_pattern, section)
                        if len(attrs) >= 2:  # Need at least 2 attributes to form a duplicate group
                            group_name = f"group{len(groups) + 1}"
                            groups[group_name] = attrs
            
            # Construct a result dictionary from extracted groups
            result = {"duplicates": []}
            all_attrs_in_groups = set()
            
            # Add all attributes found in groups
            for group_name, attrs in groups.items():
                for attr in attrs:
                    all_attrs_in_groups.add(attr)
                    result["duplicates"].append({
                        "attribute_name": attr,
                        "is_duplicate": True,
                        "duplicate_group_id": group_name
                    })
            
            if result["duplicates"]:
                logger.info(f"Successfully extracted {len(groups)} duplicate groups using text analysis")
                return result
            
            # If we reach here, we couldn't extract any structured information
            logger.warning("Could not extract structured information, returning default structure")
            return {"duplicates": []}
            
        except Exception as e:
            logger.error(f"Error in extract_json_from_response: {e}")
            return {"duplicates": []}


    #functional process 1
    def detect_duplicates_in_cluster(self, cluster_attributes: pd.DataFrame, attributes_df: pd.DataFrame = None, incremental: bool = False, provider: str = None) -> Dict[str, Any]:
        logger.info(f"Starting duplicate detection for cluster with {len(cluster_attributes)} attributes")
        """Detect duplicates within a cluster.
        
        Parameters:
            cluster_attributes (pd.DataFrame): DataFrame containing attributes in a cluster
            attributes_df (pd.DataFrame, optional): DataFrame containing all attributes
            incremental (bool, optional): Flag for incremental processing
            provider (str, optional): Model provider ('huggingface' or 'azure_openai'). If None, uses the instance's provider.
            
        Returns:
            Dict[str, Any]: Dictionary containing duplicate detection results
        """
        
        if len(cluster_attributes) <= 1:
            # No duplicates if only one attribute in the cluster
            logger.warning("Cluster has <= 1 attribute, no duplicates to find.")
            return {
                "attributes": cluster_attributes['attribute_name'].tolist(),
                "duplicates": [],
                "duplicate_groups": {}
            }
        
        else:
            # Prepare the prompt
            attributes_text = "\n".join([
                f"{i+1}. Name: {row['attribute_name']} | Definition: {row['attribute_definition']}"
                for i, (_, row) in enumerate(cluster_attributes.iterrows())
            ])
            
            prompt = f"""
            I have a set of data attributes that may contain duplicates. Analyze the following attributes and identify which ones are duplicates of each other.

            {attributes_text}

            For each attribute, determine if it is a duplicate of any other attribute in the list. Two attributes are considered duplicates if they represent the same concept or similar semantic meaning, even if they have different names or definitions.
            
            You must return your final answer as a valid JSON object in the following structure:
            ```json
            {{
                "duplicates": [
                    {{
                        "attribute_name": "name_of_the_attribute",
                        "is_duplicate": true/false,
                        "duplicate_group_id": "group_identifier_for_duplicates" 
                    }},
                    ...
                ]
            }}
            ```
            Assign the same duplicate_group_id to all attributes that are duplicates of each other. Use a simple identifier like "group1", "group2", etc. If an attribute is not a duplicate, do not include a duplicate_group_id for it.

            Importantly, during the thinking process, even when using json object, refrain from surrounding the json with "```json" or "```" as delimiter.
            """
            
            # Generate response using the specified provider if provided
            response = self.generate_response(prompt, provider=provider)
            
            # Save LLM response
            cluster_id_for_filename = "unknown_cluster"
            if not cluster_attributes.empty and 'cluster_id' in cluster_attributes.columns:
                cluster_id_for_filename = str(cluster_attributes['cluster_id'].iloc[0])
            self.save_response(f"detect_duplicates_cluster_{cluster_id_for_filename}", response)
            
            # Parse the JSON response using the enhanced extraction method
            logger.info("Parsing response from language model using enhanced extraction")
            result = self.extract_json_from_response(response)
            
            # Validate the parsed JSON structure
            try:        
                # test 1: check if the result is a dictionary and contains the "duplicates" key
                if not isinstance(result, dict) or "duplicates" not in result or not isinstance(result["duplicates"], list):
                    logger.error(f"Invalid JSON structure: 'duplicates' key missing or not a list. Response: {response}")
                    logger.warning("Using default empty result structure due to invalid JSON structure")
                    # Create a default result structure with no duplicates
                    result = {
                        "duplicates": []
                    }
                    # For each attribute in the cluster, add a non-duplicate entry
                    for _, row in cluster_attributes.iterrows():
                        result["duplicates"].append({
                            "attribute_name": row["attribute_name"],
                            "is_duplicate": False
                        })
                
                # test 2: check if each item in the "duplicates" list is a dictionary and contains the required keys
                valid_items = []
                for item in result["duplicates"]:
                    if isinstance(item, dict) and \
                       "attribute_name" in item and \
                       "is_duplicate" in item and \
                       isinstance(item["is_duplicate"], bool) and \
                       (not item["is_duplicate"] or "duplicate_group_id" in item):
                        valid_items.append(item)
                    else:
                        logger.error(f"Invalid item in 'duplicates' list: {item}. Skipping this item.")
                
                # Replace the duplicates list with only valid items
                result["duplicates"] = valid_items
                
                # If we ended up with no valid items, create default non-duplicate entries
                if not valid_items:
                    logger.warning("No valid items found in duplicates list. Creating default entries.")
                    for _, row in cluster_attributes.iterrows():
                        result["duplicates"].append({
                            "attribute_name": row["attribute_name"],
                            "is_duplicate": False
                        })
                        
            except Exception as e: # Catch any errors during validation
                logger.error(f"Error during JSON validation in detect_duplicates_in_cluster: {e}. Using default structure.")
                # Create a default result structure with no duplicates
                result = {
                    "duplicates": []
                }
                # For each attribute in the cluster, add a non-duplicate entry
                for _, row in cluster_attributes.iterrows():
                    result["duplicates"].append({
                        "attribute_name": row["attribute_name"],
                        "is_duplicate": False
                    })

            # Reformat results (integrating the logic from the previous reformat_results function)
            logger.info("Reformatting duplicate detection results")
            duplicate_groups = {}
            for item in result.get("duplicates", []):
                if item.get("is_duplicate") and "duplicate_group_id" in item:
                    group_id = item["duplicate_group_id"]
                    # Ensure group_id is a string, as it's used as a dictionary key
                    if not isinstance(group_id, str):
                        logger.warning(f"Duplicate group_id '{group_id}' is not a string. Converting. Item: {item}")
                        group_id = str(group_id)
                    if group_id not in duplicate_groups:
                        duplicate_groups[group_id] = []
                    duplicate_groups[group_id].append(item["attribute_name"])
                        
            # Create final result dictionary for this cluster
            logger.info(f"Duplicate detection complete, found {len(duplicate_groups)} duplicate groups")
            return {
                "attributes": cluster_attributes['attribute_name'].tolist(),
                "duplicates": result.get("duplicates", []),
                "duplicate_groups": duplicate_groups
            }


    #functional process 2   
    def select_best_attribute(self, duplicate_group: List[str], attributes_df: pd.DataFrame, incremental: bool = False, provider: str = None) -> str:
        logger.info(f"Selecting best attribute from group of {len(duplicate_group)} attributes")
        """Select the best attribute from a duplicate group.
        
        Parameters:
            duplicate_group (List[str]): List of attribute names in the duplicate group
            attributes_df (pd.DataFrame): DataFrame containing attribute definitions
            incremental (bool, optional): Flag for incremental processing
            provider (str, optional): Model provider ('huggingface' or 'azure_openai'). If None, uses the instance's provider.
            
        Returns:
            str: Name of the best attribute
        """
        try:
            # Get attribute details
            group_df = attributes_df[attributes_df['attribute_name'].isin(duplicate_group)].copy()
            
            # Check if we're in incremental mode and if there are any existing attributes
            if incremental and 'source' in group_df.columns and 'existing' in group_df['source'].values:
                # Prioritize existing attributes
                existing_attrs = group_df[group_df['source'] == 'existing']['attribute_name'].tolist()
                if existing_attrs:
                    # If there are multiple existing attributes, choose the one with the lowest first_seen_in_round
                    if len(existing_attrs) > 1 and 'first_seen_in_round' in group_df.columns:
                        # Ensure 'first_seen_in_round' is numeric for min() to work correctly if it's not already
                        group_df['first_seen_in_round'] = pd.to_numeric(group_df['first_seen_in_round'], errors='coerce')
                        earliest_round = group_df[group_df['source'] == 'existing']['first_seen_in_round'].min()
                        earliest_attrs = group_df[(group_df['source'] == 'existing') & 
                                                 (group_df['first_seen_in_round'] == earliest_round)]['attribute_name'].tolist()
                        if earliest_attrs: # Ensure list is not empty after filtering
                            return earliest_attrs[0]
                    return existing_attrs[0]  # Return the first existing attribute
            
            # Create text representation of attributes
            attributes_text = ""
            for _, row in group_df.iterrows():
                source_tag = f" [EXISTING]" if 'source' in row and row['source'] == 'existing' else ""
                attributes_text += f"Attribute Name: {row['attribute_name']} | {source_tag} | Attribute Definition: {row['attribute_definition']}\n"
            
            prompt = f"""
            You are a data expert follows best practice of data management practice. You are given a set of data attributes that have been identified as duplicates with each other. You are tasked to analyse those data attributes and select one from name with the best quality in terms of naming convention and definition.

            This is the set of data attributes:
            {attributes_text}

            A good attribute should have:
            1. A clear, descriptive name that follows standard naming conventions
            2. A comprehensive and precise definition
            3. Consistency with industry standards or common practices
            {'4. Additional consideration is - selection priority must be given to attributes marked as [EXISTING] unless a new attribute is significantly better' if incremental else ''}

            After your analysis, you must return your final answer as a valid JSON object with the following structure:
            ```json
            {{
                "best_attribute": "name of the best attribute",
                "reasoning": "explanation for why this attribute was selected"
            }}
            ```

            Importantly, during the thinking process, even when using json object, refrain from surrounding the json with "```json" or "```" as delimiter.
            """
            
            # Generate response using the specified provider if provided
            response = self.generate_response(prompt, provider=provider)

            # Save LLM response
            group_name_for_filename = "_" .join(sorted(duplicate_group))[:50] # Truncate if too long
            self.save_response(f"select_best_attribute_group_{group_name_for_filename}", response)
            
            # Parse the JSON response using the enhanced extraction method
            logger.debug(f"Attempting to extract JSON from response in select_best_attribute for group: {duplicate_group}. Response: {response[:500]}...")
            result = self.extract_json_from_response(response)

            if result and isinstance(result, dict) and "best_attribute" in result:
                best_attribute_name = result.get("best_attribute")
                if isinstance(best_attribute_name, str) and best_attribute_name in duplicate_group:
                    logger.info(f"Successfully selected best attribute: '{best_attribute_name}' from group {duplicate_group} using enhanced parsing.")
                    return best_attribute_name
                else:
                    logger.error(
                        f"LLM returned 'best_attribute' ('{best_attribute_name}') but it's invalid or not in the "
                        f"duplicate_group {duplicate_group}. Response: {response[:500]}"
                    )
                    logger.warning(f"Defaulting to first attribute '{duplicate_group[0]}' due to invalid 'best_attribute' value.")
            else:
                logger.error(
                    f"Failed to extract valid JSON with 'best_attribute' key from LLM response in select_best_attribute "
                    f"for group {duplicate_group}. Response: {response[:500]}"
                )
                logger.warning(f"Defaulting to first attribute '{duplicate_group[0]}' due to parsing failure or missing key.")
            
            # Fallback if parsing failed or attribute is invalid
            # Assumes duplicate_group is not empty due to prior checks in the function.
            return duplicate_group[0]
            
        except Exception as e: # Catches errors from self.generate_response() or other non-JSON issues
            logger.error(f"Error in select_best_attribute (non-JSON processing error or during generation): {e}. Defaulting to first attribute.")
            if not duplicate_group:
                logger.error("Cannot select best attribute from an empty group.")
                raise ValueError("Cannot select best attribute from an empty group.")
            return duplicate_group[0]

# main process
def detect_duplicates(attributes_with_clusters: pd.DataFrame, model_path: str, 
                     output_dir: Union[str, Path] = None, provider: str = "huggingface",
                     incremental: bool = False, tmp_llm_responses_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Detect duplicates in clustered attributes.
    
    Parameters:
        attributes_with_clusters (pd.DataFrame): DataFrame containing attributes with cluster assignments
        model_path (str): Path to the language model
        output_dir (Union[str, Path]): Directory to save outputs
        provider (str): Model provider ('huggingface' or 'azure_openai')
        incremental (bool): Flag for incremental processing
        tmp_llm_responses_dir (Optional[Union[str, Path]]): Directory to save raw LLM responses.
        
    Returns:
        pd.DataFrame: DataFrame with duplicate detection results
    """
    try:
        # Initialize duplicate detector
        detector = DuplicateDetector(model_path, provider=provider, tmp_responses_dir=tmp_llm_responses_dir)
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
            duplicate_results = detector.detect_duplicates_in_cluster(cluster_attributes, attributes_with_clusters, incremental=incremental)
            
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
            best_attribute = detector.select_best_attribute(attribute_names, attributes_with_clusters, incremental=incremental, provider=provider)
            
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
