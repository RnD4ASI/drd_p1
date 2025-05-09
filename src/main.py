import os
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import json
import yaml
import time
from dotenv import load_dotenv

from src.utility import DataUtility
from src.embedding import embed_attributes
from src.clustering import cluster_embeddings
from src.duplicate_detection import detect_duplicates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('deduplication.log')
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or use default configuration.
    
    Parameters:
        config_path (Optional[str]): Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    default_config = {
        "embedding": {
            "model_path": "model/FinLang:finance-embeddings-investopedia",
            "provider": "huggingface"
        },
        "clustering": {
            "max_cluster_size": 10,
            "max_clusters": 20
        },
        "language_model": {
            "model_path": "model/Qwen:Qwen3-1.7B",
            "provider": "huggingface"
        },
        "output": {
            "tmp_dir": "tmp",
            "result_dir": "result"
        }
    }
    
    if config_path:
        try:
            config_path = Path(config_path)
            if config_path.suffix == '.json':
                with open(config_path, 'r') as f:
                    config = json.load(f)
            elif config_path.suffix in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                logger.warning(f"Unsupported config file format: {config_path.suffix}")
                config = default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            config = default_config
    else:
        config = default_config
    
    return config

def process_attributes(input_file: str, config_path: Optional[str] = None) -> str:
    """Process data attributes to identify duplicates.
    
    Parameters:
        input_file (str): Path to the input CSV file
        config_path (Optional[str]): Path to the configuration file
        
    Returns:
        str: Path to the output CSV file
    """
    start_time = time.time()
    logger.info(f"Starting attribute deduplication process for {input_file}")
    
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Configuration loaded: {config}")
    
    # Create output directories
    tmp_dir = Path(config["output"]["tmp_dir"])
    result_dir = Path(config["output"]["result_dir"])
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    # Create subdirectories for organization
    embedding_dir = tmp_dir / "embeddings"
    clustering_dir = tmp_dir / "clustering"
    duplicate_dir = tmp_dir / "duplicates"
    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(clustering_dir, exist_ok=True)
    os.makedirs(duplicate_dir, exist_ok=True)
    
    # Load input data
    try:
        data_utility = DataUtility()
        attributes_df = data_utility.text_operation('load', input_file, file_type='csv')
        logger.info(f"Loaded {len(attributes_df)} attributes from {input_file}")
        
        # Save a copy of the original data
        attributes_df.to_csv(tmp_dir / "original_attributes.csv", index=False)
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        raise
    
    # Step 1: Apply text embedding model
    try:
        logger.info("Step 1: Applying text embedding model")
        embeddings = embed_attributes(
            attributes_df=attributes_df,
            model_path=config["embedding"]["model_path"],
            provider=config["embedding"]["provider"]
        )
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Save embeddings
        np.save(embedding_dir / "attribute_embeddings.npy", embeddings)
    except Exception as e:
        logger.error(f"Error in embedding step: {e}")
        raise
    
    # Step 2: Apply K-means clustering
    try:
        logger.info("Step 2: Applying K-means clustering")
        attributes_with_clusters, clustering_results = cluster_embeddings(
            embeddings=embeddings,
            attributes_df=attributes_df,
            max_cluster_size=config["clustering"]["max_cluster_size"],
            max_clusters=config["clustering"]["max_clusters"],
            output_dir=clustering_dir
        )
        logger.info(f"Clustered attributes into {clustering_results['n_clusters']} clusters")
        
        # Save clustering results
        with open(clustering_dir / "clustering_results.json", 'w') as f:
            # Convert numpy arrays and numpy data types to Python native types for JSON serialization
            serializable_results = {}
            for k, v in clustering_results.items():
                if isinstance(v, np.ndarray):
                    serializable_results[k] = v.tolist()
                elif isinstance(v, (np.int32, np.int64)):
                    serializable_results[k] = int(v)
                elif isinstance(v, (np.float32, np.float64)):
                    serializable_results[k] = float(v)
                else:
                    serializable_results[k] = v
            json.dump(serializable_results, f, indent=4)
    except Exception as e:
        logger.error(f"Error in clustering step: {e}")
        raise
    
    # Step 3 & 4: Detect duplicates and select best attributes
    try:
        logger.info("Step 3 & 4: Detecting duplicates and selecting best attributes")
        result_df = detect_duplicates(
            attributes_with_clusters=attributes_with_clusters,
            model_path=config["language_model"]["model_path"],
            output_dir=duplicate_dir,
            provider=config["language_model"]["provider"]
        )
        logger.info(f"Completed duplicate detection")
        
        # Save intermediate results
        result_df.to_csv(duplicate_dir / "attributes_with_duplicates.csv", index=False)
    except Exception as e:
        logger.error(f"Error in duplicate detection step: {e}")
        raise
    
    # Step 5: Generate final output
    try:
        logger.info("Step 5: Generating final output")
        
        # Create final output DataFrame
        final_df = attributes_df.copy()
        final_df['is_duplicate'] = result_df['is_duplicate']
        final_df['duplicate_group_id'] = result_df['duplicate_group_id']
        final_df['should_remove'] = ~result_df['keep']
        
        # Save final output
        output_file = result_dir / "deduplication_results.csv"
        final_df.to_csv(output_file, index=False)
        
        # Generate summary
        total_attributes = len(final_df)
        duplicate_attributes = sum(final_df['is_duplicate'])
        attributes_to_remove = sum(final_df['should_remove'])
        attributes_to_keep = total_attributes - attributes_to_remove
        
        summary = {
            "total_attributes": total_attributes,
            "duplicate_attributes": duplicate_attributes,
            "attributes_to_remove": attributes_to_remove,
            "attributes_to_keep": attributes_to_keep,
            "duplicate_percentage": round(duplicate_attributes / total_attributes * 100, 2) if total_attributes > 0 else 0,
            "processing_time_seconds": round(time.time() - start_time, 2)
        }
        
        # Save summary
        with open(result_dir / "deduplication_summary.json", 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Deduplication complete. Results saved to {output_file}")
        logger.info(f"Summary: {summary}")
        
        return str(output_file)
    except Exception as e:
        logger.error(f"Error in final output generation: {e}")
        raise

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Identify duplicate data attributes')
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    
    args = parser.parse_args()
    
    try:
        output_file = process_attributes(args.input, args.config)
        print(f"Deduplication complete. Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
