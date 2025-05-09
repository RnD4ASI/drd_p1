# Data Attribute Deduplication System

This system identifies duplicate data attributes by using embedding models, clustering, and language models to analyze and classify duplicates.

## Overview

The deduplication process follows these steps:

1. **Text Embedding**: Apply text embedding model on a concatenation of each data attribute's name and definition.
2. **Clustering**: Apply K-means to cluster the list of embedding vectors.
3. **Duplicate Detection**: For data attributes in each cluster, leverage a language model to assess which attributes are duplicated.
4. **Best Attribute Selection**: For duplicate groups, select one attribute with the best naming convention and definition.
5. **Final Output**: Generate a CSV file with the original data attributes and whether each should be removed or not.

## Directory Structure

```
drd_dedup2/
├── config.yaml           # Configuration file
├── data/                 # Input data directory
│   └── sample_attributes.csv
├── model/                # Model directory
│   ├── FinLang:finance-embeddings-investopedia/
│   └── Qwen:Qwen3-1.7B/
├── result/               # Final results directory
├── src/                  # Source code
│   ├── clustering.py     # Clustering functionality
│   ├── duplicate_detection.py # Duplicate detection using language models
│   ├── embedding.py      # Text embedding functionality
│   ├── main.py           # Main orchestration module
│   └── utility.py        # Utility functions
└── tmp/                  # Temporary files directory
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- transformers
- sentence-transformers
- torch
- pyyaml

## Usage

### Basic Usage

Run the deduplication process with the default configuration:

```bash
python -m src.main --input data/sample_attributes.csv
```

### Custom Configuration

You can provide a custom configuration file:

```bash
python -m src.main --input data/sample_attributes.csv --config custom_config.yaml
```

## Configuration

The system is configurable through a YAML or JSON file. Here are the available configuration options:

```yaml
embedding:
  # Path to the embedding model
  model_path: "model/FinLang:finance-embeddings-investopedia"

clustering:
  # Maximum size of each cluster
  max_cluster_size: 10
  # Maximum number of clusters
  max_clusters: 20

language_model:
  # Path to the language model for duplicate detection
  model_path: "model/Qwen:Qwen3-1.7B"

output:
  # Directory for temporary files
  tmp_dir: "tmp"
  # Directory for final results
  result_dir: "result"
```

## Output

The system generates the following outputs:

1. **Temporary Files** (in `tmp/` directory):
   - Embeddings of attributes
   - Clustering results and visualizations
   - Duplicate detection results for each cluster
   - Best attribute selection for each duplicate group

2. **Final Results** (in `result/` directory):
   - `deduplication_results.csv`: Original attributes with duplicate flags and removal recommendations
   - `deduplication_summary.json`: Summary statistics of the deduplication process

## Customization

The system is designed to be easily customizable:

- **Embedding Models**: Change the embedding model by updating the configuration.
- **Clustering Parameters**: Adjust the maximum cluster size and number of clusters.
- **Language Models**: Use different language models for duplicate detection.
- **Input Data**: Process different data attribute files by specifying a different input file.

## Example

The sample data attributes in `data/sample_attributes.csv` can be used to test the system. The file contains attribute names and definitions, some of which are duplicates of each other.

## Logging

The system logs information to both the console and a log file (`deduplication.log`). This includes progress updates, warnings, and errors.
