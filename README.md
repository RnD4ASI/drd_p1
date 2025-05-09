# Data Attribute Deduplication System

This system identifies duplicate data attributes by using embedding models, clustering, and language models to analyze and classify duplicates. It supports both local Hugging Face models and Azure OpenAI models, as well as incremental deduplication across multiple batches of data.

## Overview

The deduplication process follows these steps:

1. **Text Embedding**: Apply text embedding model on a concatenation of each data attribute's name and definition (using either local Hugging Face models or Azure OpenAI).
2. **Clustering**: Apply K-means to cluster the list of embedding vectors.
3. **Duplicate Detection**: For data attributes in each cluster, leverage a language model to assess which attributes are duplicated.
4. **Best Attribute Selection**: For duplicate groups, select one attribute with the best naming convention and definition.
5. **Final Output**: Generate a CSV file with the original data attributes and whether each should be removed or not.

The system supports two main operational modes:

- **Standard Mode**: Process a single batch of data attributes from scratch
- **Incremental Mode**: Process new batches of data attributes while maintaining consistency with previously deduplicated data

## Directory Structure

```
drd_dedup2/
├── config.yaml           # Configuration file
├── data/                 # Input data directory
│   └── sample_attributes.csv
├── env                   # Sample environment variables file (rename to .env for use)
├── instruction.md        # Detailed instructions for usage
├── model/                # Model directory (for local models)
│   ├── FinLang:finance-embeddings-investopedia/
│   └── Qwen:Qwen3-1.7B/
├── result/               # Final results directory
├── src/                  # Source code
│   ├── azure_integration.py # Azure OpenAI integration
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
- python-dotenv
- openai (for Azure OpenAI integration)
- azure-identity (for Azure OpenAI integration)

## Usage

### Basic Usage

Run the deduplication process with the default configuration (using local Hugging Face models):

```bash
python -m src.main --input data/sample_attributes.csv
```

### Custom Configuration

You can provide a custom configuration file:

```bash
python -m src.main --input data/sample_attributes.csv --config custom_config.yaml
```

### Using Azure OpenAI

To use Azure OpenAI for embeddings and/or language model:

```bash
# For embeddings only
python -m src.main --input data/sample_attributes.csv --embedding_provider azure_openai

# For language model only
python -m src.main --input data/sample_attributes.csv --language_model_provider azure_openai

# For both
python -m src.main --input data/sample_attributes.csv --embedding_provider azure_openai --language_model_provider azure_openai
```

### Incremental Deduplication

To process new batches of data while maintaining consistency with previous results:

```bash
# First round
python -m src.main --input data/batch1.csv

# Second round (incremental)
python -m src.main --input data/batch2.csv --previous_results result/deduplication_results.csv --incremental
```

## Configuration

The system is configurable through a YAML or JSON file. Here are the available configuration options:

```yaml
embedding:
  # Path to the embedding model or Azure OpenAI model name
  model_path: "model/FinLang:finance-embeddings-investopedia"
  # Provider for the embedding model: "huggingface" or "azure_openai"
  provider: "huggingface"
  # Uncomment and modify the following line to use Azure OpenAI for embeddings
  # model_path: "text-embedding-ada-002"
  # provider: "azure_openai"

clustering:
  # Maximum size of each cluster
  max_cluster_size: 10
  # Maximum number of clusters
  max_clusters: 20

language_model:
  # Path to the language model or Azure OpenAI model deployment name
  model_path: "model/Qwen:Qwen3-1.7B"
  # Provider for the language model: "huggingface" or "azure_openai"
  provider: "huggingface"
  # Uncomment and modify the following lines to use Azure OpenAI for language model
  # model_path: "gpt-4o"
  # provider: "azure_openai"

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
   - CSV file with all attributes and their duplicate status
   - Columns include:
     - `attribute_name`: Original attribute name
     - `attribute_definition`: Original attribute definition
     - `is_duplicate`: Whether the attribute is a duplicate (True/False)
     - `duplicate_group_id`: Group ID for duplicate attributes
     - `should_remove`: Whether the attribute should be removed (True for duplicates that are not the best attribute)
     - `source`: (In incremental mode) Whether the attribute is from a previous round or new
     - `first_seen_in_round`: (In incremental mode) The round in which the attribute was first processed

## Azure OpenAI Integration

To use Azure OpenAI, you need to set up environment variables. Create a `.env` file in the project root with the following variables:

```
SCOPE="https://cognitiveservices.azure.com/.default"
TENANT_ID="your-tenant-id"
CLIENT_ID="your-client-id"
CLIENT_SECRET="your-client-secret"
SUBSCRIPTION_KEY="your-subscription-key"
AZURE_API_VERSION="2023-05-15"
AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com"
```

## Incremental Deduplication

The incremental deduplication feature allows processing new batches of data attributes while maintaining consistency with previously deduplicated data. This is useful for:

1. Adding new attributes to an existing data dictionary
2. Ensuring consistency in deduplication decisions across multiple data batches
3. Building a comprehensive data dictionary over time

See the `instruction.md` file for detailed instructions on using this feature.

## For More Information

For detailed instructions on using all features, including Azure OpenAI integration and incremental deduplication, see the `instruction.md` file.
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
