#!/bin/bash

# Run Tests for Data Attribute Deduplication System
# This script runs both standard and incremental deduplication tests

# Check if .env file exists (required for Azure OpenAI)
if [ ! -f ".env" ] && grep -q "provider: \"azure_openai\"" config.yaml; then
    echo "Error: .env file not found but Azure OpenAI is configured in config.yaml"
    echo "Please create a .env file with Azure OpenAI credentials (copy from env template)"
    exit 1
fi

# Create directories if they don't exist
mkdir -p result
mkdir -p tmp

echo "======================================================="
echo "Data Attribute Deduplication System - Test Runner"
echo "======================================================="
echo ""

# Set timestamp for this test run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Test run started at: $(date)"
echo ""

# Run standard deduplication on batch1
echo "Running standard deduplication on batch1..."
python -m src.main --input data/batch1.csv --run_id "${TIMESTAMP}_test"

# Check if the run was successful
if [ $? -ne 0 ]; then
    echo "Error: Standard deduplication failed. Exiting."
    exit 1
fi

echo ""
echo "Standard deduplication completed successfully."
echo "Results saved to: result/${TIMESTAMP}_test/"
echo ""


# Print summary
echo "======================================================="
echo "Test Run Summary"
echo "======================================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Standard deduplication: result/${TIMESTAMP}_test/"
echo ""
echo "Latest results are also available at:"
echo "- result/${TIMESTAMP}_test/deduplication_results.csv"
echo ""
echo "To view detailed results, check the run_metadata.json and deduplication.log files in each result directory."
echo "======================================================="


# Set timestamp for this test run
PRIOR_TIMESTAMP=${TIMESTAMP}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Test run started at: $(date)"
echo ""

# Run incremental deduplication on batch2
echo "Running incremental deduplication on batch2..."
python -m src.main --input data/batch2.csv --previous_results result/${PRIOR_TIMESTAMP}_test/deduplication_results.csv --incremental --run_id "${TIMESTAMP}_test"

# Check if the run was successful
if [ $? -ne 0 ]; then
    echo "Error: Incremental deduplication failed. Exiting."
    exit 1
fi

echo ""
echo "Incremental deduplication completed successfully."
echo "Results saved to: result/${TIMESTAMP}_test/"
echo ""

# Print summary
echo "======================================================="
echo "Test Run Summary"
echo "======================================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Incremental deduplication: result/${TIMESTAMP}_test/"
echo ""
echo "Latest results are also available at:"
echo "- result/${TIMESTAMP}_test/deduplication_results.csv"
echo ""
echo "To view detailed results, check the run_metadata.json and deduplication.log files in each result directory."
echo "======================================================="
