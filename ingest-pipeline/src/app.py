import json
import boto3
from utils import secret, opensearch
from loguru import logger
import os
import sys

logger.remove()
logger.add(sys.stdout, level=os.getenv("LOG_LEVEL", "INFO"))

def get_bedrock_client(region):
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    return bedrock_client

def create_vector_embedding_with_bedrock(text, name, bedrock_client):
    payload = {"inputText": f"{text}"}
    body = json.dumps(payload)
    modelId = "amazon.titan-embed-text-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_client.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    embedding = response_body.get("embedding")
    return {"_index": name, "text": text, "vector_field": embedding}

def lambda_handler(event, context):
    logger.info("Starting")
    s3_client = boto3.client('s3')
    region = os.getenv("AWS_REGION", "us-east-1")
    name = os.getenv("INDEX_NAME", "rag")
    username = os.getenv("USERNAME", "osmaster")

    # Get the object from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # Download the file from S3
    local_file_path = '/tmp/data1.json'
    s3_client.download_file(bucket, key, local_file_path)

    # Prepare OpenSearch index with vector embeddings index mapping
    logger.info("Preparing OpenSearch Index")
    opensearch_password = secret.get_secret(name, region)
    opensearch_client = opensearch.get_opensearch_cluster_client(name, opensearch_password, region, username)

    logger.info(f"Checking if index {name} exists in OpenSearch cluster")
    exists = opensearch.check_opensearch_index(opensearch_client, name)
    if not exists:
        logger.info("Creating OpenSearch index")
        success = opensearch.create_index(opensearch_client, name)
        if success:
            logger.info("Creating OpenSearch index mapping")
            success = opensearch.create_index_mapping(opensearch_client, name)
            logger.info("OpenSearch Index mapping created")

    # Read dataset from local file
    logger.info("Reading dataset from local file")
    all_records = []
    with open(local_file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            all_records.append(record)

    # Initialize bedrock client
    bedrock_client = get_bedrock_client(region)

    # Vector embedding using Amazon Bedrock Titan text embedding
    all_json_records = []
    logger.info("Creating embeddings for records")

    # Process records
    for i, record in enumerate(all_records):
        text = record[0]  # assuming the question is the first element in the list
        records_with_embedding = create_vector_embedding_with_bedrock(text, name, bedrock_client)
        logger.info(f"Embedding for record {i} created")
        all_json_records.append(records_with_embedding)
        if i % 500 == 0 or i == len(all_records)-1:
            # Bulk put all records to OpenSearch
            success, failed = opensearch.put_bulk_in_opensearch(all_json_records, opensearch_client)
            all_json_records = []
            logger.info(f"Documents saved {success}, documents failed to save {failed}")

    logger.info("Finished creating records using Amazon Bedrock Titan text embedding")
    logger.info("Cleaning up")
    logger.info("Finished")