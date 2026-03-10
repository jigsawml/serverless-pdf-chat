"""
Delete a document: remove its Bedrock Knowledge Base data source, delete S3
objects under the document prefix, delete conversation memory, and remove the
document record from DynamoDB.
"""
import os
import json
import boto3
from aws_lambda_powertools import Logger

DOCUMENT_TABLE = os.environ["DOCUMENT_TABLE"]
MEMORY_TABLE = os.environ["MEMORY_TABLE"]
BUCKET = os.environ["BUCKET"]
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "")

ddb = boto3.resource("dynamodb")
document_table = ddb.Table(DOCUMENT_TABLE)
memory_table = ddb.Table(MEMORY_TABLE)
s3 = boto3.client("s3")
bedrock_agent = boto3.client("bedrock-agent")
logger = Logger()


@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    user_id = event["requestContext"]["authorizer"]["claims"]["sub"]
    document_id = event["pathParameters"]["documentid"]

    response = document_table.get_item(
        Key={"userid": user_id, "documentid": document_id}
    )
    if "Item" not in response:
        return {
            "statusCode": 404,
            "headers": _cors_headers(),
            "body": json.dumps({"error": "Document not found"}),
        }
    document = response["Item"]
    logger.info({"document": document})

    # Delete Bedrock KB data source if present (removes ingested vectors)
    data_source_id = document.get("data_source_id")
    if KNOWLEDGE_BASE_ID and data_source_id:
        try:
            bedrock_agent.delete_data_source(
                knowledgeBaseId=KNOWLEDGE_BASE_ID,
                dataSourceId=data_source_id,
            )
            logger.info("Deleted data source", extra={"data_source_id": data_source_id})
        except Exception as e:
            logger.warning(
                "Failed to delete data source (may already be gone)",
                extra={"error": str(e)},
            )

    # Delete conversation memory records
    for conv in document.get("conversations", []):
        try:
            memory_table.delete_item(Key={"SessionId": conv["conversationid"]})
        except Exception as e:
            logger.warning("Failed to delete memory item", extra={"error": str(e)})

    # Delete document record
    document_table.delete_item(
        Key={"userid": user_id, "documentid": document_id}
    )

    # Delete all S3 objects under this document's prefix (PDF, metadata, etc.)
    prefix = f"{user_id}/{document['filename']}/"
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            s3.delete_object(Bucket=BUCKET, Key=obj["Key"])
            logger.info("Deleted S3 object", extra={"key": obj["Key"]})

    return {
        "statusCode": 200,
        "headers": _cors_headers(),
        "body": json.dumps({}, default=str),
    }


def _cors_headers():
    return {
        "Content-Type": "application/json",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "*",
    }
