"""
S3-triggered Lambda: when a PDF is uploaded, create document and conversation
records, add a data source to the Bedrock Knowledge Base for this document's
S3 prefix, upload metadata for filtering, and start an ingestion job.
"""
import os
import json
from datetime import datetime
import boto3
import PyPDF2
import shortuuid
import urllib
from aws_lambda_powertools import Logger

DOCUMENT_TABLE = os.environ["DOCUMENT_TABLE"]
MEMORY_TABLE = os.environ["MEMORY_TABLE"]
BUCKET = os.environ["BUCKET"]
KNOWLEDGE_BASE_ID = os.environ["KNOWLEDGE_BASE_ID"]

ddb = boto3.resource("dynamodb")
document_table = ddb.Table(DOCUMENT_TABLE)
memory_table = ddb.Table(MEMORY_TABLE)
s3 = boto3.client("s3")
bedrock_agent = boto3.client("bedrock-agent")
logger = Logger()


@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    key = urllib.parse.unquote_plus(event["Records"][0]["s3"]["object"]["key"])
    split = key.split("/")
    user_id = split[0]
    file_name = split[1]  # e.g. "document.pdf" or "document.pdf/document.pdf" - first segment is the "folder" name

    document_id = shortuuid.uuid()

    s3.download_file(BUCKET, key, f"/tmp/{file_name}")

    with open(f"/tmp/{file_name}", "rb") as f:
        reader = PyPDF2.PdfReader(f)
        pages = str(len(reader.pages))

    conversation_id = shortuuid.uuid()
    timestamp = datetime.utcnow()
    timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    # S3 prefix for this document: only this PDF (and metadata) are under this prefix
    s3_prefix = f"{user_id}/{file_name}/"
    bucket_arn = f"arn:aws:s3:::{BUCKET}"

    # Upload metadata file so ingestion tags chunks with documentid for Retrieve filter
    metadata_key = f"{s3_prefix}{file_name}.metadata.json"
    metadata_body = json.dumps({"documentid": document_id})
    s3.put_object(
        Bucket=BUCKET,
        Key=metadata_key,
        ContentType="application/json",
        Body=metadata_body.encode("utf-8"),
    )

    # Create data source in the shared Knowledge Base for this document's prefix
    data_source_name = f"doc-{document_id}"
    try:
        create_response = bedrock_agent.create_data_source(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            name=data_source_name,
            description=f"Document {document_id}",
            dataSourceConfiguration={
                "type": "S3",
                "s3Configuration": {
                    "bucketArn": bucket_arn,
                    "inclusionPrefixes": [s3_prefix],
                    "bucketOwnerAccountId": boto3.client("sts").get_caller_identity()["Account"],
                },
            },
        )
        data_source_id = create_response["dataSource"]["dataSourceId"]
    except Exception as e:
        logger.exception("Failed to create data source", extra={"error": str(e)})
        docstatus = "FAILED"
        data_source_id = ""
        ingestion_job_id = ""

    if data_source_id:
        try:
            start_response = bedrock_agent.start_ingestion_job(
                knowledgeBaseId=KNOWLEDGE_BASE_ID,
                dataSourceId=data_source_id,
            )
            ingestion_job_id = start_response["ingestionJob"]["ingestionJobId"]
            docstatus = "PROCESSING"
        except Exception as e:
            logger.exception("Failed to start ingestion job", extra={"error": str(e)})
            docstatus = "FAILED"
            ingestion_job_id = ""
    else:
        docstatus = "FAILED"
        ingestion_job_id = ""

    document = {
        "userid": user_id,
        "documentid": document_id,
        "filename": file_name,
        "created": timestamp_str,
        "pages": pages,
        "filesize": str(event["Records"][0]["s3"]["object"]["size"]),
        "docstatus": docstatus,
        "conversations": [],
        "knowledge_base_id": KNOWLEDGE_BASE_ID,
    }
    if data_source_id:
        document["data_source_id"] = data_source_id
    if ingestion_job_id:
        document["ingestion_job_id"] = ingestion_job_id

    conversation = {"conversationid": conversation_id, "created": timestamp_str}
    document["conversations"].append(conversation)

    document_table.put_item(Item=document)

    conversation_record = {"SessionId": conversation_id, "History": []}
    memory_table.put_item(Item=conversation_record)

    return {"document_id": document_id, "docstatus": docstatus}
