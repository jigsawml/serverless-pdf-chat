"""
Lambda that polls Bedrock Knowledge Base ingestion jobs and updates document
status to READY or FAILED in DynamoDB.
"""
import os
import boto3
from aws_lambda_powertools import Logger

DOCUMENT_TABLE = os.environ["DOCUMENT_TABLE"]
KNOWLEDGE_BASE_ID = os.environ["KNOWLEDGE_BASE_ID"]

ddb = boto3.resource("dynamodb")
document_table = ddb.Table(DOCUMENT_TABLE)
bedrock_agent = boto3.client("bedrock-agent")
logger = Logger()


@logger.inject_lambda_context(log_event=False)
def lambda_handler(event, context):
    """
    Periodically scan the document table and check ingestion status for any
    documents still marked as PROCESSING.
    """
    updated = 0
    scan_kwargs = {}
    while True:
        response = document_table.scan(**scan_kwargs)
        for item in response.get("Items", []):
            # Only consider items still in PROCESSING with a recorded ingestion_job_id
            if item.get("docstatus") != "PROCESSING" or "ingestion_job_id" not in item:
                continue
            user_id = item["userid"]
            document_id = item["documentid"]
            ingestion_job_id = item.get("ingestion_job_id")
            if not ingestion_job_id:
                continue
            try:
                job = bedrock_agent.get_ingestion_job(
                    knowledgeBaseId=KNOWLEDGE_BASE_ID,
                    dataSourceId=item["data_source_id"],
                    ingestionJobId=ingestion_job_id,
                )
                status = job["ingestionJob"].get("status", "")
                if status == "COMPLETE":
                    document_table.update_item(
                        Key={"userid": user_id, "documentid": document_id},
                        UpdateExpression="SET docstatus = :ready REMOVE ingestion_job_id",
                        ExpressionAttributeValues={":ready": "READY"},
                    )
                    updated += 1
                    logger.info(
                        {"document_id": document_id, "status": "READY"}
                    )
                elif status == "FAILED":
                    failure_reason = job["ingestionJob"].get(
                        "failureReasons", [{}]
                    )[0].get("message", "Unknown") if job["ingestionJob"].get("failureReasons") else "Unknown"
                    document_table.update_item(
                        Key={"userid": user_id, "documentid": document_id},
                        UpdateExpression="SET docstatus = :failed, ingestion_failure = :reason REMOVE ingestion_job_id",
                        ExpressionAttributeValues={
                            ":failed": "FAILED",
                            ":reason": failure_reason[:500],
                        },
                    )
                    updated += 1
                    logger.warning(
                        {"document_id": document_id, "status": "FAILED", "reason": failure_reason}
                    )
            except Exception as e:
                logger.exception(
                    "Error checking ingestion job",
                    extra={"document_id": document_id, "error": str(e)},
                )
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break
        scan_kwargs["ExclusiveStartKey"] = last_key
    return {"documents_updated": updated}
