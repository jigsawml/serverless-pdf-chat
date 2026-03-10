"""
Generate chat response using Bedrock Knowledge Base RetrieveAndGenerate,
filtering by document ID and using DynamoDB for conversation memory.
"""
import os
import json
import boto3
from aws_lambda_powertools import Logger
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

MEMORY_TABLE = os.environ["MEMORY_TABLE"]
DOCUMENT_TABLE = os.environ["DOCUMENT_TABLE"]
MODEL_ID = os.environ["MODEL_ID"]
KNOWLEDGE_BASE_ID = os.environ["KNOWLEDGE_BASE_ID"]

ddb = boto3.resource("dynamodb")
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime")
logger = Logger()


def get_document(user_id, document_id):
    table = ddb.Table(DOCUMENT_TABLE)
    resp = table.get_item(Key={"userid": user_id, "documentid": document_id})
    return resp.get("Item")


def get_conversation_context(conversation_id, last_n=5):
    """Build a short context string from recent messages for the model."""
    history = DynamoDBChatMessageHistory(
        table_name=MEMORY_TABLE,
        session_id=conversation_id,
    )
    messages = history.messages
    if not messages or last_n <= 0:
        return ""
    recent = messages[-last_n * 2 :]  # pairs of human/ai
    parts = []
    for m in recent:
        if hasattr(m, "content"):
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            parts.append(f"{role}: {m.content}")
    if not parts:
        return ""
    return "Previous conversation:\n" + "\n".join(parts) + "\n\n"


def save_messages(conversation_id, human_input, model_output):
    history = DynamoDBChatMessageHistory(
        table_name=MEMORY_TABLE,
        session_id=conversation_id,
    )
    history.add_user_message(human_input)
    history.add_ai_message(model_output)


def retrieve_and_generate_response(document_id, question, conversation_context=""):
    """Call Bedrock RetrieveAndGenerate with filter on documentid."""
    if conversation_context:
        augmented_input = conversation_context + "Current question: " + question
    else:
        augmented_input = question

    retrieval_filter = {
        "equals": {"key": "documentid", "value": document_id},
    }
    request = {
        "input": {"text": augmented_input},
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                "modelArn": f"arn:aws:bedrock:{os.environ.get('AWS_REGION', 'us-east-1')}::foundation-model/{MODEL_ID}",
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": 5,
                        "filter": retrieval_filter,
                    }
                },
                "generationConfiguration": {
                    "temperature": 0.0,
                },
            },
        },
    }
    response = bedrock_agent_runtime.retrieve_and_generate(**request)
    output = response.get("output", {})
    text = output.get("text", "")
    return text.strip()


@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    body = event.get("body") or "{}"
    if isinstance(body, str):
        body = json.loads(body)
    file_name = body.get("fileName", "")
    human_input = body.get("prompt", "")
    document_id = event["pathParameters"]["documentid"]
    conversation_id = event["pathParameters"]["conversationid"]
    user_id = event["requestContext"]["authorizer"]["claims"]["sub"]

    document = get_document(user_id, document_id)
    if not document:
        return {
            "statusCode": 404,
            "headers": _cors_headers(),
            "body": json.dumps({"error": "Document not found"}),
        }

    docstatus = document.get("docstatus", "")
    if docstatus != "READY":
        return {
            "statusCode": 400,
            "headers": _cors_headers(),
            "body": json.dumps(
                {"error": "Document is not ready for chat yet. Status: " + docstatus}
            ),
        }

    conversation_context = get_conversation_context(conversation_id)
    try:
        answer = retrieve_and_generate_response(
            document_id, human_input, conversation_context
        )
    except Exception as e:
        logger.exception("RetrieveAndGenerate failed", extra={"error": str(e)})
        return {
            "statusCode": 500,
            "headers": _cors_headers(),
            "body": json.dumps({"error": "Failed to generate response"}),
        }

    save_messages(conversation_id, human_input, answer)

    return {
        "statusCode": 200,
        "headers": _cors_headers(),
        "body": json.dumps(answer),
    }


def _cors_headers():
    return {
        "Content-Type": "application/json",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "*",
    }
