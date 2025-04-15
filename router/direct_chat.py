from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import boto3
import traceback
import json
import os
from dotenv import load_dotenv
from modules.vector_search import VectorSearch

# Load environment variables
load_dotenv()

router = APIRouter()

# Initialize Bedrock client with specific configuration
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name="us-east-2",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)


class ContentSource(BaseModel):
    type: str
    media_type: str = None
    data: str = None


class ContentItem(BaseModel):
    type: str
    text: str = None
    source: ContentSource = None


class Message(BaseModel):
    role: str
    content: List[ContentItem]


class DirectChatRequest(BaseModel):
    userId: str
    chatbotName: str
    llm: str
    text: str


@router.post("/")
async def direct_chat(request: DirectChatRequest):
    """Chat using RAG with semantic search"""
    try:
        # Initialize Bedrock client with the correct region
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name="us-west-2",  # Make sure this matches your model's region
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )

        # Get relevant chunks
        vector_search = VectorSearch()
        similar_chunks = vector_search.search_similar_chunks(
            request.text, request.userId, request.chatbotName)

        # Prepare context from chunks
        if similar_chunks:
            context = "\n\n".join([chunk['text'] for chunk in similar_chunks])
            print(f"Using context from {len(similar_chunks)} chunks")
        else:
            print("No relevant context found")
            return {"response": "I couldn't find relevant information to answer your question. Please try rephrasing or ask another question."}

        # Prepare the messages in the correct format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Answer this question about a research paper using ONLY the provided context. "
                            f"If you can't fully answer from the context, say what you can and note what's missing.\n\n"
                            f"Question: {request.text}\n\n"
                            f"Context from paper:\n{context}"
                        )
                    }
                ]
            }
        ]

        # Make the API call - removed inferenceProfile parameter
        try:
            print(
                f"Making Bedrock API call to model: anthropic.claude-3-5-sonnet-20240620-v1:0")
            response = bedrock.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "messages": messages
                }),
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response.get('body').read())
            return {"response": response_body.get('content')[0].get('text', '').strip()}

        except bedrock.exceptions.ValidationException as ve:
            print(f"Bedrock validation error: {str(ve)}")
            return {"error": "There was an issue with the model configuration. Please check your model access."}

        except Exception as e:
            print(f"Bedrock API error: {str(e)}")
            return {"error": "There was an error processing your request. Please try again later."}

    except Exception as e:
        print(f"Error in direct_chat:\n{traceback.format_exc()}")
        return {"error": str(e)}


@router.get("/verify/{user_id}/{bot_name}")
async def verify_knowledge_base(user_id: str, bot_name: str):
    """Verify the knowledge base exists and is accessible"""
    try:
        s3_client = boto3.client('s3')

        # Check for uploaded files
        uploads_path = f"{user_id}/{bot_name}/uploads/"
        upload_response = s3_client.list_objects_v2(
            Bucket=os.getenv("S3_BUCKET_NAME"),
            Prefix=uploads_path
        )

        # Check for embeddings
        embeddings_path = f"{user_id}/{bot_name}/embeddings/"
        embedding_response = s3_client.list_objects_v2(
            Bucket=os.getenv("S3_BUCKET_NAME"),
            Prefix=embeddings_path
        )

        return {
            "uploads": {
                "path": uploads_path,
                "files": [obj['Key'] for obj in upload_response.get('Contents', [])]
            },
            "embeddings": {
                "path": embeddings_path,
                "files": [obj['Key'] for obj in embedding_response.get('Contents', [])]
            }
        }

    except Exception as e:
        return {"error": str(e)}
