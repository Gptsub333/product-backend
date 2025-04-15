from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import boto3
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

# Initialize Bedrock client
bedrock = boto3.client("bedrock-runtime", region_name="us-east-2")

router = APIRouter()


class DirectChatRequest(BaseModel):
    userId: str
    chatbotName: str
    llm: str = "claude_3_5_sonnet"
    text: str


@router.post("/")
async def direct_chat(request_data: DirectChatRequest):
    """Direct chat that fetches the paper text and sends it with the question"""
    user_id = request_data.userId
    bot_name = request_data.chatbotName
    query = request_data.text

    # Define available LLM models
    LLM_MODELS = {
        "claude_3_5_sonnet": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "claude_3_5_sonnet_v2": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "nova_lite": "us.amazon.nova-lite-v1:0",
        "nova_pro": "us.amazon.nova-pro-v1:0"
    }
    model_id = LLM_MODELS.get(
        request_data.llm, LLM_MODELS["claude_3_5_sonnet"])

    # Get S3 bucket name
    s3_bucket = os.getenv("S3_BUCKET_NAME")
    if not s3_bucket:
        raise HTTPException(
            status_code=500, detail="S3_BUCKET_NAME not found in environment variables")

    try:
        # Try different possible locations for text files
        paper_text = ""
        possible_paths = [
            f"{user_id}/{bot_name}/texts/",
            f"users/{user_id}/chatbots/{bot_name}/texts/"
        ]

        for prefix in possible_paths:
            try:
                # List objects in this path
                response = s3_client.list_objects_v2(
                    Bucket=s3_bucket,
                    Prefix=prefix
                )

                # If we found files
                if 'Contents' in response:
                    print(f"Found text files in {prefix}")

                    # Process each text file
                    for item in response['Contents']:
                        if item['Key'].endswith('.txt'):
                            s3_response = s3_client.get_object(
                                Bucket=s3_bucket, Key=item['Key'])
                            paper_text += s3_response['Body'].read().decode(
                                'utf-8') + "\n\n"

                    # If we found content, stop looking
                    if paper_text:
                        break
            except Exception as e:
                print(f"Error checking path {prefix}: {str(e)}")
                continue

        if not paper_text:
            return {"response": "I couldn't find any paper content to analyze. Please make sure a document has been uploaded."}

        # Trim if it's too long (Claude has a context window limit)
        if len(paper_text) > 150000:
            paper_text = paper_text[:150000] + "...(truncated)"

        # Prepare prompt with paper text
        prompt = (
            f"I have a research paper with the following content. Please answer the question based on this paper.\n\n"
            f"PAPER CONTENT:\n{paper_text}\n\n"
            f"QUESTION: {query}\n\n"
            f"Please provide a detailed and accurate answer based only on the information in the paper. "
            f"If the answer is not found in the paper content, please indicate that."
        )

        # Call Bedrock
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.2  # Lower temperature for more factual responses
        }

        response = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )
        data = json.loads(response["body"].read())
        return {"response": data["content"][0]["text"].strip()}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
