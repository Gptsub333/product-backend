from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import json
import boto3
from datetime import datetime
import uuid
import subprocess
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

router = APIRouter()

# Get S3 bucket name from environment variables
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
if not S3_BUCKET_NAME:
    print("Warning: S3_BUCKET_NAME not found in environment variables")
    # Fallback bucket name if not in environment
    S3_BUCKET_NAME = "chatbot-automation-hackathon-team-parvez"

# Adjust the request model to reflect the complete structure


class IntentRequestData(BaseModel):
    userId: str
    botName: str
    botPurpose: str
    botTone: str
    prompt: str
    urls: str = None
    faqs: str = None
    avatarColor: str = None
    files: list = None
    intents: list = None
    text: str = None  # Added for backward compatibility


@router.post("/")
async def query_intent(request_data: IntentRequestData, request: Request):
    input_data = request_data.dict()

    # Extract chatbot information from the input data
    prompt = f"""
    Extract the chatbot name, purpose, and tone from the following text, then generate 3 relevant questions a user might ask this chatbot.
    For each question, identify the primary user intent and any key entities.
    
    Chatbot name: {input_data['botName']}
    Purpose: {input_data['botPurpose']}
    Tone: {input_data['botTone']}
    Additional context: {input_data['prompt']}
    
    Format your response as a JSON object with:
    1. chatbot_info: extracted name, purpose, and tone
    2. questions: array of 3 objects, each with:
       - user_message: the question text
       - intent: primary user intent (like greeting, query, request)
       - entity: object with key-value pairs of important entities in the question
    """

    try:
        # Direct AWS Bedrock client setup for Claude 3.5
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-2"  # Update with your preferred region
        )

        # Claude 3.5 Sonnet model call
        response = bedrock_runtime.invoke_model(
            modelId="us.anthropic.claude-3-5-sonnet-20240620-v1:0",  # Claude 3.5 model ID
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )

        # Parse the response
        response_body = json.loads(response.get('body').read())
        llm_response = response_body.get('content')[0].get('text')

        # Try to parse the response as JSON
        try:
            parsed_output = json.loads(llm_response)
        except json.JSONDecodeError:
            # If not valid JSON, extract JSON portion from the text
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = llm_response[start_idx:end_idx]
                parsed_output = json.loads(json_str)
            else:
                raise ValueError("Unable to extract JSON from response")

        # Format the response in the desired JSON structure
        intents_data = []
        if "questions" in parsed_output:
            for question in parsed_output["questions"]:
                intent_item = {
                    "userMessage": question["user_message"],
                    "intent": question["intent"],
                }

                # Add entities if they exist
                if "entity" in question and question["entity"]:
                    for entity_key, entity_value in question["entity"].items():
                        intent_item["entityName"] = entity_key
                        intent_item["entityValue"] = entity_value
                        break  # Just take the first entity for simplicity

                intents_data.append(intent_item)

        result = {
            "userId": input_data["userId"],
            "botName": input_data["botName"],
            "botPurpose": input_data["botPurpose"],
            "botTone": input_data["botTone"],
            "prompt": input_data["prompt"],
            "urls": input_data.get("urls"),
            "faqs": input_data.get("faqs"),
            "avatarColor": input_data.get("avatarColor"),
            "files": input_data.get("files"),
            "intents": intents_data
        }

        # Store in S3
        s3_client = boto3.client('s3')

        # Use the bucket name from environment variables
        bucket_name = S3_BUCKET_NAME

        # Create the path structure: user_id/chatbot_name/response.json
        file_key = f"{input_data['userId']}/{input_data['botName']}/response.json"

        # Save the result to S3
        s3_client.put_object(
            Body=json.dumps(result, indent=2),
            Bucket=bucket_name,
            Key=file_key,
            ContentType='application/json'
        )

        # Process files if any are provided
        if input_data.get("files") and len(input_data["files"]) > 0:
            for file_info in input_data["files"]:
                if isinstance(file_info, dict) and "fileLocation" in file_info:
                    file_location = file_info["fileLocation"]

                    # Extract username and chatbot name
                    username = input_data["userId"]
                    chatbotname = input_data["botName"]

                    # Call setup.py to process the file
                    print(
                        f"Processing file: {file_location} for user: {username}, chatbot: {chatbotname}")

                    # Get the absolute path to setup.py
                    setup_script = os.path.join(os.path.dirname(
                        os.path.dirname(__file__)), "setup.py")

                    # Run setup.py with the parameters
                    subprocess.run([
                        sys.executable,
                        setup_script,
                        username,
                        chatbotname,
                        file_location
                    ])
                    print(f"File processing completed for: {file_location}")

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process: {str(e)}")
