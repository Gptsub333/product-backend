from fastapi import APIRouter, HTTPException, Form, UploadFile, File, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
import json
import boto3
from datetime import datetime
import os
import uuid
import subprocess
import sys
from dotenv import load_dotenv
import io

# Load environment variables
load_dotenv()

router = APIRouter()

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "chatbot-automation-hackathon-team-parvez")

@router.post("/")
async def query_intent(
    userId: str = Form(...),
    botName: str = Form(...),
    botPurpose: str = Form(...),
    botTone: str = Form(...),
    prompt: str = Form(...),
    urls: Optional[str] = Form(None),
    faqs: Optional[str] = Form(None),
    avatarColor: Optional[str] = Form(None),
    intents: Optional[str] = Form(None),
    files: List[UploadFile] = File(None),
    request: Request = None,
):
    try:
        # Step 1: Build the LLM prompt
        llm_prompt = f"""
        Extract the chatbot name, purpose, and tone from the following text, then generate 3 relevant questions a user might ask this chatbot.
        For each question, identify the primary user intent and any key entities.

        Chatbot name: {botName}
        Purpose: {botPurpose}
        Tone: {botTone}
        Additional context: {prompt}

        Format your response as a JSON object with:
        1. chatbot_info: extracted name, purpose, and tone
        2. questions: array of 3 objects, each with:
           - user_message: the question text
           - intent: primary user intent (like greeting, query, request)
           - entity: object with key-value pairs of important entities in the question
        """

        # Step 2: Call Claude 3.5 Sonnet via AWS Bedrock
        bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-2")

        response = bedrock_runtime.invoke_model(
            modelId="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {"role": "user", "content": llm_prompt}
                ]
            })
        )

        response_body = json.loads(response.get('body').read())
        llm_response = response_body.get('content')[0].get('text')

        # Step 3: Try to parse the response as JSON
        try:
            parsed_output = json.loads(llm_response)
        except json.JSONDecodeError:
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = llm_response[start_idx:end_idx]
                parsed_output = json.loads(json_str)
            else:
                raise ValueError("Unable to extract JSON from response")

        # Step 4: Extract intents from LLM output
        intents_data = []
        if "questions" in parsed_output:
            for question in parsed_output["questions"]:
                intent_item = {
                    "userMessage": question["user_message"],
                    "intent": question["intent"],
                }

                # Add entities if present
                if "entity" in question and question["entity"]:
                    for k, v in question["entity"].items():
                        intent_item["entityName"] = k
                        intent_item["entityValue"] = v
                        break

                intents_data.append(intent_item)

        # Step 5: Upload final result JSON to S3
        s3_client = boto3.client('s3')

        result = {
            "userId": userId,
            "botName": botName,
            "botPurpose": botPurpose,
            "botTone": botTone,
            "prompt": prompt,
            "urls": urls,
            "faqs": faqs,
            "avatarColor": avatarColor,
            "files": [file.filename for file in files] if files else [],
            "intents": intents_data
        }

        result_key = f"{userId}/{botName}/response.json"
        s3_client.put_object(
            Body=json.dumps(result, indent=2),
            Bucket=S3_BUCKET_NAME,
            Key=result_key,
            ContentType='application/json'
        )

        # Step 6: Upload files to S3 and process them
        # Inside your if files: block
        if files:
            for file in files:
                file_key = f"{userId}/{botName}/uploads/{file.filename}"
                content = await file.read()

                # Wrap content in BytesIO so it has a .read() method
                file_stream = io.BytesIO(content)

                s3_client.upload_fileobj(
                    Fileobj=file_stream,
                    Bucket=S3_BUCKET_NAME,
                    Key=file_key
                )

                # Optional: Call setup.py with the S3 path or any other logic
                setup_script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "setup.py")
                subprocess.run([
                    sys.executable,
                    setup_script,
                    userId,
                    botName,
                    file_key
                ])
                print(f"File processed: {file.filename}")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process: {str(e)}")
