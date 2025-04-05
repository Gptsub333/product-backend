from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json
import boto3

router = APIRouter()

# Adjust the request model to reflect the new structure
class IntentRequestData(BaseModel):
    chatbot_name: str
    bot_purpose: str
    bot_tone: str
    text: str

@router.post("/")
async def query_intent(request_data: IntentRequestData):
    input_data = request_data.dict()
    
    # Extract chatbot information from the input data
    prompt = f"""
    Extract the chatbot name, purpose, and tone from the following text, then generate 3 relevant questions a user might ask this chatbot.
    For each question, identify the primary user intent and any key entities.
    
    Chatbot name: {input_data['chatbot_name']}
    Purpose: {input_data['bot_purpose']}
    Tone: {input_data['bot_tone']}
    Text: {input_data['text']}
    
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
        
        # Return the formatted questions with intent and entity information
        if "questions" in parsed_output:
            return {"questions": parsed_output["questions"], "chatbot_info": parsed_output["chatbot_info"]}
        else:
            return parsed_output
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process: {str(e)}")
