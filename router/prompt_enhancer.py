from botocore.exceptions import ClientError
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List
import boto3
import json
import os
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


router = APIRouter()


class EnhancementRequest(BaseModel):
    baseText: str = Field(..., description="Core description of the chatbot")
    format: str = Field("statement", description="Output format type")
    keywords: List[str] = Field(
        default=[], description="Key terms to emphasize")
    length: int = Field(200, description="Approximate word count target")
    style: str = Field("formal", description="Framing style of sentences")
    template: str = Field("", description="Optional predefined template")
    tone: str = Field("friendly", description="Conversation style")


S3_BUCKET_NAME = os.getenv(
    "S3_BUCKET_NAME", "chatbot-automation-hackathon-team-parvez")
s3_client = boto3.client('s3')
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name="us-east-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

PROMPT_ENHANCEMENT_TEMPLATE = """Transform this chatbot description into a professional system prompt:

**Base Description**: {baseText}
**Format Requested**: {format}
**Target Length**: ~{length} words
**Style**: {style}
**Tone**: {tone}
**Keywords to Include**: {keywords}
**Template Type**: {template}

You are a Prompt Engineering Assistant specialized in creating comprehensive, structured prompts for AI language models. Your purpose is to help users transform basic descriptions into detailed, effective prompts.
Note: give only the content that you want to give to the AI agent this response is to be automatically sent to the Agent so skip the greeting or small talk give the content from the first word itself.

YOUR RESPONSIBILITIES:

Transform brief descriptions into detailed, structured prompts
Ask clarifying questions to extract necessary information
Organize instructions into logical categories
Provide specific rather than general guidance
Explain your reasoning when helpful

PROMPT CREATION PROCESS:

First, identify the core purpose and domain of the assistant being created
Determine what knowledge sources or information the assistant should access
Define clear responsibilities and specific tasks the assistant should perform
Establish explicit boundaries and limitations
Create response guidelines including formatting preferences and structure
Define appropriate tone and communication style

PROMPT STRUCTURE TO FOLLOW:
When generating prompts, structure them with these components:

IDENTITY & PURPOSE: A clear statement of who the AI is and its primary function
KNOWLEDGE ACCESS: What information sources the AI can reference
CORE RESPONSIBILITIES: Specific tasks and functions the AI should perform
LIMITATIONS & BOUNDARIES: Clear restrictions on what the AI should not do
RESPONSE GUIDELINES: How the AI should structure and format its answers
TONE & APPROACH: Use the specified '{tone}' tone and '{style}' style

RESPONSE FORMAT:

Begin with a brief explanation of what you're providing
Format according to the '{format}' preference
Use headings, bullet points, and categorization for clarity
Target approximately {length} words in your response
Naturally incorporate these keywords: {keywords}
If information is missing, identify what additional details would improve the prompt
Also make sure you are addressing the AI that you are giving these instructions to, the description you provide will be given to a AI agent itself.

Always prioritize clarity, specificity, and practical usability in your generated prompts. Remember that effective prompts provide clear guidance while avoiding unnecessary constraints.
Note: give only the content that you want to give to the AI agent this response is to be automatically sent to the Agent so skip the greeting or small talk give the content from the first word itself.
"""


@router.post("/enhance")
async def enhance_prompt_direct(request: EnhancementRequest = Body(...)):
    try:
        # Format keywords for prompt
        keywords_formatted = ", ".join(
            request.keywords) if request.keywords else "None specified"

        # Directly use the prompt from request body with all parameters
        enhancement_prompt = PROMPT_ENHANCEMENT_TEMPLATE.format(
            baseText=request.baseText,
            format=request.format,
            length=request.length,
            style=request.style,
            tone=request.tone,
            keywords=keywords_formatted,
            template=request.template or "standard"
        )

        response = bedrock_runtime.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": enhancement_prompt
                    }]
                }]
            }),
            contentType="application/json",
            accept="application/json"
        )

        response_body = json.loads(response['body'].read())
        enhanced_prompt = response_body['content'][0]['text'].strip()

        return {
            # "status": "success",
            # "parameters": {
            #     "baseText": request.baseText,
            #     "format": request.format,
            #     "keywords": request.keywords,
            #     "length": request.length,
            #     "style": request.style,
            #     "template": request.template,
            #     "tone": request.tone
            # },
            "enhanced_prompt": enhanced_prompt
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Enhancement failed: {str(e)}"
        )
