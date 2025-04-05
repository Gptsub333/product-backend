import boto3
import json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-2")

# Define available LLM models
LLM_MODELS = {
    "claude_3_5_sonnet": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude_3_5_sonnet_v2": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "nova_lite": "us.amazon.nova-lite-v1:0",
    "nova_pro": "us.amazon.nova-pro-v1:0"
}

def run(input_data: dict):
    llm_choice = input_data.get("llm", "claude_3_5_sonnet")  # Default to Claude 3.5 Sonnet if no input provided
    topic = input_data.get("text", "")

    if llm_choice not in LLM_MODELS:
        return {"error": "Invalid LLM choice. Please choose from: 'claude_3_5_sonnet', 'claude_3_5_sonnet_v2', 'nova_lite', 'nova_pro'"}

    model_id = LLM_MODELS[llm_choice]

    prompt = (
        f"Generate a response for the following topic:\n"
        f"Topic: {topic}\n"
        f"Tone: Informative, professional, and engaging."
        f"Only return the generated response."
        f"Avoid any additional text or explanation.\n"
        f"Make sure the response is clear and concise.\n"
        f"Do not include any examples or references.\n"
        f"Do not include any disclaimers or warnings.\n"
    )

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.7
    }

    try:
        response = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )
        data = json.loads(response["body"].read())
        return {"response": data["content"][0]["text"].strip()}
    except Exception as e:
        return {"error": str(e)}
