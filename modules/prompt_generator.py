import boto3
import json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-2")

def run(input_data: dict):
    topic = input_data.get("text", "")
    
    prompt = (
        f"Generate a Claude-compatible prompt to respond to this topic:\n"
        f"{topic}\n"
        f"Tone: Informative, professional, engaging.\n"
        f"Only return the generated prompt."
        f"Avoid any additional text or explanation.\n"
        f"Make sure the prompt is clear and concise.\n"
        f"Do not include any examples or references.\n"
        f"Do not include any disclaimers or warnings.\n"
        f"Do not include any personal opinions or biases.\n"
        f"Do not include any unnecessary details or information.\n"
        f"Do not include any formatting or special characters.\n"
    )

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.7
    }

    try:
        response = bedrock.invoke_model(
            modelId="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )
        data = json.loads(response["body"].read())
        return {"prompt": data["content"][0]["text"].strip()}
    except Exception as e:
        return {"error": str(e)}
