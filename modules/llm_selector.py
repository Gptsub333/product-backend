import boto3
import json
import os
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

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

# Define available LLM models
LLM_MODELS = {
    "claude_3_5_sonnet": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude_3_5_sonnet_v2": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "nova_lite": "us.amazon.nova-lite-v1:0",
    "nova_pro": "us.amazon.nova-pro-v1:0"
}


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_context_from_embeddings(query, user_id, bot_name, top_k=3):
    """Retrieve relevant context based on the query embedding"""
    try:
        # Get S3 bucket name
        s3_bucket = os.getenv("S3_BUCKET_NAME")
        if not s3_bucket:
            return None

        # Get the index map for this chatbot
        index_map_key = f"{user_id}/{bot_name}/index_map.json"

        try:
            response = s3_client.get_object(
                Bucket=s3_bucket, Key=index_map_key)
            index_map = json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            print(f"Error loading index map: {str(e)}")
            return None

        # Generate embedding for the query
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = response.data[0].embedding

        # Find the most relevant chunks
        results = []
        for item in index_map:
            # Get the embedding data
            embedding_key = item["embedding_s3_path"].replace(
                f"s3://{s3_bucket}/", "")

            response = s3_client.get_object(
                Bucket=s3_bucket, Key=embedding_key)
            embedding_data = json.loads(
                response['Body'].read().decode('utf-8'))

            # Calculate similarity
            similarity = cosine_similarity(
                query_embedding, embedding_data["embedding"])

            results.append({
                "chunk_id": item["chunk_id"],
                "similarity": similarity,
                "text": embedding_data["chunk_text"]
            })

        # Sort by similarity and return top k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = results[:top_k]

        # Combine the text from the top results
        context_text = "\n\n---\n\n".join([r["text"] for r in top_results])
        return context_text

    except Exception as e:
        print(f"Error getting context: {str(e)}")
        return None


def run(input_data: dict):
    # Default to Claude 3.5 Sonnet if no input provided
    llm_choice = input_data.get("llm", "claude_3_5_sonnet")
    query = input_data.get("text", "")
    user_id = input_data.get("userId")
    bot_name = input_data.get("botName")

    if not query:
        return {"error": "No query text provided"}

    if llm_choice not in LLM_MODELS:
        return {"error": "Invalid LLM choice. Please choose from: 'claude_3_5_sonnet', 'claude_3_5_sonnet_v2', 'nova_lite', 'nova_pro'"}

    model_id = LLM_MODELS[llm_choice]

    # If user_id and bot_name are provided, get relevant context
    context = None
    if user_id and bot_name:
        context = get_context_from_embeddings(query, user_id, bot_name)

    if context:
        prompt = (
            f"Answer the following question using the provided context:\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            f"Use the context to provide a detailed answer. If the question cannot be answered from the context, say so."
        )
    else:
        prompt = (
            f"Generate a response for the following topic:\n"
            f"Topic: {query}\n"
            f"Tone: Informative, professional, and engaging."
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
