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
            print("S3_BUCKET_NAME not found in environment variables")
            return None

        # Get the index map for this chatbot - use botName parameter correctly
        index_map_key = f"{user_id}/{bot_name}/index_map.json"
        print(f"Looking for index map at: {index_map_key}")

        try:
            response = s3_client.get_object(
                Bucket=s3_bucket, Key=index_map_key)
            index_map = json.loads(response['Body'].read().decode('utf-8'))
            print(
                f"Successfully loaded index map with {len(index_map)} entries")
        except Exception as e:
            print(f"Error loading index map from {index_map_key}: {str(e)}")
            # Try fallback paths for backward compatibility
            fallback_paths = [
                f"users/{user_id}/chatbots/{bot_name}/index_map.json",
                f"{user_id}/chatbots/{bot_name}/index_map.json",
                f"{user_id}/{bot_name}/index_map.json"
            ]

            for path in fallback_paths:
                try:
                    print(f"Trying fallback path: {path}")
                    response = s3_client.get_object(Bucket=s3_bucket, Key=path)
                    index_map = json.loads(
                        response['Body'].read().decode('utf-8'))
                    print(
                        f"Successfully loaded index map from fallback path: {path}")
                    break
                except Exception:
                    continue
            else:
                print("Failed to load index map from any path")
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
            try:
                # Get the embedding data
                embedding_key = item["embedding_s3_path"].replace(
                    f"s3://{s3_bucket}/", "")
                print(
                    f"Attempting to retrieve embedding from: {embedding_key}")

                response = s3_client.get_object(
                    Bucket=s3_bucket, Key=embedding_key)
                embedding_data = json.loads(
                    response['Body'].read().decode('utf-8'))

                # Calculate similarity
                similarity = cosine_similarity(
                    query_embedding, embedding_data["embedding"])
                print(
                    f"Chunk ID: {item['chunk_id']}, Similarity: {similarity:.4f}")

                results.append({
                    "chunk_id": item["chunk_id"],
                    "similarity": similarity,
                    "text": embedding_data["chunk_text"]
                })
            except Exception as e:
                print(f"Error processing embedding {embedding_key}: {str(e)}")
                continue

        if not results:
            print("No successful embeddings processed")
            return None

        # Sort by similarity and return top k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = results[:top_k]

        print(
            f"Top {len(top_results)} results with similarities: {[round(r['similarity'], 4) for r in top_results]}")

        # Combine the text from the top results
        context_text = "\n\n---\n\n".join([r["text"] for r in top_results])

        # Truncate if too long (Claude 3.5 Sonnet can handle ~200k tokens)
        # We'll limit to ~150k characters to be safe
        if len(context_text) > 150000:
            print(
                f"Context too long ({len(context_text)} characters), truncating...")
            context_text = context_text[:150000] + "...(truncated)"

        return context_text

    except Exception as e:
        print(f"Error getting context: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run(input_data: dict):
    llm_choice = input_data.get("llm", "claude_3_5_sonnet")
    query = input_data.get("text", "")
    user_id = input_data.get("userId")
    bot_name = input_data.get("botName")

    if user_id is None and "chatbotName" in input_data:  # Handle different key name
        bot_name = input_data.get("chatbotName")

    print(
        f"Processing request: LLM={llm_choice}, userId={user_id}, botName={bot_name}")
    print(f"Query: {query}")

    if not query:
        return {"error": "No query text provided"}

    if llm_choice not in LLM_MODELS:
        return {"error": f"Invalid LLM choice. Please choose from: {', '.join(LLM_MODELS.keys())}"}

    model_id = LLM_MODELS[llm_choice]

    # If user_id and bot_name are provided, get relevant context
    context = None
    if user_id and bot_name:
        context = get_context_from_embeddings(query, user_id, bot_name)
        if context:
            print(f"Retrieved context of length: {len(context)} characters")
        else:
            print("No context retrieved")

    if context:
        prompt = (
            f"I will provide you with sections from a research paper, and a question about the paper. "
            f"Answer the question using ONLY the information provided in these sections. "
            f"If the provided sections don't contain information to answer the question, "
            f"say 'The provided sections don't contain information to answer this question.' "
            f"DO NOT make up or infer information that isn't explicitly stated in the provided sections.\n\n"
            f"Question: {query}\n\n"
            f"Paper sections:\n{context}\n\n"
            f"Answer the question based ONLY on the information in the provided paper sections:"
        )
    else:
        prompt = (
            f"I don't have enough information to answer questions about specific papers. "
            f"The query was: {query}\n\n"
            f"Please upload the paper or provide more context first."
        )

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.2  # Lower temperature to reduce hallucination
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
