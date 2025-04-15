import json
import os
import tiktoken
import boto3
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pathlib import Path
from .text_chunking import TextChunker


def generate_and_store_embeddings(
    chatbot_config_path: str = ".chatbot_config",
    temp_dir: str = None
) -> str:
    """Generate embeddings for document chunks and store in S3"""
    # Load environment variables
    load_dotenv()

    # Get S3 configuration
    s3_bucket = os.getenv("S3_BUCKET_NAME")
    if not s3_bucket:
        raise ValueError("S3_BUCKET_NAME not found in .env file")

    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "ap-south-1")
    )

    # Load chatbot configuration
    if os.path.exists(chatbot_config_path):
        with open(chatbot_config_path, "r") as f:
            chatbot_config = json.load(f)
    else:
        raise ValueError(
            f"Chatbot configuration file {chatbot_config_path} not found")

    # Set the temp directory
    if temp_dir is None:
        temp_dir = Path(os.path.dirname(__file__)) / "temp"
    else:
        temp_dir = Path(temp_dir)
    temp_dir.mkdir(exist_ok=True)

    embeddings_dir = temp_dir / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)

    # Smaller chunks for better retrieval
    chunker = TextChunker(chunk_size=500)
    index_map = []

    s3_texts_prefix = f"{chatbot_config['s3_path']}/texts/"

    if 'username' in chatbot_config:
        fallback_prefix = f"users/{chatbot_config['username']}/chatbots/{chatbot_config['chatbot_name']}/texts/"
        fallback_response = s3_client.list_objects_v2(
            Bucket=s3_bucket, Prefix=fallback_prefix)

        if ('Contents' in fallback_response and
                ('Contents' not in s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_texts_prefix))):
            print(
                f"Warning: Found text files in fallback location: {fallback_prefix}")
            for item in fallback_response['Contents']:
                src_key = item['Key']
                if src_key.endswith('.txt'):
                    dest_key = f"{s3_texts_prefix}{os.path.basename(src_key)}"
                    print(f"Copying {src_key} to {dest_key}")
                    s3_client.copy_object(
                        CopySource={'Bucket': s3_bucket, 'Key': src_key},
                        Bucket=s3_bucket,
                        Key=dest_key
                    )

    try:
        response = s3_client.list_objects_v2(
            Bucket=s3_bucket,
            Prefix=s3_texts_prefix
        )

        if 'Contents' not in response:
            print(f"No text files found in S3")
            return None

        for item in response['Contents']:
            s3_key = item['Key']
            if not s3_key.endswith('.txt'):
                continue

            print(f"Processing S3 file: s3://{s3_bucket}/{s3_key}")

            # Get text content
            s3_response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
            text = s3_response['Body'].read().decode('utf-8')

            # Split into chunks
            chunks = chunker.split_text(text)
            print(f"Split document into {len(chunks)} chunks")

            # Process each chunk
            for i, chunk in enumerate(chunks):
                chunk_id = f"{Path(s3_key).stem}_chunk_{i}"

                # Generate embedding
                embedding = generate_embedding(
                    text=chunk['text'],
                    model_name="amazon.titan-embed-text-v2:0"
                )

                if embedding:
                    # Store chunk data
                    chunk_data = {
                        "chunk_id": chunk_id,
                        "chunk_text": chunk['text'],
                        "embedding": embedding,
                        "token_count": chunk['token_count'],
                        "source_file": s3_key,
                        "chunk_index": i
                    }

                    # Save chunk data to S3
                    s3_chunk_key = f"{chatbot_config['s3_path']}/embeddings/{chunk_id}.json"
                    s3_client.put_object(
                        Bucket=s3_bucket,
                        Key=s3_chunk_key,
                        Body=json.dumps(chunk_data),
                        ContentType="application/json"
                    )

                    # Add to index map
                    index_map.append({
                        "chunk_id": chunk_id,
                        "source_file": s3_key,
                        "chunk_index": i,
                        "token_count": chunk['token_count'],
                        "s3_path": f"s3://{s3_bucket}/{s3_chunk_key}"
                    })

                    print(
                        f"Processed chunk {i+1}/{len(chunks)} for {chunk_id}")

        # Save index map
        s3_index_key = f"{chatbot_config['s3_path']}/index_map.json"
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=s3_index_key,
            Body=json.dumps(index_map),
            ContentType="application/json"
        )

        print(f"Processed {len(index_map)} total chunks")
        return f"s3://{s3_bucket}/{s3_index_key}"

    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def select_model_for_chunk(token_count: int) -> str:
    """
    Simple logic to always select Cohere for embedding
    """
    return "claude"


def generate_embedding(
    text: str,
    model_name: str,
    openai_client: Any = None
) -> Optional[List[float]]:
    try:
        # Initialize Bedrock client with specific configuration
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name="us-east-2",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )

        print(f"Generating embedding for text of length: {len(text)}")

        # Use Titan embedding model v2
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({
                "inputText": text.strip()  # Ensure text is stripped of whitespace
            }),
            contentType="application/json",
            accept="application/json"
        )

        response_body = json.loads(response['body'].read())
        print("Successfully generated embedding")
        return response_body['embedding']

    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        # Print first 5 chars for debugging
        print(f"AWS Access Key ID: {os.getenv('AWS_ACCESS_KEY_ID')[:5]}...")
        print(f"AWS Region: {os.getenv('AWS_REGION', 'us-east-2')}")
        import traceback
        traceback.print_exc()
        return None


# Run
if __name__ == "__main__":
    index_map_path = generate_and_store_embeddings()
    print(f"Embeddings complete. Index map saved at: {index_map_path}")
