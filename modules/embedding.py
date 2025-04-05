import json
import os
import tiktoken
import openai
import boto3
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pathlib import Path


def generate_and_store_embeddings(
    chatbot_config_path: str = ".chatbot_config",
    temp_dir: str = None
) -> str:
    """
    Generate embeddings for document chunks from S3 and store back to S3
    """
    # Load environment variables
    load_dotenv()

    # Initialize OpenAI client
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("API key for OpenAI not found in .env file")
    openai.api_key = openai_api_key

    # Get S3 configuration
    s3_bucket = os.getenv("S3_BUCKET_NAME")
    if not s3_bucket:
        raise ValueError("S3_BUCKET_NAME not found in .env file")

    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )

    # Load chatbot configuration
    if os.path.exists(chatbot_config_path):
        with open(chatbot_config_path, "r") as f:
            chatbot_config = json.load(f)
    else:
        raise ValueError(
            f"Chatbot configuration file {chatbot_config_path} not found")

    # Set the temp directory if provided
    if temp_dir is None:
        temp_dir = Path(os.path.dirname(__file__)) / "temp"
    else:
        temp_dir = Path(temp_dir)
    temp_dir.mkdir(exist_ok=True)

    # Create embeddings directory
    embeddings_dir = temp_dir / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)

    # Initialize tokenizer for counting tokens
    cl_tokenizer = tiktoken.get_encoding("cl100k_base")

    # Define models and their token limits
    models = {
        "small": {"name": "text-embedding-3-small", "max_tokens": 8191, "dimension": 1536},
        "large": {"name": "text-embedding-3-large", "max_tokens": 8191, "dimension": 3072},
        "ada": {"name": "text-embedding-ada-002", "max_tokens": 8191, "dimension": 1536}
    }

    # Create index map
    index_map = []

    # Get the S3 path prefix for text files
    s3_texts_prefix = f"{chatbot_config['s3_path']}/texts/"

    # Check if there's a mismatch between the config and where files are actually saved
    if 'username' in chatbot_config:
        # Try the hardcoded path as a fallback
        fallback_prefix = f"users/{chatbot_config['username']}/chatbots/{chatbot_config['chatbot_name']}/texts/"

        # List objects using fallback path
        fallback_response = s3_client.list_objects_v2(
            Bucket=s3_bucket,
            Prefix=fallback_prefix
        )

        # If files exist in the fallback location but not in the configured location
        if ('Contents' in fallback_response and
                ('Contents' not in s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_texts_prefix))):

            print(
                f"Warning: Found text files in incorrect location: {fallback_prefix}")
            print(
                f"Will process those instead of looking in: {s3_texts_prefix}")

            # Copy files to the correct location
            for item in fallback_response['Contents']:
                src_key = item['Key']
                if src_key.endswith('.txt'):
                    dest_key = f"{s3_texts_prefix}{os.path.basename(src_key)}"
                    print(f"Copying {src_key} to {dest_key}")

                    # Copy the object to the correct location
                    s3_client.copy_object(
                        CopySource={'Bucket': s3_bucket, 'Key': src_key},
                        Bucket=s3_bucket,
                        Key=dest_key
                    )

    # List all text files in the S3 path
    try:
        response = s3_client.list_objects_v2(
            Bucket=s3_bucket,
            Prefix=s3_texts_prefix
        )

        if 'Contents' not in response:
            print(
                f"No text files found in S3 at s3://{s3_bucket}/{s3_texts_prefix}")
            return None

        # Process each text file directly from S3
        for item in response['Contents']:
            s3_key = item['Key']
            filename = os.path.basename(s3_key)

            if not filename.endswith('.txt'):
                continue

            print(f"Processing S3 file: s3://{s3_bucket}/{s3_key}")

            # Download text content from S3
            try:
                s3_response = s3_client.get_object(
                    Bucket=s3_bucket, Key=s3_key)
                text = s3_response['Body'].read().decode('utf-8')

                # Create a chunk ID from the filename (without extension)
                chunk_id = Path(filename).stem

                # Count tokens
                token_count = len(cl_tokenizer.encode(text))

                # Select appropriate model
                selected_model = select_model_for_chunk(token_count)

                # Generate embedding
                embedding = generate_embedding(
                    text=text,
                    model_name=models[selected_model]["name"],
                    openai_client=openai
                )

                if embedding:
                    # Prepare data for storage
                    embedding_data = {
                        "chunk_id": chunk_id,
                        "chunk_text": text,
                        "model_used": models[selected_model]["name"],
                        "embedding": embedding,
                        "token_count": token_count,
                        "original_file": filename,
                        "source_s3_path": f"s3://{s3_bucket}/{s3_key}"
                    }

                    # Define S3 path for this embedding
                    s3_embedding_key = f"{chatbot_config['s3_path']}/embeddings/embedding_data.json"

                    # Upload embedding directly to S3
                    s3_client.put_object(
                        Bucket=s3_bucket,
                        Key=s3_embedding_key,
                        Body=json.dumps(embedding_data, ensure_ascii=False),
                        ContentType="application/json"
                    )

                    # Optionally save locally for backup/debugging
                    local_embedding_path = embeddings_dir / "embedding_data.json"
                    with open(local_embedding_path, 'w', encoding='utf-8') as f:
                        json.dump(embedding_data, f,
                                  ensure_ascii=False, indent=2)

                    # Add to index map
                    index_map.append({
                        "chunk_id": chunk_id,
                        "text_s3_path": f"s3://{s3_bucket}/{s3_key}",
                        "embedding_s3_path": f"s3://{s3_bucket}/{s3_embedding_key}",
                        "model_used": models[selected_model]["name"],
                        "token_count": token_count,
                        "local_embedding_path": str(local_embedding_path) if local_embedding_path.exists() else None
                    })

                    print(f"Processed and stored embedding for {chunk_id}")
                    print(
                        f"Embedding stored at S3 location: s3://{s3_bucket}/{s3_embedding_key}")
                    print(f"Local embedding path: {local_embedding_path}")
                    print(
                        f"This embedding will be retrieved using chunk_id: {chunk_id}")

            except Exception as e:
                print(f"Error processing S3 file {s3_key}: {str(e)}")
                continue

        # Save index map to S3
        s3_index_key = f"{chatbot_config['s3_path']}/index_map.json"
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=s3_index_key,
            Body=json.dumps(index_map, ensure_ascii=False),
            ContentType="application/json"
        )

        # Also save locally
        index_map_path = temp_dir / "index_map.json"
        with open(index_map_path, 'w', encoding='utf-8') as f:
            json.dump(index_map, f, ensure_ascii=False, indent=2)

        print(
            f"Saved index map with {len(index_map)} entries to S3: s3://{s3_bucket}/{s3_index_key}")
        return f"s3://{s3_bucket}/{s3_index_key}"

    except Exception as e:
        print(f"Error listing files in S3: {str(e)}")
        return None


def select_model_for_chunk(token_count: int) -> str:
    """
    Select the appropriate OpenAI embedding model based on document characteristics.

    We'll use a simple strategy:
    - Small texts (< 2000 tokens): text-embedding-3-small
    - Medium & larger texts: text-embedding-ada-002 (more cost effective)
    - For particularly important chunks: text-embedding-3-large

    For simplicity, we'll use token count as the main factor.
    """
    if token_count < 2000:
        return "small"  # text-embedding-3-small
    else:
        return "ada"  # text-embedding-ada-002 (more cost effective)

    # Large model is available but more expensive - could be used for specialized cases
    # return "large"  # text-embedding-3-large


def generate_embedding(
    text: str,
    model_name: str,
    openai_client: Any
) -> Optional[List[float]]:
    """
    Generate embedding for the given text using OpenAI's API.
    """
    try:
        response = openai_client.embeddings.create(
            model=model_name,
            input=text
        )
        return response.data[0].embedding

    except Exception as e:
        print(f"Error generating embedding with {model_name}: {str(e)}")
        return None


# Example usage:
if __name__ == "__main__":
    # Generate and store embeddings from text files in temp directory
    index_map_path = generate_and_store_embeddings()

    print(
        f"Embeddings generated and stored. Index map available at: {index_map_path}")
