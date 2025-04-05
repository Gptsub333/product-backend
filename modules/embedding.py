import json
import os
import tiktoken
import openai
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pathlib import Path


def generate_and_store_embeddings(
    temp_dir: str = None
) -> str:
    """
    Generate embeddings for document chunks stored in temp directory using only OpenAI models
    and store them locally in the same directory.

    Args:
        temp_dir: Path to temp directory (defaults to "temp" in modules folder)

    Returns:
        Path to the index map JSON file
    """
    # Load environment variables
    load_dotenv()

    # Initialize OpenAI client
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("API key for OpenAI not found in .env file")

    # Use this updated initialization
    openai.api_key = openai_api_key  # Set the API key as a module-level variable

    # Set the temp directory if not provided, using Path for cross-platform compatibility
    if temp_dir is None:
        temp_dir = Path(os.path.dirname(__file__)) / "temp"
    else:
        temp_dir = Path(temp_dir)

    # Ensure temp directory exists
    temp_dir.mkdir(exist_ok=True)

    # Create embeddings directory within temp
    embeddings_dir = temp_dir / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)

    # Initialize tokenizer for counting tokens
    cl_tokenizer = tiktoken.get_encoding("cl100k_base")

    # Define models and their token limits (all OpenAI)
    models = {
        "small": {
            "name": "text-embedding-3-small",
            "max_tokens": 8191,
            "dimension": 1536
        },
        "large": {
            "name": "text-embedding-3-large",
            "max_tokens": 8191,
            "dimension": 3072
        },
        "ada": {
            "name": "text-embedding-ada-002",
            "max_tokens": 8191,
            "dimension": 1536
        }
    }

    # Create index map
    index_map = []

    # Get all text files in the temp directory
    txt_files = [f for f in os.listdir(temp_dir) if f.endswith('.txt')]

    # Process each text file as a chunk
    for i, txt_file in enumerate(txt_files):
        file_path = temp_dir / txt_file

        # Read text content
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Create a chunk ID from the filename (without extension)
        chunk_id = Path(txt_file).stem

        # Count tokens
        token_count = len(cl_tokenizer.encode(text))

        # Select appropriate model based on document length
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
                "original_file": txt_file
            }

            # Define local path for this chunk embedding
            embedding_file_path = embeddings_dir / f"{chunk_id}_embedding.json"

            # Write to local file
            with open(embedding_file_path, 'w', encoding='utf-8') as f:
                json.dump(embedding_data, f, ensure_ascii=False, indent=2)

            # Add to index map
            index_map.append({
                "chunk_id": chunk_id,
                "embedding_path": str(embedding_file_path),
                "model_used": models[selected_model]["name"],
                "token_count": token_count,
                "original_file": txt_file
            })

    # Save index map locally
    index_map_path = temp_dir / "index_map.json"
    with open(index_map_path, 'w', encoding='utf-8') as f:
        json.dump(index_map, f, ensure_ascii=False, indent=2)

    return str(index_map_path)


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
