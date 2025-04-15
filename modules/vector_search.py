import boto3
import json
import numpy as np
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
from .text_chunking import TextChunker
import time

load_dotenv()


class VectorSearch:
    def __init__(self):
        self.SIMILARITY_THRESHOLD = 0.01  # Lowered threshold since scores are very low
        self.MAX_CHUNKS = 3  # Reduced to avoid noise
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name="us-east-2",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        self.s3_client = boto3.client('s3')

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text with rate limiting and retries"""
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                response = self.bedrock.invoke_model(
                    modelId="amazon.titan-embed-text-v2:0",
                    body=json.dumps({
                        "inputText": text.strip()
                    }),
                    contentType="application/json",
                    accept="application/json"
                )

                return json.loads(response['body'].read())['embedding']

            except Exception as e:
                print(f"Embedding attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print("All embedding attempts failed")
                    return None

    def search_similar_chunks(self, query: str, user_id: str, bot_name: str) -> List[Dict]:
        try:
            print(f"\n=== Starting vector search for query: {query} ===")

            # For general questions about the paper, modify the query
            if query.lower().strip() in ["what is the paper about", "what is this paper about", "summarize the paper"]:
                print(
                    "General paper question detected - prioritizing abstract and introduction")
                # Boost scores for first chunks containing abstract/intro
                chunk_boost = 0.05
            else:
                chunk_boost = 0

            # Get query embedding
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                raise Exception("Failed to generate query embedding")

            # Load index map
            index_map_key = f"{user_id}/{bot_name}/index_map.json"
            try:
                print(f"Looking for index map at: {index_map_key}")
                index_map = json.loads(
                    self.s3_client.get_object(
                        Bucket=os.getenv("S3_BUCKET_NAME"),
                        Key=index_map_key
                    )['Body'].read()
                )
                print(f"Loaded index map with {len(index_map)} chunks")

                # Process chunks and calculate similarities
                similar_chunks = []
                for chunk_info in index_map:
                    # Load chunk data
                    chunk_data = json.loads(
                        self.s3_client.get_object(
                            Bucket=os.getenv("S3_BUCKET_NAME"),
                            Key=chunk_info['s3_path'].split(
                                's3://' + os.getenv("S3_BUCKET_NAME") + '/')[1]
                        )['Body'].read()
                    )

                    base_similarity = self.cosine_similarity(
                        query_embedding, chunk_data['embedding'])

                    # Apply boost for early chunks (abstract/intro) for general questions
                    chunk_index = chunk_data.get('chunk_index', 0)
                    if chunk_index < 2:  # Boost first two chunks
                        similarity = base_similarity + chunk_boost
                    else:
                        similarity = base_similarity

                    print(
                        f"Chunk {chunk_data['chunk_id']}: base_similarity = {base_similarity:.4f}, final_similarity = {similarity:.4f}")

                    if similarity > self.SIMILARITY_THRESHOLD:
                        similar_chunks.append({
                            'text': chunk_data['chunk_text'],
                            'similarity': similarity,
                            'source': chunk_info.get('source_file', 'unknown'),
                            'chunk_index': chunk_index
                        })

                # Sort by similarity and select top chunks
                similar_chunks.sort(
                    key=lambda x: x['similarity'], reverse=True)
                selected_chunks = similar_chunks[:self.MAX_CHUNKS]

                print(f"\n=== Search Results ===")
                print(f"Found {len(similar_chunks)} chunks above threshold")
                print(f"Selected top {len(selected_chunks)} chunks")
                for chunk in selected_chunks:
                    print(
                        f"Chunk {chunk['chunk_index']}: similarity {chunk['similarity']:.4f}")

                return selected_chunks

            except Exception as e:
                print(f"Error processing chunks: {str(e)}")
                return []

        except Exception as e:
            print(f"Error in vector search: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def verify_index_map(self, user_id: str, bot_name: str) -> bool:
        """Verify that the index map exists and is accessible"""
        try:
            paths_to_try = [
                f"users/{user_id}/chatbots/{bot_name}/index_map.json",
                f"{user_id}/{bot_name}/index_map.json"
            ]

            for path in paths_to_try:
                try:
                    self.s3_client.head_object(
                        Bucket=os.getenv("S3_BUCKET_NAME"),
                        Key=path
                    )
                    print(f"Found index map at: {path}")
                    return True
                except self.s3_client.exceptions.ClientError:
                    continue

            print(f"No index map found for user {user_id} and bot {bot_name}")
            print(
                "Please ensure you have processed documents first using the document upload endpoint")
            return False

        except Exception as e:
            print(f"Error verifying index map: {str(e)}")
            return False
