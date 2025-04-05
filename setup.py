# chatbot_manager.py (place in root directory)

# import .modules.embedding as embedding_module
from .modules import embedding as embedding_module
from .modules.file_input import DocumentParser
import os
import boto3
import json
import sys
import datetime
import openai
from pathlib import Path
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv

# Adjust the Python path to include the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ChatbotManager:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Get S3 configuration
        self.s3_bucket = os.getenv("S3_BUCKET_NAME")
        if not self.s3_bucket:
            raise ValueError("S3_BUCKET_NAME not found in .env file")

        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )

        # Local temp directory for processing
        self.temp_dir = Path("product-backend/modules/temp")
        self.temp_dir.mkdir(exist_ok=True, parents=True)

        # Initialize parser
        self.parser = DocumentParser()

    def user_exists(self, username: str) -> bool:
        """Check if a user already exists in S3 bucket"""
        try:
            # Check if user directory exists in S3
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=f"users/{username}/",
                MaxKeys=1
            )
            return 'Contents' in response
        except Exception as e:
            print(f"Error checking if user exists: {str(e)}")
            return False

    def create_user(self, username: str) -> bool:
        """Create a new user directory in S3"""
        try:
            # Create empty placeholder file to represent user directory
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=f"users/{username}/.metadata",
                Body=json.dumps({
                    "created_at": str(datetime.datetime.now()),
                    "username": username
                })
            )
            print(f"Created user: {username}")
            return True
        except Exception as e:
            print(f"Error creating user: {str(e)}")
            return False

    def create_chatbot(self, username: str, chatbot_name: str) -> str:
        """Create a new chatbot for a user"""
        # Check if user exists, create if not
        if not self.user_exists(username):
            if not self.create_user(username):
                raise Exception(f"Failed to create user: {username}")

        # Create chatbot directory in S3
        chatbot_path = f"users/{username}/chatbots/{chatbot_name}"
        try:
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=f"{chatbot_path}/.metadata",
                Body=json.dumps({
                    "created_at": str(datetime.datetime.now()),
                    "chatbot_name": chatbot_name
                })
            )
            print(f"Created chatbot: {chatbot_name} for user: {username}")
            return chatbot_path
        except Exception as e:
            print(f"Error creating chatbot: {str(e)}")
            return None

    def process_document_for_chatbot(self, username: str, chatbot_name: str, file_path: str) -> dict:
        """Process a document and upload to S3 for a specific chatbot"""
        # Create chatbot if needed
        chatbot_path = self.create_chatbot(username, chatbot_name)

        # Save the chatbot path for file_input.py and embedding.py to use
        with open(".chatbot_config", "w") as f:
            json.dump({
                "username": username,
                "chatbot_name": chatbot_name,
                "s3_bucket": self.s3_bucket,
                "s3_path": chatbot_path
            }, f)

        # Parse document - this now saves directly to S3
        parsed_text = self.parser.parse_document(file_path, save_output=True)

        # Generate embeddings from S3 text files
        index_map_path = embedding_module.generate_and_store_embeddings()

        # Handle empty return value gracefully
        if not index_map_path:
            print(
                "Warning: No embeddings were generated. The document may be empty or there was an error.")
            index_map = []
        else:
            # Get index map from path
            if index_map_path.startswith("s3://"):
                # Extract bucket and key from s3://bucket/key format
                s3_parts = index_map_path.replace("s3://", "").split("/", 1)
                s3_bucket = s3_parts[0]
                s3_key = s3_parts[1]

                # Get index map from S3
                response = self.s3_client.get_object(
                    Bucket=s3_bucket, Key=s3_key)
                index_map = json.loads(response['Body'].read().decode('utf-8'))
            else:
                # If the index map path is a local file
                if os.path.exists(index_map_path):
                    with open(index_map_path, 'r') as f:
                        index_map = json.load(f)
                else:
                    index_map = []

        return {
            "username": username,
            "chatbot_name": chatbot_name,
            "document_processed": os.path.basename(file_path),
            "chunks_processed": len(index_map),
            "index_map_location": index_map_path
        }

    def get_chatbot_data(self, username: str, chatbot_name: str) -> Dict:
        """Get chatbot data from S3"""
        chatbot_path = f"users/{username}/chatbots/{chatbot_name}"

        try:
            # Get index map
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key=f"{chatbot_path}/index_map.json"
            )
            index_map = json.loads(response['Body'].read().decode('utf-8'))

            return {
                "username": username,
                "chatbot_name": chatbot_name,
                "index_map": index_map
            }
        except Exception as e:
            raise Exception(f"Error retrieving chatbot data: {str(e)}")

    def query_with_bedrock(self,
                           username: str,
                           chatbot_name: str,
                           query: str,
                           model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0") -> str:
        """Query chatbot using Bedrock with RAG from S3"""
        # Get chatbot data
        chatbot_data = self.get_chatbot_data(username, chatbot_name)
        index_map = chatbot_data["index_map"]

        # Download embeddings from S3 to temp
        for item in index_map:
            embedding_s3_path = item["embedding_s3_path"]
            # Extract key from s3://bucket/key format
            embedding_key = embedding_s3_path.split(
                f"s3://{self.s3_bucket}/")[1]
            local_path = self.temp_dir / \
                f"temp_{os.path.basename(embedding_key)}"

            # Download embedding file
            self.s3_client.download_file(
                Bucket=self.s3_bucket,
                Key=embedding_key,
                Filename=str(local_path)
            )
            item["local_embedding_path"] = str(local_path)

        # Generate embedding for query
        openai.api_key = os.getenv("OPENAI_API_KEY")
        query_embedding = self.generate_query_embedding(query)

        # Find most similar documents
        relevant_chunks = self.find_relevant_chunks(query_embedding, index_map)
        context = "\n\n---\n\n".join([chunk["text"]
                                     for chunk in relevant_chunks])

        # Query Bedrock
        return self.query_bedrock(query, context, model_id)

    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a query"""
        # Initialize OpenAI client
        openai.api_key = os.getenv("OPENAI_API_KEY")

        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        return response.data[0].embedding

    def find_relevant_chunks(self, query_embedding: List[float], index_map: List[Dict], top_k: int = 3) -> List[Dict]:
        """Find most relevant chunks for a query embedding"""
        results = []

        for item in index_map:
            local_path = item["local_embedding_path"]

            # Load embedding data
            with open(local_path, 'r') as f:
                embedding_data = json.load(f)

            # Calculate similarity
            similarity = self.cosine_similarity(
                query_embedding, embedding_data["embedding"])

            results.append({
                "chunk_id": item["chunk_id"],
                "similarity": similarity,
                "text": embedding_data["chunk_text"]
            })

        # Sort by similarity and return top k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def query_bedrock(self, query: str, context: str, model_id: str) -> str:
        """Query AWS Bedrock with context"""
        import boto3
        import json

        bedrock = boto3.client('bedrock-runtime')

        prompt = f"""
Human: I need information about the following topic:

{query}

Here is some relevant information that might help:

{context}

Please use this information to provide a comprehensive answer.
"""

        try:
            response = bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
            )

            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
        except Exception as e:
            return f"Error querying Bedrock: {str(e)}"


def main():
    if len(sys.argv) < 3:
        print("Usage: python app.py <username> <chatbot_name> [<file_path>]")
        return

    username = sys.argv[1]
    chatbot_name = sys.argv[2]

    try:
        manager = ChatbotManager()

        # If only creating the chatbot structure
        if len(sys.argv) == 3:
            chatbot_path = manager.create_chatbot(username, chatbot_name)
            if chatbot_path:
                print(
                    f"Chatbot created successfully at s3://{manager.s3_bucket}/{chatbot_path}")
            else:
                print(f"Failed to create chatbot: {chatbot_name}")

        # If also processing a document
        elif len(sys.argv) == 4:
            file_path = sys.argv[3]
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return

            result = manager.process_document_for_chatbot(
                username, chatbot_name, file_path)
            print(f"Document processed: {result['document_processed']}")
            print(f"Chunks processed: {result['chunks_processed']}")
            print(f"Index map location: {result['index_map_location']}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
