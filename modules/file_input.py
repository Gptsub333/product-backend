import os
import tempfile
from typing import Optional, Union, List, Tuple
from pathlib import Path
import boto3
import json
from dotenv import load_dotenv


class DocumentParser:
    def __init__(self):
        # Don't initialize any models at startup - will initialize on-demand
        self.ocr_initialized = False
        self.ocr_reader = None

        # Load environment variables
        load_dotenv()

        # Get S3 configuration
        self.s3_bucket = os.getenv("S3_BUCKET_NAME")
        if self.s3_bucket:
            self.s3_enabled = True
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION", "us-east-1")
            )

            # Load chatbot configuration if available
            self.chatbot_config = None
            if os.path.exists(".chatbot_config"):
                with open(".chatbot_config", "r") as f:
                    self.chatbot_config = json.load(f)
        else:
            self.s3_enabled = False

        # Create temp directory if it doesn't exist
        # Use Path for cross-platform compatibility
        self.temp_dir = Path(os.path.dirname(__file__)) / "temp"
        self.temp_dir.mkdir(exist_ok=True)

    def parse_document(self, file_path: str, save_output: bool = False) -> str:
        """
        Parse text from various document formats (local or S3)

        Args:
            file_path: Path to the document file (local path or s3:// URL)
            save_output: Whether to save output to temp file and S3

        Returns:
            Extracted text from the document
        """
        # Check if file_path is an S3 URL
        if file_path.startswith("s3://"):
            # Extract bucket and key
            bucket, key = self._parse_s3_uri(file_path)
            # Download temp file and return path
            local_path = self._download_from_s3(bucket, key)
            try:
                # Process the temp file
                text = self._process_local_file(local_path)
                # Clean up
                os.unlink(local_path)
                # Save if requested
                if save_output:
                    return self.save_to_temp_file(text, os.path.basename(key))
                return text
            except Exception as e:
                # Clean up on error
                if os.path.exists(local_path):
                    os.unlink(local_path)
                raise e
        else:
            # Process local file
            return self._process_local_file(file_path, save_output)

    def _parse_s3_uri(self, s3_uri: str) -> Tuple[str, str]:
        """Parse an S3 URI into bucket and key"""
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")

        # Remove s3:// prefix
        s3_path = s3_uri[5:]

        # Split into bucket and key
        parts = s3_path.split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")

        bucket = parts[0]
        key = parts[1]

        return bucket, key

    def _download_from_s3(self, bucket: str, key: str) -> str:
        """Download file from S3 to a temporary location"""
        # Get file extension for temp file
        _, extension = os.path.splitext(key)

        # Create a temp file with the same extension
        temp_file = tempfile.NamedTemporaryFile(suffix=extension, delete=False)
        temp_path = temp_file.name
        temp_file.close()

        try:
            # Download the file from S3
            print(f"Downloading s3://{bucket}/{key} to {temp_path}")
            self.s3_client.download_file(
                Bucket=bucket,
                Key=key,
                Filename=temp_path
            )
            return temp_path
        except Exception as e:
            # Clean up the temp file if download fails
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise Exception(f"Error downloading from S3: {str(e)}")

    def _process_local_file(self, file_path: str, save_output: bool = False) -> str:
        """Process a local file and extract text"""
        # Convert to Path object for cross-platform handling
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file extension
        extension = file_path.suffix.lower()

        # Process based on file type
        if extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
            text = self.parse_image(str(file_path))
        elif extension == '.pdf':
            text = self.parse_pdf(str(file_path))
        elif extension == '.docx':
            text = self.parse_docx(str(file_path))
        elif extension in ['.txt', '.md', '.markdown']:
            text = self.parse_text_file(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {extension}")

        # Save to temp file if requested
        if save_output:
            return self.save_to_temp_file(text, file_path.name)

        return text

    def save_to_temp_file(self, text: str, original_filename: str) -> str:
        """
        Save extracted text primarily to S3 for concurrent access
        """
        # Create a unique filename based on the original name and timestamp
        import time
        basename = Path(original_filename).stem
        timestamp = int(time.time())
        temp_filename = f"{basename}_{timestamp}.txt"

        print(f"Saving file {temp_filename}")
        print(f"S3 enabled: {self.s3_enabled}")
        print(f"Chatbot config: {self.chatbot_config}")

        # If S3 is enabled and we have chatbot config, save directly to S3 first
        if self.s3_enabled and self.chatbot_config:
            # Use the correct S3 path from the config
            s3_key = f"{self.chatbot_config['s3_path']}/texts/{temp_filename}"

            try:
                # Save directly to S3 from memory
                print(
                    f"Attempting to save to S3: s3://{self.s3_bucket}/{s3_key}")
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=s3_key,
                    Body=text,
                    ContentType="text/plain"
                )
                s3_path = f"s3://{self.s3_bucket}/{s3_key}"
                print(f"Successfully uploaded text to S3: {s3_path}")

                # Also save locally for backup/debugging
                temp_path = self.temp_dir / temp_filename
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Also saved locally to {temp_path}")

                return s3_path  # Return the S3 path instead of local path
            except Exception as e:
                print(f"Error uploading to S3: {str(e)}")
                print(f"S3 bucket: {self.s3_bucket}")
                print(f"S3 key: {s3_key}")
                # Fall back to local storage on error
        else:
            # If S3 not enabled or failed, save locally
            temp_path = self.temp_dir / temp_filename
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(text)

        return str(temp_path)

    def _init_ocr(self):
        """Initialize EasyOCR on demand"""
        if not self.ocr_initialized:
            try:
                import easyocr
                # Only load the English language model to keep it lightweight
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
                self.ocr_initialized = True
            except ImportError:
                raise ImportError(
                    "easyocr is not installed. Install it with: pip install easyocr")

    def parse_image(self, image_path: str) -> str:
        """Extract text from an image using EasyOCR"""
        # Lazily initialize OCR only when needed
        self._init_ocr()

        # Use EasyOCR to recognize text
        results = self.ocr_reader.readtext(image_path)

        # Extract text from results
        text = ""
        for detection in results:
            text += detection[1] + "\n"

        return text

    def parse_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF document using PyMuPDF"""
        # Import fitz only when needed
        import fitz

        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
        except Exception as e:
            raise Exception(f"Error parsing PDF: {str(e)}")
        return text

    def parse_docx(self, docx_path: str) -> str:
        """Extract text from a DOCX document"""
        # Import docx only when needed
        import docx

        try:
            doc = docx.Document(docx_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            raise Exception(f"Error parsing DOCX: {str(e)}")

    def parse_text_file(self, file_path: str) -> str:
        """Extract text from a plain text file (TXT or MD)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Error parsing text file: {str(e)}")


# Example usage
if __name__ == "__main__":
    parser = DocumentParser()

    # Example: Parse a document
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        try:
            text = parser.parse_document(file_path, save_output=True)
            print(f"Extracted text from {file_path}:")
            print("-" * 50)
            print(text[:500] + "..." if len(text) >
                  500 else text)  # Show preview of text
            print("-" * 50)

            # Show where the output was saved
            if file_path.startswith("s3://") or (text and text.startswith("s3://")):
                print(f"Full text saved to S3")
            else:
                print(f"Full text saved to temp directory")
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("Usage: python document_parser.py <file_path or s3://bucket/key>")
