import tiktoken
from typing import List, Dict
import re


class TextChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def split_text(self, text: str) -> List[Dict[str, any]]:
        """Split text into chunks with metadata"""

        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text).strip()

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_size = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)

            # If adding this sentence would exceed chunk size
            if current_size + sentence_tokens > self.chunk_size and current_chunk:
                # Store current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'token_count': self.count_tokens(chunk_text),
                    'char_count': len(chunk_text),
                    'start_sentence': i - len(current_chunk),
                    'end_sentence': i - 1
                })

                # Start new chunk with overlap
                # Only take the last few sentences based on overlap size
                overlap_tokens = 0
                overlap_sentences = []
                for sent in reversed(current_chunk):
                    sent_tokens = self.count_tokens(sent)
                    if overlap_tokens + sent_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break

                current_chunk = overlap_sentences
                current_size = sum(self.count_tokens(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_size += sentence_tokens

        # Add final chunk if it exists
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'token_count': self.count_tokens(chunk_text),
                'char_count': len(chunk_text),
                'start_sentence': len(sentences) - len(current_chunk),
                'end_sentence': len(sentences) - 1
            })

        print(f"Split document into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(
                f"Chunk {i}: {chunk['token_count']} tokens, {chunk['char_count']} chars")

        return chunks
