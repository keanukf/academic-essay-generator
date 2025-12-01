"""PDF text extraction and chunking utilities."""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List
import re


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract all text from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text as a single string
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: For PDF parsing errors
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text_parts.append(text)
        
        doc.close()
        return "\n\n".join(text_parts)
    except Exception as e:
        raise Exception(f"Error extracting text from PDF {pdf_path}: {str(e)}")


def estimate_tokens(text: str) -> int:
    """
    Estimate token count (approximately 4 characters per token).
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    token_estimator=None
) -> List[str]:
    """
    Split text into chunks with overlap.
    
    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap size in tokens
        token_estimator: Function to estimate tokens (default: estimate_tokens)
        
    Returns:
        List of text chunks
    """
    if token_estimator is None:
        token_estimator = estimate_tokens
    
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text:
        return []
    
    chunks = []
    current_pos = 0
    text_length = len(text)
    
    # Estimate characters per token (approximately 4)
    chars_per_token = 4
    chunk_char_size = chunk_size * chars_per_token
    overlap_char_size = chunk_overlap * chars_per_token
    
    while current_pos < text_length:
        # Calculate end position
        end_pos = min(current_pos + chunk_char_size, text_length)
        
        # Extract chunk
        chunk = text[current_pos:end_pos]
        
        # Try to break at sentence boundary if not at end
        if end_pos < text_length:
            # Look for sentence endings within last 20% of chunk
            search_start = max(current_pos, end_pos - int(chunk_char_size * 0.2))
            last_period = chunk.rfind('.', search_start - current_pos)
            last_newline = chunk.rfind('\n', search_start - current_pos)
            
            # Prefer period, then newline
            break_point = max(last_period, last_newline)
            if break_point > chunk_char_size * 0.5:  # Only if reasonable
                chunk = chunk[:break_point + 1]
                end_pos = current_pos + break_point + 1
        
        chunks.append(chunk.strip())
        
        # Move position forward with overlap
        if end_pos >= text_length:
            break
        current_pos = end_pos - overlap_char_size
    
    return chunks


def load_pdfs_from_directory(directory: Path, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """
    Load all PDFs from a directory and return chunked text.
    
    Args:
        directory: Directory containing PDF files
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap size in tokens
        
    Returns:
        List of text chunks from all PDFs
        
    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    all_chunks = []
    pdf_files = list(directory.glob("*.pdf"))
    
    if not pdf_files:
        print(f"Warning: No PDF files found in {directory}")
        return []
    
    for pdf_path in pdf_files:
        try:
            print(f"Loading PDF: {pdf_path.name}")
            text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(text, chunk_size, chunk_overlap)
            all_chunks.extend(chunks)
            print(f"  Extracted {len(chunks)} chunks from {pdf_path.name}")
        except Exception as e:
            print(f"  Error processing {pdf_path.name}: {str(e)}")
            continue
    
    return all_chunks

