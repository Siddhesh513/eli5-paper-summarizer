"""
PDF Processing Module.
Handles arXiv paper fetching and text extraction with section detection.
"""
import re
import tempfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import arxiv
import fitz  # PyMuPDF
import requests


@dataclass
class PaperContent:
    """Container for extracted paper content."""
    title: str
    authors: list[str]
    abstract: str
    full_text: str
    sections: dict[str, str]
    pdf_path: Optional[str] = None


def extract_arxiv_id(url_or_id: str) -> str:
    """
    Extract arXiv ID from various URL formats or direct ID.
    
    Args:
        url_or_id: arXiv URL or paper ID (e.g., "2301.00001" or "https://arxiv.org/abs/2301.00001")
    
    Returns:
        Clean arXiv ID string
    
    Raises:
        ValueError: If no valid arXiv ID found
    """
    # Common patterns
    patterns = [
        r"arxiv\.org/abs/(\d+\.\d+)",
        r"arxiv\.org/pdf/(\d+\.\d+)",
        r"^(\d+\.\d+)(?:v\d+)?$",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    
    raise ValueError(f"Could not extract arXiv ID from: {url_or_id}")


def fetch_arxiv_pdf(url_or_id: str) -> tuple[str, dict]:
    """
    Fetch PDF from arXiv and return path with metadata.
    
    Args:
        url_or_id: arXiv URL or paper ID
    
    Returns:
        Tuple of (pdf_path, metadata_dict)
    """
    arxiv_id = extract_arxiv_id(url_or_id)
    
    # Fetch metadata using arxiv library
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(search.results())
    
    # Download PDF to temp file
    temp_dir = tempfile.mkdtemp()
    pdf_path = paper.download_pdf(dirpath=temp_dir)
    
    metadata = {
        "title": paper.title,
        "authors": [author.name for author in paper.authors],
        "abstract": paper.summary,
        "published": paper.published.isoformat(),
        "arxiv_id": arxiv_id,
        "url": paper.entry_id,
    }
    
    return str(pdf_path), metadata


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF using PyMuPDF.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Extracted text as string
    """
    doc = fitz.open(pdf_path)
    text_parts = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        text_parts.append(text)
    
    doc.close()
    return "\n\n".join(text_parts)


def detect_sections(text: str) -> dict[str, str]:
    """
    Detect and extract common paper sections.
    
    Args:
        text: Full paper text
    
    Returns:
        Dictionary mapping section names to content
    """
    # Common section headers (case-insensitive)
    section_patterns = [
        (r"(?i)^(?:1\.?\s*)?introduction\s*$", "Introduction"),
        (r"(?i)^(?:2\.?\s*)?(?:related\s*work|background)\s*$", "Related Work"),
        (r"(?i)^(?:\d\.?\s*)?(?:method(?:ology)?|approach|model)\s*$", "Methods"),
        (r"(?i)^(?:\d\.?\s*)?(?:experiment(?:s)?|evaluation)\s*$", "Experiments"),
        (r"(?i)^(?:\d\.?\s*)?(?:result(?:s)?|finding(?:s)?)\s*$", "Results"),
        (r"(?i)^(?:\d\.?\s*)?(?:discussion)\s*$", "Discussion"),
        (r"(?i)^(?:\d\.?\s*)?(?:conclusion(?:s)?|summary)\s*$", "Conclusion"),
        (r"(?i)^(?:abstract)\s*$", "Abstract"),
    ]
    
    lines = text.split("\n")
    sections = {}
    current_section = "Preamble"
    current_content = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check if line matches any section header
        matched = False
        for pattern, section_name in section_patterns:
            if re.match(pattern, line_stripped):
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = section_name
                current_content = []
                matched = True
                break
        
        if not matched:
            current_content.append(line)
    
    # Save last section
    if current_content:
        sections[current_section] = "\n".join(current_content).strip()
    
    return sections


def process_paper(url_or_id: str) -> PaperContent:
    """
    Complete pipeline: fetch, extract, and structure paper content.
    
    Args:
        url_or_id: arXiv URL or paper ID
    
    Returns:
        PaperContent object with all extracted information
    """
    # Fetch PDF and metadata
    pdf_path, metadata = fetch_arxiv_pdf(url_or_id)
    
    # Extract text
    full_text = extract_text_from_pdf(pdf_path)
    
    # Detect sections
    sections = detect_sections(full_text)
    
    return PaperContent(
        title=metadata["title"],
        authors=metadata["authors"],
        abstract=metadata["abstract"],
        full_text=full_text,
        sections=sections,
        pdf_path=pdf_path,
    )


def process_uploaded_pdf(pdf_file) -> PaperContent:
    """
    Process an uploaded PDF file (from Streamlit).
    
    Args:
        pdf_file: Streamlit UploadedFile object
    
    Returns:
        PaperContent object
    """
    # Save to temp file
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / pdf_file.name
    
    with open(temp_path, "wb") as f:
        f.write(pdf_file.getbuffer())
    
    # Extract text
    full_text = extract_text_from_pdf(str(temp_path))
    sections = detect_sections(full_text)
    
    # Try to extract title from first lines
    lines = full_text.split("\n")[:10]
    title = next((l.strip() for l in lines if len(l.strip()) > 10), "Uploaded Paper")
    
    return PaperContent(
        title=title,
        authors=[],
        abstract=sections.get("Abstract", ""),
        full_text=full_text,
        sections=sections,
        pdf_path=str(temp_path),
    )


if __name__ == "__main__":
    # Quick test
    test_url = "https://arxiv.org/abs/2301.00001"
    try:
        paper = process_paper(test_url)
        print(f"Title: {paper.title}")
        print(f"Authors: {', '.join(paper.authors[:3])}...")
        print(f"Sections found: {list(paper.sections.keys())}")
        print(f"Total text length: {len(paper.full_text)} chars")
    except Exception as e:
        print(f"Test failed: {e}")
